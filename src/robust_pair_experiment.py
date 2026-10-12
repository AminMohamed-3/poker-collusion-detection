"""Four fixed robustness candidates, or one explicit all-label final refit.

Run from work/poker: .venv/Scripts/python.exe src/robust_pair_experiment.py
Add --fit-final to save outputs_v2/pair_model_robust.joblib separately.
No submission or existing baseline/v2 model/feature file is replaced. Cached
hand-level features stay fixed: thinning does not recompute player baselines.
"""
from __future__ import annotations

import os

for variable in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '2'

import argparse
import gc
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import joblib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA, SOURCE = ROOT / 'data', ROOT / 'outputs_v2'
REPORT = ROOT / 'reports' / 'robust-pair-experiment'
FAMILIES = ['none', 'directed_transfer', 'soft_play', 'coordinated_isolation']
LABELS = {name: i for i, name in enumerate(FAMILIES)}
KEYS = ['pair_id', 'hand_id', 'table_id']
SPLIT_SEED = 2901
TRAIN_VIEWS = [(0.7, 7307), (0.4, 7304)]
HELDOUT_VIEW = (0.65, 7365)
MODEL_PARAMS = dict(n_estimators=226, learning_rate=.03, num_leaves=15,
                    max_depth=-1, min_child_samples=20, colsample_bytree=.8,
                    reg_lambda=5, n_jobs=2, verbosity=-1, random_state=7329,
                    deterministic=True, force_col_wise=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def thin(frame, fraction, seed):
    """Keep floor(fraction*n), at least one, using label-independent hand hashes."""
    if not 0 < fraction <= 1:
        raise ValueError('fraction must lie in (0, 1]')
    order = frame.with_columns(pl.col('hand_id').hash(seed=seed).alias('_sample_key'))
    order = order.sort('pair_id', 'table_id', '_sample_key', 'hand_id')
    order = order.with_columns(
        pl.col('hand_id').cum_count().over('pair_id', 'table_id').alias('_position'),
        (pl.len().over('pair_id', 'table_id') * fraction).floor()
        .cast(pl.UInt32).clip(lower_bound=1).alias('_keep'))
    return order.filter(pl.col('_position') <= pl.col('_keep')).select(frame.columns)


def aggregate(frame, raw_features):
    # Match advanced_features.py's aggregations and column order exactly.
    return frame.group_by('pair_id', 'table_id').agg(
        pl.len().alias('shared_hands'),
        *[pl.col(c).mean().alias(c + '_mean') for c in raw_features],
        *[pl.col(c).std().fill_null(0).alias(c + '_std') for c in raw_features],
        *[pl.col(c).max().alias(c + '_max') for c in raw_features],
        *[pl.col(c).quantile(.95).alias(c + '_q95') for c in raw_features],
    ).sort('pair_id')


def matrix(frame, features):
    values = frame.select(features).to_numpy().astype(np.float32)
    return np.nan_to_num(values, nan=0, posinf=0, neginf=0)


def stable_ap(truth, score, weights=None):
    """Input rows must already be in pair_id order, as in the official metric."""
    truth = np.asarray(truth, dtype=np.float64)
    weights = np.ones(len(truth)) if weights is None else np.asarray(weights)
    order = np.argsort(-np.asarray(score), kind='mergesort')
    positive_weight = truth[order] * weights[order]
    if not positive_weight.sum():
        return 0.
    precision = np.cumsum(positive_weight) / np.cumsum(weights[order])
    return float(np.sum(precision * positive_weight) / positive_weight.sum())


def metrics(truth, risk):
    target = np.asarray(truth) > 0
    prevalence = float(target.mean())
    weights = np.where(target, .01 / prevalence, .99 / (1 - prevalence))
    return {'pair_ap': stable_ap(target, risk),
            'sklearn_pair_ap': float(average_precision_score(target, risk)),
            'pair_ap_at_hypothetical_1pct_prevalence': stable_ap(target, risk, weights),
            'observed_positive_prevalence': prevalence}


def assert_disjoint(train, held):
    for key in ['pair_id', 'table_id']:
        if set(train[key]) & set(held[key]):
            raise ValueError(f'Training/heldout overlap in {key}')
    train_players = set(train['player_1']) | set(train['player_2'])
    held_players = set(held['player_1']) | set(held['player_2'])
    if train_players & held_players:
        raise ValueError('Training/heldout player overlap')


def peak_memory_bytes():
    """Read the current process's OS peak resident memory, including native libs."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]

        kernel, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError('Cannot read process memory counters')
        return int(counters.PeakWorkingSetSize)
    import resource
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if platform.system() == 'Darwin' else maximum * 1024)


def self_test():
    raw = pl.DataFrame({'pair_id': ['a'] * 5 + ['b'] * 5,
                        'table_id': ['t1'] * 5 + ['t2'] * 5,
                        'hand_id': [f'h{i}' for i in range(10)],
                        'value': [float(i) for i in range(10)]})
    for fraction, seed in TRAIN_VIEWS + [HELDOUT_VIEW]:
        selected = thin(raw, fraction, seed)
        reversed_selection = thin(raw.reverse(), fraction, seed)
        assert selected.equals(reversed_selection)
        assert selected.group_by('pair_id').len()['len'].to_list() == [int(5 * fraction)] * 2
        summary = aggregate(selected, ['value'])
        for row in summary.iter_rows(named=True):
            values = selected.filter(pl.col('pair_id') == row['pair_id'])['value'].to_numpy()
            assert row['shared_hands'] == len(values)
            assert np.isclose(row['value_mean'], values.mean())
            assert np.isclose(row['value_std'], values.std(ddof=1))
        assert set(summary.columns) == {'pair_id', 'table_id', 'shared_hands',
                                        'value_mean', 'value_std', 'value_max', 'value_q95'}
    assert np.isclose(stable_ap([1, 0, 1], [.5, .5, .5]), 5 / 6)
    assert stable_ap([1, 0, 1], [.9, .8, .7], [.01, .99, .01]) < stable_ap([1, 0, 1], [.9, .8, .7])
    example = pl.DataFrame({'pair_id': ['a'], 'table_id': ['t1'],
                            'player_1': ['p1'], 'player_2': ['p2']})
    try:
        assert_disjoint(example, example)
    except ValueError:
        pass
    else:
        raise AssertionError('Leakage guard accepted overlapping partitions')


def run():
    started = time.perf_counter()
    self_test()
    paths = sorted((SOURCE / 'hands_development').glob('batch_*.parquet'))
    if not paths:
        raise ValueError('No v2 development hand caches found')
    inputs = [DATA / 'development_labels.csv', SOURCE / 'pairs_development.parquet',
              SOURCE / 'validation_pairs.csv', SOURCE / 'hands_development' / 'manifest.json',
              ROOT / 'src' / 'train.py', ROOT / 'src' / 'advanced_features.py', *paths]
    fingerprints = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    labels = pl.read_csv(DATA / 'development_labels.csv')
    if labels['pair_id'].n_unique() != labels.height:
        raise ValueError('Duplicate development labels')
    if labels.filter(~pl.col('label_status').is_in(['confirmed_target', 'confirmed_non_target'])).height:
        raise ValueError('Only explicit confirmed labels may enter this experiment')
    full = pl.read_parquet(SOURCE / 'pairs_development.parquet').join(labels, on='pair_id').sort('pair_id')
    if full.height != labels.height or full['pair_id'].n_unique() != labels.height:
        raise ValueError('Pair summaries do not cover the explicit labels once each')
    features = [c for c in full.columns if c not in KEYS +
                ['player_1', 'player_2', 'label', 'label_status', 'behavior_family']]
    target = np.array([LABELS[name] for name in full['behavior_family']])
    if not np.array_equal(target > 0, full['label'].to_numpy() == 1):
        raise ValueError('Label and behavior family disagree')
    train_indices, held_indices = next(GroupShuffleSplit(
        n_splits=1, test_size=.25, random_state=SPLIT_SEED).split(full, target, full['table_id']))
    training, held = full[train_indices.tolist()], full[held_indices.tolist()]
    assert_disjoint(training, held)
    saved_held = set(pl.read_csv(SOURCE / 'validation_pairs.csv')['pair_id'])
    if set(held['pair_id']) != saved_held:
        raise ValueError('Fixed fold does not match saved v2 fold 0 validation membership')
    train_tables, held_tables = set(training['table_id']), set(held['table_id'])
    expected_keys = full.select('pair_id', 'table_id')
    views = {f'train_{fraction}': [] for fraction, _ in TRAIN_VIEWS}
    views['held_0.65'] = []
    reference_counts = []
    raw_feature_names = None
    for path in paths:
        raw = pl.read_parquet(path)
        if raw.select('pair_id', 'table_id').unique().join(
                expected_keys, on=['pair_id', 'table_id'], how='anti').height:
            raise ValueError('Raw cache includes a pair outside the explicit labeled scope')
        names = [c for c in raw.columns if c not in KEYS]
        if raw_feature_names is not None and names != raw_feature_names:
            raise ValueError('Raw cache schemas differ')
        raw_feature_names = names
        reference_counts.append(raw.group_by('pair_id', 'table_id').len())
        train_raw = raw.filter(pl.col('table_id').is_in(list(train_tables)))
        held_raw = raw.filter(pl.col('table_id').is_in(list(held_tables)))
        for fraction, seed in TRAIN_VIEWS:
            if train_raw.height:
                views[f'train_{fraction}'].append(aggregate(thin(train_raw, fraction, seed), names))
        if held_raw.height:
            views['held_0.65'].append(aggregate(thin(held_raw, *HELDOUT_VIEW), names))
        print(f'aggregated {path.name}', flush=True)
        del raw, train_raw, held_raw
        gc.collect()
    counts = pl.concat(reference_counts).group_by('pair_id', 'table_id').agg(pl.col('len').sum())
    expected = full.select('pair_id', 'table_id', 'shared_hands').join(counts, on=['pair_id', 'table_id'])
    if expected.height != full.height or expected.filter(pl.col('len') != pl.col('shared_hands')).height:
        raise ValueError('Raw cache counts do not match the v2 pair summaries')
    assembled = {}
    for name, pieces in views.items():
        frame = pl.concat(pieces).join(labels, on='pair_id').sort('pair_id')
        reference = held if name.startswith('held') else training
        fraction = float(name.split('_')[1])
        if frame.height != reference.height or set(frame['pair_id']) != set(reference['pair_id']):
            raise ValueError(f'Missing or duplicate pairs after thinning {name}')
        exact = frame.select('pair_id', 'shared_hands').join(reference.select(
            'pair_id', (pl.col('shared_hands') * fraction).floor().cast(pl.UInt32).clip(lower_bound=1)
            .alias('expected')), on='pair_id')
        if exact.filter(pl.col('shared_hands') != pl.col('expected')).height:
            raise ValueError('A pair spans multiple batches or has an incorrect sampling count')
        assembled[name] = frame
    augmented = pl.concat([training, assembled['train_0.7'].select(training.columns),
                           assembled['train_0.4'].select(training.columns)])
    assert_disjoint(augmented, held)
    if augmented.group_by('pair_id').len().filter(pl.col('len') != 3).height:
        raise ValueError('Augmented pairs must have exactly three training views')
    validations = {'full_history': held, 'history_fraction_0.65': assembled['held_0.65']}
    results, predictions, retained_models = [], [], {}
    for drop_count in [False, True]:
        selected = [c for c in features if not (drop_count and c == 'shared_hands')]
        for use_augmentation in [False, True]:
            candidate = ('augmented' if use_augmentation else 'unaugmented') + (
                '_without_shared_hands' if drop_count else '_with_shared_hands')
            data = augmented if use_augmentation else training
            X = matrix(data, selected)
            y = np.array([LABELS[name] for name in data['behavior_family']])
            weights = np.full(data.height, 1 / 3 if use_augmentation else 1., dtype=np.float64)
            model = lgb.LGBMClassifier(**MODEL_PARAMS)
            model.fit(X, y, sample_weight=weights)
            if not np.array_equal(model.classes_, np.arange(len(FAMILIES))):
                raise ValueError('Unexpected class order')
            result = {'candidate': candidate, 'training_rows': data.height,
                      'training_total_weight': float(weights.sum()), 'features': len(selected)}
            for history, validation in validations.items():
                prob = model.predict_proba(matrix(validation, selected))
                risk = 1 - prob[:, 0]
                truth = np.array([LABELS[name] for name in validation['behavior_family']])
                result[history] = metrics(truth, risk)
                predictions.append(validation.select('pair_id', 'table_id', 'label', 'behavior_family',
                    'shared_hands').with_columns(pl.lit(candidate).alias('candidate'),
                        pl.lit(history).alias('history'), pl.Series('risk_score', risk)))
            results.append(result)
            if use_augmentation:
                retained_models[candidate] = {'model': model, 'features': selected}
            print(json.dumps(result), flush=True)
            del model, X, y, weights
            gc.collect()
    for relative, fingerprint in fingerprints.items():
        if sha(ROOT / relative) != fingerprint:
            raise ValueError(f'Input changed during the experiment: {relative}')
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(__file__), 'input_sha256': fingerprints,
        'versions': {'python': platform.python_version(), 'polars': pl.__version__,
                     'lightgbm': lgb.__version__, 'numpy': np.__version__},
        'model_parameters': MODEL_PARAMS, 'split_seed': SPLIT_SEED,
        'train_views': [{'fraction': 1., 'weight': 1 / 3}] +
                       [{'fraction': fraction, 'seed': seed, 'weight': 1 / 3}
                        for fraction, seed in TRAIN_VIEWS],
        'heldout_view': {'fraction': HELDOUT_VIEW[0], 'seed': HELDOUT_VIEW[1]},
        'sampling': 'floor(fraction*n), at least one; stable hand hash with recorded Polars version; independent view seeds',
        'partition': {'train_pairs': training.height, 'heldout_pairs': held.height,
                      'train_tables': sorted(train_tables), 'heldout_tables': sorted(held_tables),
                      'heldout_targets': int(held['label'].sum()),
                      'pair_table_player_overlap': False, 'matches_saved_v2_fold0': True},
        'shared_hands': {name: {'min': int(frame['shared_hands'].min()),
                               'median': float(frame['shared_hands'].median()),
                               'mean': float(frame['shared_hands'].mean()),
                               'max': int(frame['shared_hands'].max())}
                         for name, frame in {'training_original': training,
                                             **assembled, 'held_original': held}.items()},
        'candidates': results, 'input_hashes_unchanged': True,
        'self_tests_passed': True, 'peak_resident_memory_bytes': peak_memory_bytes(),
        'elapsed_seconds': time.perf_counter() - started,
        'limitations': ['One fixed fold and one draw per history fraction; not a final model selection.',
            'One-percent prior is sensitivity only; actual evaluation prevalence remains unknown.',
            'Per-hand features and player baseline residuals remain based on full development gameplay.',
            'Positive pair labels are retained even when thinning removes informative hands.',
            'Each augmented view receives one-third weight; pairs retain total weight one.',
            'No evidence model or submission is trained, selected, modified, or uploaded.'],
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    short = 'history_fraction_0.65'
    best_unaugmented = max((r for r in results if r['candidate'].startswith('unaugmented')),
                           key=lambda r: r[short]['pair_ap'])
    best_augmented = max((r for r in results if r['candidate'].startswith('augmented')),
                         key=lambda r: r[short]['pair_ap'])
    if (best_augmented[short]['pair_ap'] > best_unaugmented[short]['pair_ap'] and
            best_augmented[short]['pair_ap_at_hypothetical_1pct_prevalence'] >
            best_unaugmented[short]['pair_ap_at_hypothetical_1pct_prevalence']):
        artifact = REPORT / 'augmented_pair_candidate.joblib'
        bundle = retained_models[best_augmented['candidate']]
        bundle['experiment_candidate'] = best_augmented['candidate']
        bundle['trained_on_heldout_pools'] = False
        joblib.dump(bundle, artifact)
        report['review_candidate_artifact'] = str(artifact.relative_to(ROOT))
        report['review_candidate_note'] = ('This fold-only augmented candidate beat the best unaugmented '
            'candidate on both shortened-history AP and the 1% diagnostic. It is not refitted on all '
            'development labels and is not a selected submission model.')
    report['memory_target_below_2gib_met'] = report['peak_resident_memory_bytes'] < 2 * 1024**3
    (REPORT / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    pl.concat(predictions).write_csv(REPORT / 'heldout_predictions.csv')
    print(f'Wrote {REPORT}; peak resident memory {report["peak_resident_memory_bytes"] / 2**20:.1f} MiB', flush=True)


def fit_final():
    """Refit only the prescribed augmented-with-count candidate on all labels."""
    started = time.perf_counter()
    self_test()
    paths = sorted((SOURCE / 'hands_development').glob('batch_*.parquet'))
    if not paths:
        raise ValueError('No v2 development caches found')
    protected_model = SOURCE / 'pair_model.joblib'
    inputs = [DATA / 'development_labels.csv', SOURCE / 'pairs_development.parquet',
              SOURCE / 'hands_development' / 'manifest.json', protected_model,
              ROOT / 'src' / 'train.py', ROOT / 'src' / 'advanced_features.py', *paths]
    fingerprints = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    labels = pl.read_csv(DATA / 'development_labels.csv')
    if labels.height != 1860 or labels['pair_id'].n_unique() != 1860:
        raise ValueError('This declared final refit requires exactly 1860 unique explicit labels')
    confirmed = (((pl.col('label') == 0) & (pl.col('label_status') == 'confirmed_non_target')) |
                 ((pl.col('label') == 1) & (pl.col('label_status') == 'confirmed_target')))
    if labels.filter(~confirmed).height or set(labels['behavior_family']) != set(FAMILIES):
        raise ValueError('Final refit requires consistent confirmed labels and all four classes')
    full = pl.read_parquet(SOURCE / 'pairs_development.parquet').join(labels, on='pair_id').sort('pair_id')
    if full.height != labels.height or full['pair_id'].n_unique() != labels.height:
        raise ValueError('Full pair summaries do not cover the explicit labels exactly once')
    features = [c for c in full.columns if c not in KEYS +
                ['player_1', 'player_2', 'label', 'label_status', 'behavior_family']]
    expected_features = joblib.load(protected_model)['features']
    if len(features) != 653 or features != expected_features or 'shared_hands' not in features:
        raise ValueError('Final feature columns must match all 653 v2 pair features, in order')
    full_y = np.array([LABELS[name] for name in full['behavior_family']])
    if not np.array_equal(full_y > 0, full['label'].to_numpy() == 1):
        raise ValueError('Binary labels and behavior classes disagree')
    expected_keys = full.select('pair_id', 'table_id')
    pieces = {fraction: [] for fraction, _ in TRAIN_VIEWS}
    observed_counts = []
    names = None
    for path in paths:
        raw = pl.read_parquet(path)
        if raw.select('pair_id', 'table_id').unique().join(
                expected_keys, on=['pair_id', 'table_id'], how='anti').height:
            raise ValueError('Unlisted pair in raw training cache')
        current_names = [c for c in raw.columns if c not in KEYS]
        if names is not None and current_names != names:
            raise ValueError('Raw feature schemas differ')
        names = current_names
        observed_counts.append(raw.group_by('pair_id', 'table_id').len())
        for fraction, seed in TRAIN_VIEWS:
            pieces[fraction].append(aggregate(thin(raw, fraction, seed), names))
        print(f'final refit aggregated {path.name}', flush=True)
        del raw
        gc.collect()
    counts = pl.concat(observed_counts).group_by('pair_id', 'table_id').agg(pl.col('len').sum())
    checked = full.select('pair_id', 'table_id', 'shared_hands').join(counts, on=['pair_id', 'table_id'])
    if checked.height != full.height or checked.filter(pl.col('len') != pl.col('shared_hands')).height:
        raise ValueError('Raw pair-hand counts do not match the full v2 summaries')
    views = [full]
    for fraction, _ in TRAIN_VIEWS:
        view = pl.concat(pieces[fraction]).join(labels, on='pair_id').sort('pair_id')
        if view.height != full.height or set(view['pair_id']) != set(full['pair_id']):
            raise ValueError('Each final training view must contain every explicit pair exactly once')
        expected = full.select('pair_id', (pl.col('shared_hands') * fraction).floor()
                               .cast(pl.UInt32).clip(lower_bound=1).alias('expected'))
        if view.select('pair_id', 'shared_hands').join(expected, on='pair_id').filter(
                pl.col('shared_hands') != pl.col('expected')).height:
            raise ValueError('Incorrect sampled hand counts or pair split between batches')
        views.append(view.select(full.columns))
    training = pl.concat(views)
    weights = np.full(training.height, 1 / 3, dtype=np.float64)
    by_pair = training.select('pair_id').with_columns(pl.Series('weight', weights)).group_by('pair_id').agg(
        pl.len().alias('views'), pl.col('weight').sum().alias('total_weight'))
    if (training.height != 5580 or by_pair.height != 1860 or
            by_pair.filter(pl.col('views') != 3).height or
            not np.allclose(by_pair['total_weight'].to_numpy(), 1., rtol=0, atol=1e-12)):
        raise ValueError('Each of the 1860 pairs must have three views totaling weight one')
    X = matrix(training, features)
    y = np.array([LABELS[name] for name in training['behavior_family']])
    class_counts = np.bincount(full_y, minlength=4)
    if not np.allclose(np.bincount(y, weights=weights, minlength=4), class_counts, rtol=0, atol=1e-9):
        raise ValueError('Augmentation changed total class weights')
    model = lgb.LGBMClassifier(**MODEL_PARAMS)
    model.fit(X, y, sample_weight=weights)
    if (not np.array_equal(model.classes_, np.arange(4)) or model.n_features_in_ != 653 or
            model.booster_.current_iteration() != 226):
        raise ValueError('Final model has an unexpected feature count, class order, or tree iteration count')
    for relative, fingerprint in fingerprints.items():
        if sha(ROOT / relative) != fingerprint:
            raise ValueError(f'Input changed during final refit: {relative}')
    provenance = {
        'purpose': 'all_explicit_labels_final_refit', 'candidate': 'augmented_with_shared_hands',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
        'input_sha256': fingerprints, 'model_parameters': MODEL_PARAMS,
        'versions': {'python': platform.python_version(), 'polars': pl.__version__,
                     'lightgbm': lgb.__version__, 'numpy': np.__version__},
        'source_pairs': full.height, 'training_rows': training.height,
        'training_tables': full['table_id'].n_unique(), 'all_labeled_pools_included': True,
        'class_pair_counts': {name: int(class_counts[i]) for i, name in enumerate(FAMILIES)},
        'class_training_rows': {name: int((y == i).sum()) for i, name in enumerate(FAMILIES)},
        'total_training_weight': float(weights.sum()), 'min_pair_weight': float(by_pair['total_weight'].min()),
        'max_pair_weight': float(by_pair['total_weight'].max()), 'unlisted_pairs': 0,
        'views': [{'fraction': 1., 'weight': 1 / 3}] +
                 [{'fraction': fraction, 'seed': seed, 'weight': 1 / 3} for fraction, seed in TRAIN_VIEWS],
        'sampling': 'floor(fraction*n), at least one; label-independent seeded Polars hand hash',
        'feature_count': len(features), 'feature_columns_match_original_v2_model': True,
        'original_v2_model_unchanged': True, 'input_hashes_unchanged': True,
        'self_tests_passed': True, 'peak_resident_memory_bytes': peak_memory_bytes(),
        'elapsed_seconds': time.perf_counter() - started,
        'limitations': ['Final refit has no held-out score of its own; refer to the separate robustness experiment.',
                        'Per-hand features and player baseline residuals are unchanged by thinning.',
                        'No evidence ranking or submission was modified by this refit.'],
    }
    bundle = {'model': model, 'features': features, 'classes': model.classes_.tolist(),
              'class_names': FAMILIES, 'fit_provenance': provenance}
    artifact = SOURCE / 'pair_model_robust.joblib'
    temporary = artifact.with_suffix('.joblib.tmp')
    joblib.dump(bundle, temporary)
    reloaded = joblib.load(temporary)
    np.testing.assert_allclose(model.predict_proba(X[:32]), reloaded['model'].predict_proba(X[:32]),
                               rtol=0, atol=1e-12)
    temporary.replace(artifact)
    final_report = {'artifact': str(artifact.relative_to(ROOT)), 'artifact_sha256': sha(artifact),
                    'reload_predictions_match': True, **provenance}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'final_fit.json').write_text(json.dumps(final_report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: final_report[key] for key in
        ['artifact', 'artifact_sha256', 'source_pairs', 'training_rows', 'feature_count',
         'class_pair_counts', 'total_training_weight', 'min_pair_weight', 'max_pair_weight',
         'original_v2_model_unchanged', 'peak_resident_memory_bytes', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--self-test', action='store_true', help='Run narrow synthetic checks without training')
    mode.add_argument('--fit-final', action='store_true', help='Fit the prescribed augmented model on all 1860 labels once')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print('Sampling, aggregation, stable AP, and leakage guard checks passed.')
    elif args.fit_final:
        fit_final()
    else:
        run()
