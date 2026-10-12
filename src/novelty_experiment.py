"""One fixed negative-only IsolationForest and a predeclared risk floor.

Development fold 0 only. Reuses saved multiclass predictions, fits no supervised
model, and writes only private reports; production artifacts stay untouched.
"""
from __future__ import annotations

import os

for key in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '2'

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit

from open_set_experiment import metric_set
from robust_pair_experiment import FAMILIES, LABELS, assert_disjoint, peak_memory_bytes, sha

ROOT = Path(__file__).resolve().parents[1]
DATA, SOURCE, REPORT = ROOT / 'data', ROOT / 'outputs_v2', ROOT / 'reports'
PARAMS = dict(n_estimators=200, max_samples=256, random_state=7329, n_jobs=2,
              max_features=1.0, contamination='auto', bootstrap=False, warm_start=False)


def training_percentile(training_scores, query_scores):
    """Empirical CDF P(training score <= query), with right-inclusive ties."""
    reference = np.sort(np.asarray(training_scores, dtype=np.float64))
    if len(reference) == 0 or not np.isfinite(reference).all():
        raise ValueError('The reference must contain finite training-negative scores')
    query = np.asarray(query_scores, dtype=np.float64)
    if not np.isfinite(query).all():
        raise ValueError('The query scores must be finite')
    return np.searchsorted(reference, query, side='right') / len(reference)


def score_report(truth, risk):
    result = {'overall_targets_vs_none': metric_set(truth > 0, risk), 'each_family_vs_none': {}}
    for family in FAMILIES[1:]:
        mask = (truth == LABELS[family]) | (truth == 0)
        result['each_family_vs_none'][family] = metric_set(truth[mask] > 0, risk[mask])
    return result


def run():
    started = time.perf_counter()
    np.testing.assert_array_equal(training_percentile([1, 1, 3, 4], [0, 1, 2, 4, 5]),
                                  [0, .5, .5, 1, 1])
    inputs = [DATA / 'development_labels.csv', SOURCE / 'pairs_development.parquet',
              SOURCE / 'validation_pairs.csv', REPORT / 'open-set-predictions.csv',
              REPORT / 'open-set-results.json', ROOT / 'src' / 'open_set_experiment.py',
              ROOT / 'src' / 'robust_pair_experiment.py']
    protected = [SOURCE / 'pair_model_robust.joblib', SOURCE / 'pair_model.joblib',
                 SOURCE / 'hand_model.joblib', ROOT / 'outputs' / 'submission_baseline.csv',
                 SOURCE / 'submission_advanced.csv', SOURCE / 'submission_advanced_robust.csv']
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    protected_before = {str(p.relative_to(ROOT)): {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns}
                        for p in protected}
    previous = json.loads((REPORT / 'open-set-results.json').read_text(encoding='utf-8'))
    for relative, fingerprint in previous['input_sha256'].items():
        if sha(ROOT / relative) != fingerprint:
            raise ValueError(f'Saved prediction provenance no longer matches: {relative}')
    labels = pl.read_csv(DATA / 'development_labels.csv')
    confirmed = (((pl.col('label') == 0) & (pl.col('label_status') == 'confirmed_non_target')) |
                 ((pl.col('label') == 1) & (pl.col('label_status') == 'confirmed_target')))
    if (labels.height != 1860 or labels['pair_id'].n_unique() != 1860 or
            labels.filter(~confirmed).height or set(labels['behavior_family']) != set(FAMILIES)):
        raise ValueError('Expected 1860 unique explicit, internally consistent development labels')
    full = pl.read_parquet(SOURCE / 'pairs_development.parquet').join(labels, on='pair_id').sort('pair_id')
    if full.height != labels.height or full['pair_id'].n_unique() != labels.height:
        raise ValueError('Pair summaries must match each explicit label once')
    excluded = {'pair_id', 'hand_id', 'table_id', 'player_1', 'player_2',
                'label', 'label_status', 'behavior_family'}
    features = [c for c in full.columns if c not in excluded]
    if (len(features) != 653 or features != previous['feature_names'] or
            any(not full.schema[c].is_numeric() for c in features)):
        raise ValueError('Expected the exact same 653 numeric gameplay features')
    truth = np.array([LABELS[name] for name in full['behavior_family']])
    if not np.array_equal(truth > 0, full['label'].to_numpy() == 1):
        raise ValueError('Behavior and binary labels disagree')
    train_index, held_index = next(GroupShuffleSplit(
        n_splits=1, test_size=.25, random_state=2901).split(full, truth, full['table_id']))
    training, held = full[train_index.tolist()].sort('pair_id'), full[held_index.tolist()].sort('pair_id')
    assert_disjoint(training, held)
    if set(held['pair_id']) != set(pl.read_csv(SOURCE / 'validation_pairs.csv')['pair_id']):
        raise ValueError('Held-out membership must match saved v2 fold 0')
    negatives = training.filter((pl.col('behavior_family') == 'none') & (pl.col('label') == 0) &
                                (pl.col('label_status') == 'confirmed_non_target'))
    if negatives.height != 1129 or training.height != 1403 or held.height != 457:
        raise ValueError('Unexpected fixed-split cardinality')
    assert_disjoint(negatives, held)
    X_normal = negatives.select(features).to_numpy().astype(np.float32)
    X_held = held.select(features).to_numpy().astype(np.float32)
    if not np.isfinite(X_normal).all() or not np.isfinite(X_held).all():
        raise ValueError('Raw feature matrices must be finite; no imputation is fitted')
    model = IsolationForest(**PARAMS)
    model.fit(X_normal)
    if len(model.estimators_) != 200 or model.max_samples_ != 256:
        raise ValueError('The fit did not follow the declared size')
    normal_scores = -model.score_samples(X_normal)
    novelty = -model.score_samples(X_held)
    percentile = training_percentile(normal_scores, novelty)
    floor = .5 * np.clip((percentile - .95) / .05, 0, 1)
    if not ((percentile >= 0).all() and (percentile <= 1).all() and
            (floor >= 0).all() and (floor <= .5).all()):
        raise ValueError('Percentile or prescribed floor is outside its bounds')
    held_truth = np.array([LABELS[name] for name in held['behavior_family']])
    standalone = score_report(held_truth, novelty)
    standalone['score'] = 'negative score_samples; larger is more anomalous'
    standalone['percentile_only'] = score_report(held_truth, percentile)
    metadata = held.select('pair_id', 'table_id', 'label', 'behavior_family')
    saved = pl.read_csv(REPORT / 'open-set-predictions.csv').filter(pl.col('objective') == 'multiclass')
    configurations = ['all_families', *FAMILIES[1:]]
    if (set(saved['training_omission']) != set(configurations) or saved.height != 4 * held.height or
            saved.select(pl.struct('pair_id', 'training_omission').n_unique()).item() != saved.height):
        raise ValueError('Expected exactly four unique multiclass prediction configurations')
    comparisons, prediction_frames = [], []
    for omitted in configurations:
        candidate = saved.filter(pl.col('training_omission') == omitted).sort('pair_id')
        if not candidate.select(metadata.columns).equals(metadata):
            raise ValueError('Saved prediction IDs, groups, or truth do not match held-out input')
        risk = candidate['risk_score'].to_numpy()
        if not np.isfinite(risk).all() or not ((risk >= 0).all() and (risk <= 1).all()):
            raise ValueError('Saved risk score is invalid')
        baseline = score_report(held_truth, risk)
        expected = next(row for row in previous['results'] if row['objective'] == 'multiclass' and
                        (row['omitted_training_family'] or 'all_families') == omitted)
        np.testing.assert_allclose(baseline['overall_targets_vs_none']['ap'],
                                   expected['overall_targets_vs_none']['ap'], rtol=0, atol=1e-12)
        blended = np.maximum(risk, floor)
        changed = blended > risk
        comparison = {'training_omission': omitted, 'multiclass': baseline,
                      'fixed_blend': score_report(held_truth, blended),
                      'raised_score_counts_by_truth': {family: int((changed & (held_truth == LABELS[family])).sum())
                                                       for family in FAMILIES}}
        if omitted != 'all_families':
            known_mask = (held_truth == 0) | ((held_truth > 0) & (held_truth != LABELS[omitted]))
            for name, values in [('multiclass', risk), ('fixed_blend', blended)]:
                comparison[name]['omitted_family_vs_none'] = comparison[name]['each_family_vs_none'][omitted]
                comparison[name]['known_families_vs_none'] = metric_set(held_truth[known_mask] > 0, values[known_mask])
        comparisons.append(comparison)
        prediction_frames.append(metadata.with_columns(pl.lit(omitted).alias('training_omission'),
            pl.Series('multiclass_risk', risk), pl.Series('novelty_score', novelty),
            pl.Series('normal_train_percentile', percentile), pl.Series('novelty_risk_floor', floor),
            pl.Series('blended_risk', blended)))
        print(json.dumps({'training_omission': omitted,
                          'overall_ap_before': baseline['overall_targets_vs_none']['ap'],
                          'overall_ap_after': comparison['fixed_blend']['overall_targets_vs_none']['ap'],
                          'raised_counts': comparison['raised_score_counts_by_truth']}), flush=True)
    distribution = {}
    for family in FAMILIES:
        mask = held_truth == LABELS[family]
        distribution[family] = {'pairs': int(mask.sum()), 'median_novelty': float(np.median(novelty[mask])),
                                'median_percentile': float(np.median(percentile[mask])),
                                'above_95pct_training_reference': int((percentile[mask] > .95).sum()),
                                'above_99pct_training_reference': int((percentile[mask] > .99).sum()),
                                'at_maximum_percentile': int((percentile[mask] == 1).sum())}
    for relative, fingerprint in fingerprints.items():
        if sha(ROOT / relative) != fingerprint:
            raise ValueError(f'Input changed during experiment: {relative}')
    for relative, before in protected_before.items():
        path = ROOT / relative
        if sha(path) != before['sha256'] or path.stat().st_mtime_ns != before['mtime_ns']:
            raise ValueError(f'Protected artifact changed: {relative}')
    output = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
        'input_sha256': fingerprints, 'protected_artifacts_before_and_after': protected_before,
        'protected_artifacts_unchanged': True, 'input_hashes_unchanged': True,
        'feature_count': len(features), 'feature_names': features, 'raw_features_finite': True,
        'numeric_preprocessing': 'float32 conversion only; no fitted scaling or imputation',
        'model_parameters': PARAMS, 'fitted_novelty_models': 1, 'fitted_supervised_models': 0,
        'fixed_fold': {'seed': 2901, 'test_size': .25, 'matches_saved_v2_fold0': True,
                       'training_pairs': training.height, 'training_confirmed_none_pairs': negatives.height,
                       'fitted_positive_pairs': 0, 'fitted_unlisted_pairs': 0,
                       'heldout_pairs': held.height, 'heldout_targets': int((held_truth > 0).sum()),
                       'training_pools': training['table_id'].n_unique(),
                       'heldout_pools': held['table_id'].n_unique(), 'pair_table_player_overlap': False},
        'reference': {'source': 'in-sample scores of the 1129 fitted confirmed-none TRAIN pairs only',
                      'definition': 'count(training novelty <= query novelty) / 1129; right-inclusive ties',
                      'normal_score_quantiles': {str(q): float(np.quantile(normal_scores, q))
                                                 for q in [0, .5, .95, .99, 1]},
                      'heldout_reference_rows': 0},
        'blend': 'max(multiclass_risk, 0.5 * clip((normal_train_percentile - 0.95) / 0.05, 0, 1))',
        'standalone': standalone, 'comparisons': comparisons, 'heldout_score_distribution': distribution,
        'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                     'polars': pl.__version__, 'numpy': np.__version__},
        'elapsed_seconds': time.perf_counter() - started, 'peak_resident_memory_bytes': peak_memory_bytes(),
        'limitations': [
            'Known held-out families proxy novelty; the actual fourth mechanism is unobserved.',
            'Empirical training-normal percentiles and the risk floor are ranking scores, not calibrated probabilities.',
            'The normal reference is in-sample; its held-out tail rate need not equal 5%.',
            'The 1% prevalence calculation is a reweighted diagnostic, not inferred test prevalence.',
            'One fixed split, one fixed novelty model, and one prescribed blend; no tuning or full evaluation predictions.',
            'Saved multiclass controls use full histories without the production robust model history augmentation.',
            'No behavior-family or evidence-ranking benefits are evaluated; no model or submission is saved.'],
    }
    (REPORT / 'novelty-results.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    pl.concat(prediction_frames).write_csv(REPORT / 'novelty-predictions.csv')
    negatives.select('pair_id', 'table_id').with_columns(
        pl.Series('novelty_score', normal_scores)).write_csv(REPORT / 'novelty-training-reference.csv')
    print(json.dumps({'standalone': standalone['overall_targets_vs_none'],
                      'elapsed_seconds': output['elapsed_seconds'],
                      'protected_artifacts_unchanged': True}), flush=True)


if __name__ == '__main__':
    run()
