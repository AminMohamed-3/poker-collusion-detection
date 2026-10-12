"""Eight fixed, pool-held-out LightGBM fits for unseen-family generalization.

Uses existing v2 pair summaries only. Omitted families are removed from training,
never relabeled. Writes private reports only; no model or submission is saved.
"""
from __future__ import annotations

import os

for key in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '2'

import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from robust_pair_experiment import (FAMILIES, LABELS, MODEL_PARAMS, assert_disjoint,
                                    matrix, peak_memory_bytes, self_test, sha, stable_ap)

ROOT = Path(__file__).resolve().parents[1]
DATA, SOURCE, REPORT = ROOT / 'data', ROOT / 'outputs_v2', ROOT / 'reports'


def metric_set(truth, risk):
    truth = np.asarray(truth, dtype=bool)
    prevalence = float(truth.mean())
    if not 0 < prevalence < 1:
        raise ValueError('Evaluation subset must contain targets and confirmed non-targets')
    weights = np.where(truth, .01 / prevalence, .99 / (1 - prevalence))
    return {'pairs': len(truth), 'positives': int(truth.sum()),
            'confirmed_non_targets': int((~truth).sum()), 'prevalence': prevalence,
            'ap': stable_ap(truth, risk),
            'sklearn_ap': float(average_precision_score(truth, risk)),
            'roc_auc': float(roc_auc_score(truth, risk)),
            'ap_at_hypothetical_1pct_prevalence': stable_ap(truth, risk, weights)}


def run():
    started = time.perf_counter()
    self_test()
    inputs = [DATA / 'development_labels.csv', SOURCE / 'pairs_development.parquet',
              SOURCE / 'validation_pairs.csv', ROOT / 'src' / 'robust_pair_experiment.py',
              ROOT / 'src' / 'train.py']
    protected = [SOURCE / 'pair_model_robust.joblib', SOURCE / 'pair_model.joblib',
                 SOURCE / 'hand_model.joblib', *sorted(SOURCE.glob('submission_*.csv'))]
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    protected_before = {str(p.relative_to(ROOT)): {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns}
                        for p in protected}
    labels = pl.read_csv(DATA / 'development_labels.csv')
    if labels.height != 1860 or labels['pair_id'].n_unique() != 1860:
        raise ValueError('Expected exactly 1860 unique explicit development labels')
    confirmed = (((pl.col('label') == 0) & (pl.col('label_status') == 'confirmed_non_target')) |
                 ((pl.col('label') == 1) & (pl.col('label_status') == 'confirmed_target')))
    if labels.filter(~confirmed).height or set(labels['behavior_family']) != set(FAMILIES):
        raise ValueError('Inconsistent or non-confirmed labels')
    full = pl.read_parquet(SOURCE / 'pairs_development.parquet').join(labels, on='pair_id').sort('pair_id')
    if full.height != 1860 or full['pair_id'].n_unique() != 1860:
        raise ValueError('Pair summaries must match explicit labels once each')
    excluded_columns = {'pair_id', 'hand_id', 'table_id', 'player_1', 'player_2',
                        'label', 'label_status', 'behavior_family'}
    features = [c for c in full.columns if c not in excluded_columns]
    if len(features) != 653 or any(not full.schema[c].is_numeric() for c in features):
        raise ValueError('Expected 653 numeric v2 gameplay features')
    truth = np.array([LABELS[name] for name in full['behavior_family']])
    if not np.array_equal(truth > 0, full['label'].to_numpy() == 1):
        raise ValueError('Behavior and binary labels disagree')
    train_index, held_index = next(GroupShuffleSplit(
        n_splits=1, test_size=.25, random_state=2901).split(full, truth, full['table_id']))
    training, held = full[train_index.tolist()], full[held_index.tolist()]
    assert_disjoint(training, held)
    if set(held['pair_id']) != set(pl.read_csv(SOURCE / 'validation_pairs.csv')['pair_id']):
        raise ValueError('Held-out membership does not match saved v2 fold 0')
    held_truth = np.array([LABELS[name] for name in held['behavior_family']])
    X_held = matrix(held, features)
    results, predictions = [], []
    for omitted in [None, *FAMILIES[1:]]:
        subset = training if omitted is None else training.filter(pl.col('behavior_family') != omitted)
        assert_disjoint(subset, held)
        if omitted is not None and omitted in set(subset['behavior_family']):
            raise ValueError('Omitted family remains in training')
        if subset.filter(pl.col('label') == 0).height != training.filter(pl.col('label') == 0).height:
            raise ValueError('Omission must leave all confirmed training negatives intact')
        X = matrix(subset, features)
        y_multi = np.array([LABELS[name] for name in subset['behavior_family']])
        for objective in ['multiclass', 'binary']:
            y = y_multi if objective == 'multiclass' else (y_multi > 0).astype(np.int32)
            model = lgb.LGBMClassifier(**MODEL_PARAMS)
            model.fit(X, y)
            classes = model.classes_.tolist()
            if classes.count(0) != 1 or (objective == 'binary' and classes != [0, 1]):
                raise ValueError('Unexpected class mapping')
            if objective == 'multiclass' and omitted is not None and LABELS[omitted] in classes:
                raise ValueError('Omitted positive class was not removed from the multiclass model')
            prob = model.predict_proba(X_held)
            risk = 1 - prob[:, classes.index(0)]
            result = {
                'omitted_training_family': omitted, 'objective': objective,
                'training_pairs': subset.height, 'removed_training_pairs': training.height - subset.height,
                'training_family_counts': {key[0]: group.height for key, group in subset.group_by('behavior_family')},
                'model_classes': classes, 'boosting_rounds': model.booster_.current_iteration(),
                'overall_targets_vs_none': metric_set(held_truth > 0, risk),
                'each_family_vs_none': {}, 'risk_medians_by_truth': {},
            }
            for family in FAMILIES:
                family_mask = held_truth == LABELS[family]
                result['risk_medians_by_truth'][family] = float(np.median(risk[family_mask]))
                if family != 'none':
                    mask = family_mask | (held_truth == 0)
                    result['each_family_vs_none'][family] = metric_set(held_truth[mask] > 0, risk[mask])
            if omitted is not None:
                result['omitted_family_vs_none'] = result['each_family_vs_none'][omitted]
                known = (held_truth == 0) | ((held_truth > 0) & (held_truth != LABELS[omitted]))
                result['known_families_vs_none'] = metric_set(held_truth[known] > 0, risk[known])
            results.append(result)
            predictions.append(held.select('pair_id', 'table_id', 'label', 'behavior_family').with_columns(
                pl.lit(omitted or 'all_families').alias('training_omission'),
                pl.lit(objective).alias('objective'), pl.Series('risk_score', risk)))
            print(json.dumps({'omitted_family': omitted, 'objective': objective,
                              'omitted_ap': result.get('omitted_family_vs_none', {}).get('ap'),
                              'overall_ap': result['overall_targets_vs_none']['ap'],
                              'overall_ap_1pct': result['overall_targets_vs_none']['ap_at_hypothetical_1pct_prevalence']}), flush=True)
            del model, prob
            gc.collect()
        del X
    if len(results) != 8:
        raise ValueError('Experiment must remain bounded to eight fits')
    for relative, fingerprint in fingerprints.items():
        if sha(ROOT / relative) != fingerprint:
            raise ValueError(f'Input changed during experiment: {relative}')
    for relative, before in protected_before.items():
        path = ROOT / relative
        if sha(path) != before['sha256'] or path.stat().st_mtime_ns != before['mtime_ns']:
            raise ValueError(f'Protected model or submission changed: {relative}')
    prediction_frame = pl.concat(predictions)
    if prediction_frame.height != 8 * 457 or prediction_frame.select(
            pl.struct('pair_id', 'training_omission', 'objective').n_unique()).item() != prediction_frame.height:
        raise ValueError('Prediction output cardinality or uniqueness error')
    output = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
        'input_sha256': fingerprints, 'protected_artifacts_before_and_after': protected_before,
        'protected_artifacts_unchanged': True, 'input_hashes_unchanged': True,
        'feature_names': features, 'feature_count': len(features), 'model_parameters': MODEL_PARAMS,
        'versions': {'python': platform.python_version(), 'lightgbm': lgb.__version__,
                     'polars': pl.__version__, 'numpy': np.__version__},
        'fixed_fold': {'seed': 2901, 'test_size': .25, 'matches_saved_v2_fold0': True,
                       'training_pairs': training.height, 'heldout_pairs': held.height,
                       'heldout_targets': int((held_truth > 0).sum()),
                       'training_tables': sorted(set(training['table_id'])),
                       'heldout_tables': sorted(set(held['table_id'])),
                       'pair_table_player_overlap': False},
        'fitted_models': len(results), 'results': results,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_resident_memory_bytes': peak_memory_bytes(),
        'limitations': [
            'Leave-one-known-family-out is a proxy; the undisclosed fourth mechanism remains unknown.',
            'Only existing full-history v2 pair summaries are used; no history augmentation is included.',
            'The all-family binary/multiclass control isolates objective, not a direct replacement test against robust augmentation.',
            'One fixed held-out pool split; no tuning, repeated-seed search, or model selection on hidden truth.',
            'The 1% prevalence metric is a sensitivity diagnostic, not inferred evaluation prevalence.',
            'Binary risk supplies no behavior or evidence class; those competition components are not evaluated.',
            'No model, evidence ranking, or submission is saved or modified.'],
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'open-set-results.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    prediction_frame.write_csv(REPORT / 'open-set-predictions.csv')
    print(f'Completed eight fits; protected artifacts unchanged; {output["elapsed_seconds"]:.2f}s, '
          f'{output["peak_resident_memory_bytes"] / 2**20:.2f} MiB peak', flush=True)


if __name__ == '__main__':
    run()
