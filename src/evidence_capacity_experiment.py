"""Private, fixed comparison of evidence ranker capacity on the reused screen fold."""
from __future__ import annotations

import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl

from rank_evidence import PARAMS, context, family_columns, matrix
from relative_evidence_experiment import ROOT, load, records, sha, summarize
from sequence_evidence_experiment import FEATURES

REPORT = ROOT / 'reports/evidence-capacity-experiment'
VARIANTS = {
    'v6_control': dict(PARAMS),
    'longer': {**PARAMS, 'n_estimators': 800},
    'larger_regularized': {**PARAMS, 'n_estimators': 800, 'num_leaves': 31,
                           'min_child_samples': 40, 'reg_lambda': 15},
}


def load_sequence():
    data, current, _, truths, pairpred, held_tables = load()
    extra = pl.read_parquet(ROOT / 'reports/sequence-evidence-experiment/development.parquet')
    data = context(data.join(extra, on=['pair_id', 'hand_id'], validate='1:1'), FEATURES)
    features = current + FEATURES + ['context_' + c for c in FEATURES]
    assert len(features) == 268
    return data, features, truths, pairpred, held_tables


def fit_variant(training, features, params):
    training = family_columns(training.sort('pair_id', 'hand_id'), 'behavior_family')
    groups = training.group_by('pair_id', maintain_order=True).agg(pl.len().alias('n'), pl.col('relevance').sum())
    assert groups['relevance'].min() >= 1
    model = lgb.LGBMRanker(**params)
    model.fit(matrix(training, features), training['relevance'].to_numpy(), group=groups['n'].to_numpy())
    return model


def compare(rows, candidate, baseline='v6_control'):
    frame = pl.DataFrame(rows).filter(pl.col('candidate').is_in([baseline, candidate]))
    wide = frame.pivot(on='candidate', index=['pair_id', 'table_id', 'family'], values='map5')
    grouped = wide.group_by('table_id').agg((pl.col(candidate) - pl.col(baseline)).sum().alias('delta'), pl.len().alias('n'))
    grouped = grouped.sort('table_id')
    ix = np.random.default_rng(849217).integers(0, grouped.height, (5000, grouped.height))
    boot = grouped['delta'].to_numpy()[ix].sum(axis=1) / grouped['n'].to_numpy()[ix].sum(axis=1)
    return {'delta': float((wide[candidate] - wide[baseline]).mean()),
            'paired_pool_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
            'unique_pairs': wide.height, 'unique_pools': grouped.height,
            'by_family_delta': {f[0]: float((g[candidate] - g[baseline]).mean())
                                for f, g in wide.group_by('family')}}


def main():
    REPORT.mkdir(parents=True, exist_ok=False)
    protected = ['outputs_v6/submission_sequence_evidence.csv', 'outputs_v6/sequence_ranker.joblib',
                 'outputs_v5/submission_event_tails.csv', 'outputs_v5/pair_model_event_tails.joblib']
    fingerprints = {p: sha(ROOT / p) for p in protected}
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
            'hypothesis': 'The current 238-tree evidence ranker may underfit useful action interactions.',
            'variants': VARIANTS, 'feature_count': 268, 'protected': fingerprints,
            'scope': 'Reused fold0, identical features and family inputs, three sequential fits; diagnostic only.',
            'selection': 'Advance a candidate only if held-out MAP improves at least 0.005 and no family falls by more than 0.025. Prefer the highest MAP among passing candidates. Freeze it before further validation.',
            'limits': ['Train retrieval is optimistic and diagnostic only.',
                       'Unannotated hands are noisy ranking preferences, not confirmed negatives.',
                       'Pool IDs are used for splitting/bootstrap only; pair/hand IDs for grouping/join/tie-breaking only.',
                       'No evaluation labels, IDs as predictors, or generator internals used.']}
    (REPORT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    start = time.perf_counter()
    data, features, truths, pairpred, held_tables = load_sequence()
    train = data.filter(~pl.col('table_id').is_in(held_tables)).sort('pair_id', 'hand_id')
    held = data.filter(pl.col('table_id').is_in(held_tables)).join(
        pairpred.select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
    assert not set(train['table_id']) & set(held['table_id'])
    results, allrows = [], []
    for name, params in VARIANTS.items():
        t0 = time.perf_counter()
        model = fit_variant(train, features, params)
        train_score = model.predict(matrix(family_columns(train, 'behavior_family'), features), num_threads=2)
        held_score = model.predict(matrix(family_columns(held, 'predicted_behavior'), features), num_threads=2)
        assert np.isfinite(train_score).all() and np.isfinite(held_score).all()
        train_rows = records(train, train_score, truths, 'training_diagnostic', name)
        held_rows = records(held, held_score, truths, 'reused_fold0', name)
        allrows.extend(held_rows)
        result = {'candidate': name, 'seconds': time.perf_counter() - t0,
                  'train': summarize(train_rows), 'held': summarize(held_rows)}
        results.append(result)
        held.select('pair_id', 'hand_id').with_columns(pl.Series('score', held_score)).write_parquet(REPORT / f'{name}-scores.parquet')
        joblib.dump({'model': model, 'features': features, 'parameters': params}, REPORT / f'{name}.joblib')
        print(json.dumps(result), flush=True)
    comparisons = {name: compare(allrows, name) for name in VARIANTS if name != 'v6_control'}
    passing = [name for name, result in comparisons.items()
               if result['delta'] >= .005 and min(result['by_family_delta'].values()) >= -.025]
    selected = max(passing, key=lambda name: comparisons[name]['delta']) if passing else None
    assert fingerprints == {p: sha(ROOT / p) for p in protected}
    payload = {'results': results, 'comparisons': comparisons, 'selected': selected,
               'protected_unchanged': True, 'seconds': time.perf_counter() - start,
               'training_pools': train['table_id'].n_unique(), 'held_pools': held['table_id'].n_unique()}
    pl.DataFrame(allrows).write_csv(REPORT / 'pairs.csv')
    (REPORT / 'metrics.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    main()
