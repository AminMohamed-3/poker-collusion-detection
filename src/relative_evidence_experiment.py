"""One fixed evidence candidate; private, bounded, pool-disjoint comparisons.

Adds order statistics and missing standardized context within each pair's observed
gameplay. IDs are grouping/join keys only. No production artifact is modified.
"""
from __future__ import annotations

import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from evidence_context_experiment import ap5
from rank_evidence import PARAMS, FAMILIES, KEYS, context, family_columns, matrix

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports'/'relative-evidence-experiment'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_context(frame, base):
    """Pair-local empirical midrank and z-score, independent of labels and IDs."""
    non_events = [c for c in base if not c.startswith('v2_')]
    result = context(frame, base)
    result = result.with_columns(*[
        ((pl.col(c).rank(method='average').over('pair_id') - .5) /
         pl.len().over('pair_id')).cast(pl.Float32).alias('percentile_'+c)
        for c in base])
    extra = ['context_'+c for c in non_events] + ['percentile_'+c for c in base]
    return result, extra


def load():
    labels = pl.read_csv(ROOT/'data/development_labels.csv')
    evidence = pl.read_csv(ROOT/'data/development_evidence.csv').select('pair_id', 'hand_id')
    raw = pl.read_parquet(sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet')))
    assert evidence.unique().height == evidence.height
    assert raw.select(pl.struct('pair_id', 'hand_id').n_unique()).item() == raw.height
    assert set(raw['pair_id']) == set(labels['pair_id'])
    base = [c for c in raw.columns if c not in KEYS]
    target = raw.join(labels.filter(pl.col('label') == 1).select('pair_id', 'behavior_family'), on='pair_id')
    data, extra = relative_context(target, base)
    data = data.join(evidence.with_columns(pl.lit(1).alias('relevance')), on=['pair_id', 'hand_id'], how='left')
    data = data.with_columns(pl.col('relevance').fill_null(0)).sort('pair_id', 'hand_id')
    assert data['relevance'].sum() == evidence.height
    current = base + ['context_'+c for c in base if c.startswith('v2_')] + ['family_'+f for f in FAMILIES]
    candidate = current + extra
    assert len(current) == 222 and len(candidate) == 492
    assert set(candidate).isdisjoint(KEYS + ['behavior_family', 'relevance'])
    assert data.select(pl.col(extra).is_finite().all()).row(0) == (True,)*len(extra)
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    pairpred = pl.read_csv(ROOT/'outputs_v2/validation_pairs.csv')
    held_tables = raw.filter(pl.col('pair_id').is_in(pairpred['pair_id'].to_list()))['table_id'].unique().to_list()
    assert set(raw.filter(pl.col('table_id').is_in(held_tables))['pair_id']) == set(pairpred['pair_id'])
    return data, current, candidate, truths, pairpred, held_tables


def fit(data, features):
    train = family_columns(data.sort('pair_id', 'hand_id'), 'behavior_family')
    groups = train.group_by('pair_id', maintain_order=True).agg(pl.len().alias('n'), pl.col('relevance').sum())
    assert groups['relevance'].min() >= 1 and groups['n'].sum() == train.height
    model = lgb.LGBMRanker(**PARAMS)
    model.fit(matrix(train, features), train['relevance'].to_numpy(), group=groups['n'].to_numpy())
    return model


def records(held, predictions, truths, tag, candidate):
    # Seeded random tie breaker independent of feature values and labels.
    scored = held.select('pair_id', 'hand_id', 'table_id', 'behavior_family').with_columns(
        pl.Series('score', predictions), pl.Series('tie', np.random.default_rng(931207).random(held.height)))
    output = []
    for key, frame in scored.group_by('pair_id'):
        order = frame.sort(['score', 'tie'], descending=[True, False])['hand_id'].head(5).to_list()
        output.append({'pair_id': key[0], 'table_id': frame['table_id'][0], 'family': frame['behavior_family'][0],
                       'experiment': tag, 'candidate': candidate, 'map5': ap5(order, truths[key[0]])})
    return output


def summarize(rows):
    return {'map5': float(np.mean([r['map5'] for r in rows])), 'pairs': len(rows),
            'by_family': {f: float(np.mean([r['map5'] for r in rows if r['family'] == f]))
                          for f in FAMILIES if any(r['family'] == f for r in rows)}}


def bootstrap(rows, tag):
    df = pl.DataFrame([r for r in rows if r['experiment'] == tag])
    wide = df.pivot(on='candidate', index=['pair_id', 'table_id'], values='map5')
    pools = wide.group_by('table_id').agg((pl.col('relative')-pl.col('current')).sum().alias('delta'), pl.len().alias('n'))
    rng = np.random.default_rng(849217)
    ix = rng.integers(0, pools.height, size=(2000, pools.height))
    delta = pools['delta'].to_numpy()[ix].sum(axis=1) / pools['n'].to_numpy()[ix].sum(axis=1)
    return np.quantile(delta, [.025, .975]).tolist()


def screen():
    REPORT.mkdir(exist_ok=True)
    result_path = REPORT/'screen.json'
    if result_path.exists() or (REPORT/'screen-start.json').exists():
        raise FileExistsError('Screen was already started; preserve its record')
    protected = [ROOT/'outputs_v5/submission_event_tails.csv', ROOT/'outputs_v5/pair_model_event_tails.joblib',
                 ROOT/'outputs_v3/evidence_ranker.joblib']
    protected_hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'candidate': '492-feature ranker: unchanged 222 features + 107 missing pair z-scores + 163 pair empirical midranks',
            'parameters': PARAMS, 'fits': 8, 'source_sha256': sha(__file__),
            'screen_gate': {'known_map5_gain_min': .015, 'known_family_regression_max': .02,
                            'omitted_family_mean_gain_min': 0., 'omitted_family_regression_max': .025},
            'scope': 'Existing fold0 only; 100-pool less-reused confirmation set remains unselected/unscored.',
            'protected_sha256': protected_hashes}
    (REPORT/'screen-start.json').write_text(json.dumps(plan, indent=2)+'\n')
    started = time.perf_counter()
    data, current, candidate, truths, pairpred, held_tables = load()
    train = data.filter(~pl.col('table_id').is_in(held_tables))
    held = data.filter(pl.col('table_id').is_in(held_tables)).join(pairpred.select('pair_id', 'predicted_behavior'), on='pair_id').sort('pair_id', 'hand_id')
    assert not set(train['table_id']) & set(held['table_id'])
    assert not set(train['pair_id']) & set(held['pair_id'])
    results, allrows = [], []
    for excluded in [None, *FAMILIES]:
        tag = 'known' if excluded is None else 'omit_'+excluded
        training = train if excluded is None else train.filter(pl.col('behavior_family') != excluded)
        validation = held if excluded is None else held.filter(pl.col('behavior_family') == excluded)
        for name, features in [('current', current), ('relative', candidate)]:
            t0 = time.perf_counter()
            model = fit(training, features)
            if excluded is None:
                predicted = model.predict(matrix(family_columns(validation, 'predicted_behavior'), features), num_threads=2)
            else:
                # Unknown family is never fed to either model. Marginalize over
                # supported families equally; this is a retrieval proxy, not an
                # end-to-end pair-classifier result.
                choices = [f for f in FAMILIES if f != excluded]
                predicted = np.mean([model.predict(matrix(family_columns(validation.with_columns(
                    pl.lit(f).alias('hypothesis')), 'hypothesis'), features), num_threads=2) for f in choices], axis=0)
            assert np.isfinite(predicted).all()
            rows = records(validation, predicted, truths, tag, name)
            result = {'experiment': tag, 'candidate': name, 'training_pairs': training['pair_id'].n_unique(),
                      'features': len(features), 'seconds': time.perf_counter()-t0, **summarize(rows)}
            results.append(result); allrows.extend(rows)
            print(json.dumps(result), flush=True)
    lookup = {(r['experiment'], r['candidate']): r for r in results}
    gains = {tag: lookup[tag, 'relative']['map5']-lookup[tag, 'current']['map5'] for tag in ['known', *['omit_'+f for f in FAMILIES]]}
    family_deltas = {f: lookup['known', 'relative']['by_family'][f]-lookup['known', 'current']['by_family'][f] for f in FAMILIES}
    omitted = [gains['omit_'+f] for f in FAMILIES]
    passed = gains['known'] >= .015 and min(family_deltas.values()) >= -.02 and np.mean(omitted) >= 0 and min(omitted) >= -.025
    assert {str(p.relative_to(ROOT)): sha(p) for p in protected} == protected_hashes
    result = {'plan': plan, 'results': results, 'gains': gains, 'known_family_deltas': family_deltas,
              'known_paired_pool_bootstrap_95': bootstrap(allrows, 'known'), 'screen_passed': bool(passed),
              'protected_unchanged': True, 'seconds': time.perf_counter()-started,
              'limitations': ['Reused fold0; screening results are exploratory, not independent confirmation.',
                  'Omitted-family evidence uses equal supported-family marginalization, not a fitted pair classifier.',
                  'Annotation zero denotes unlisted hands; no confirmed-negative claim is made.',
                  'No final model, evaluation predictions or submission produced.']}
    pl.DataFrame(allrows).write_csv(REPORT/'screen-pairs.csv')
    result_path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'gains': gains, 'screen_passed': bool(passed), 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['screen'])
    parser.parse_args()
    screen()
