"""Fixed MAP-focused evidence model and equal-rank ensemble screening.

XGBoost MAP and NDCG share all parameters, providing an objective-matched
comparison. The existing LightGBM model is the production control. No tuning.
"""
from __future__ import annotations
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import polars as pl
import xgboost as xgb
from relative_evidence_experiment import load, fit, sha, records, summarize, ROOT
from rank_evidence import family_columns, matrix, FAMILIES

REPORT = ROOT/'reports'/'map-evidence-experiment'
XGB_PARAMS = dict(n_estimators=238, learning_rate=.03, tree_method='hist', device='cpu',
                  max_depth=0, max_leaves=15, grow_policy='lossguide', min_child_weight=5,
                  colsample_bytree=.8, reg_lambda=5, n_jobs=2, random_state=7329,
                  lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=5,
                  eval_metric='map@5')


def fit_map(frame, features, objective='rank:map'):
    frame = family_columns(frame.sort('pair_id', 'hand_id'), 'behavior_family')
    group = frame.group_by('pair_id', maintain_order=True).len()['len'].to_numpy()
    model = xgb.XGBRanker(objective=objective, **XGB_PARAMS)
    model.fit(matrix(frame, features), frame['relevance'].to_numpy(), group=group, verbose=False)
    return model


def mean_ranks(frame, current, alternative):
    # Equal weight, average ties, no score calibration or fitted blend weight.
    scored = frame.select('pair_id').with_columns(pl.Series('a', current), pl.Series('b', alternative))
    return scored.select(((pl.col('a').rank(method='average').over('pair_id') +
                           pl.col('b').rank(method='average').over('pair_id')) /
                          (2*pl.len().over('pair_id'))).alias('score'))['score'].to_numpy()


def interval(rows, candidate):
    df = pl.DataFrame([r for r in rows if r['experiment'] == 'known' and r['candidate'] in ['current', candidate]])
    wide = df.pivot(on='candidate', index=['pair_id', 'table_id'], values='map5')
    pools = wide.group_by('table_id').agg((pl.col(candidate)-pl.col('current')).sum().alias('delta'), pl.len().alias('n'))
    ix = np.random.default_rng(849217).integers(0, pools.height, size=(2000, pools.height))
    gains = pools['delta'].to_numpy()[ix].sum(axis=1)/pools['n'].to_numpy()[ix].sum(axis=1)
    return np.quantile(gains, [.025, .975]).tolist()


def main():
    REPORT.mkdir(exist_ok=True)
    if (REPORT/'start.json').exists():
        raise FileExistsError('Preserve completed or interrupted fixed experiment')
    protected = [ROOT/'outputs_v5/submission_event_tails.csv', ROOT/'outputs_v5/pair_model_event_tails.joblib', ROOT/'outputs_v3/evidence_ranker.joblib']
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'src/relative_evidence_experiment.py', ROOT/'src/rank_evidence.py']}
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'xgboost': xgb.__version__,
            'parameters': XGB_PARAMS, 'candidates': ['map', 'equal_rank_blend'], 'diagnostic_control': 'xgb_ndcg',
            'features': 'Existing 222, unchanged', 'known_gate': {'map_gain_min': .015, 'family_regression_max': .02},
            'omitted_gate': {'unweighted_mean_gain_min': 0., 'max_family_regression': .025},
            'selection': 'Among candidates passing all screens choose greatest known-family MAP; no blend-weight tuning.',
            'fit_budget': '3 known-family fits; if either candidate passes known gate, 6 omitted-family fits; otherwise stop.',
            'protected': fingerprints, 'sources': sources,
            'reference': 'https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/learning_to_rank.html'}
    (REPORT/'start.json').write_text(json.dumps(plan, indent=2)+'\n')
    started = time.perf_counter()
    data, features, _, truths, pairpred, held_tables = load()
    train = data.filter(~pl.col('table_id').is_in(held_tables))
    held = data.filter(pl.col('table_id').is_in(held_tables)).join(pairpred.select('pair_id', 'predicted_behavior'), on='pair_id').sort('pair_id','hand_id')
    assert not set(train['table_id']) & set(held['table_id'])
    values = matrix(family_columns(held, 'predicted_behavior'), features)
    scores, results, allrows = {}, [], []
    for name in ['current', 'map', 'xgb_ndcg']:
        t0 = time.perf_counter()
        model = fit(train, features) if name == 'current' else fit_map(train, features, 'rank:map' if name == 'map' else 'rank:ndcg')
        scores[name] = model.predict(values)
        assert np.isfinite(scores[name]).all()
        rows = records(held, scores[name], truths, 'known', name)
        result = {'experiment': 'known', 'candidate': name, 'seconds': time.perf_counter()-t0, **summarize(rows)}
        results.append(result); allrows.extend(rows)
        print(json.dumps(result), flush=True)
    scores['equal_rank_blend'] = mean_ranks(held, scores['current'], scores['map'])
    rows = records(held, scores['equal_rank_blend'], truths, 'known', 'equal_rank_blend')
    result = {'experiment': 'known', 'candidate': 'equal_rank_blend', **summarize(rows)}
    results.append(result); allrows.extend(rows)
    print(json.dumps(result), flush=True)
    held.select('pair_id', 'hand_id', 'table_id').with_columns(*[pl.Series(n, p) for n, p in scores.items()]).write_parquet(REPORT/'known-hand-scores.parquet')
    lookup = {r['candidate']: r for r in results}
    eligibility = {name: lookup[name]['map5']-lookup['current']['map5'] >= .015 and
                   min(lookup[name]['by_family'][f]-lookup['current']['by_family'][f] for f in FAMILIES) >= -.02
                   for name in ['map', 'equal_rank_blend']}
    if any(eligibility.values()):
        for excluded in FAMILIES:
            tag = 'omit_'+excluded
            training = train.filter(pl.col('behavior_family') != excluded)
            validation = held.filter(pl.col('behavior_family') == excluded)
            supported = [f for f in FAMILIES if f != excluded]
            prediction = {}
            for name in ['current', 'map']:
                model = fit(training, features) if name == 'current' else fit_map(training, features)
                prediction[name] = np.mean([model.predict(matrix(family_columns(validation.with_columns(
                    pl.lit(f).alias('hypothesis')), 'hypothesis'), features)) for f in supported], axis=0)
            prediction['equal_rank_blend'] = mean_ranks(validation, prediction['current'], prediction['map'])
            for name, scores in prediction.items():
                rows = records(validation, scores, truths, tag, name)
                result = {'experiment': tag, 'candidate': name, **summarize(rows)}
                results.append(result); allrows.extend(rows)
                print(json.dumps(result), flush=True)
        by_key = {(r['experiment'], r['candidate']): r for r in results}
        for name in eligibility:
            gains = [by_key['omit_'+f, name]['map5']-by_key['omit_'+f, 'current']['map5'] for f in FAMILIES]
            eligibility[name] = eligibility[name] and np.mean(gains) >= 0 and min(gains) >= -.025
    selected = max((n for n, valid in eligibility.items() if valid), key=lambda n: lookup[n]['map5'], default=None)
    assert {str(p.relative_to(ROOT)): sha(p) for p in protected} == fingerprints
    payload = {'plan': plan, 'results': results, 'eligible': eligibility, 'selected': selected,
               'known_bootstrap_95': {n: interval(allrows, n) for n in eligibility},
               'seconds': time.perf_counter()-started, 'protected_unchanged': True,
               'limitations': ['Reused screening fold; no independent confirmation or public improvement claimed.',
                   'Two fixed candidate outputs examined; bootstrap does not correct model-selection bias.',
                   'Changing boosting library also changes optimization; the XGBoost NDCG control isolates objective within that library.',
                   'Unlisted target-pair hands are noisy preference zeros, not confirmed negatives.',
                   'The less-reused confirmation pools remain unselected and unscored.']}
    pl.DataFrame(allrows).write_csv(REPORT/'screen-pairs.csv')
    (REPORT/'screen.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({'eligible': eligibility, 'selected': selected, 'seconds': payload['seconds']}), flush=True)


if __name__ == '__main__':
    main()
