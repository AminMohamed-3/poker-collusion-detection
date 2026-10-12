"""Compare a listwise objective and fixed reciprocal-rank fusion on two pool splits."""
from __future__ import annotations
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import json
from datetime import datetime, timezone
import time
import joblib
import numpy as np
import polars as pl
from evidence_capacity_experiment import ROOT, VARIANTS, load_sequence, fit_variant, compare
from equity_evidence_features import FEATURES as EQ
from rank_evidence import PARAMS, context, family_columns, matrix
from relative_evidence_experiment import records, summarize, sha

REPORT = ROOT / 'reports/evidence-ensemble-experiment'
XENDCG_PARAMS = {k: v for k, v in PARAMS.items() if not k.startswith('lambdarank_') and k != 'label_gain'}
XENDCG_PARAMS['objective'] = 'rank_xendcg'
RRF_K = 10


def fusion(frame, scores, members):
    ranked = frame.select('pair_id').with_columns(*[pl.Series(name, scores[name]) for name in members])
    return ranked.select(pl.mean_horizontal(*[
        1. / (RRF_K + pl.col(name).rank(method='average', descending=True).over('pair_id'))
        for name in members]).alias('fused'))['fused'].to_numpy()


def main():
    REPORT.mkdir(exist_ok=False)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
            'objective': XENDCG_PARAMS, 'fusion': {'k': RRF_K, 'weights': 'equal',
            'three_members': ['v6_control', 'capacity', 'equity'], 'four_members': ['v6_control', 'capacity', 'equity', 'xendcg']},
            'scope': 'Two previously used pool-disjoint validation splits; no learned weights or stacking on these validation labels.',
            'advance': 'Choose largest mean MAP gain across both splits among candidates with gain >=0.005 overall and neither split loss >0.005.'}
    (REPORT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    data, current, truths, pairpred, pools0 = load_sequence()
    data = context(data.join(pl.read_parquet(ROOT / 'reports/equity-evidence-experiment/development.parquet'),
                            on=['pair_id', 'hand_id'], validate='1:1'), EQ)
    equity_features = current + EQ + ['context_' + c for c in EQ]
    parent = ROOT / 'reports/sequence-evidence-experiment/confirmation-v2'
    pools1 = json.loads((parent / 'start.json').read_text())['held_pools']
    assert not set(pools0) & set(pools1)
    specs = [('v6_control', current, PARAMS), ('capacity', current, VARIANTS['larger_regularized']),
             ('equity', equity_features, PARAMS), ('xendcg', current, XENDCG_PARAMS)]
    results, allrows = [], []
    for fold, pools, predictions in [('screen', pools0, pairpred), ('confirmation', pools1, pl.read_csv(parent / 'full-pair-predictions.csv'))]:
        train = data.filter(~pl.col('table_id').is_in(pools))
        held = data.filter(pl.col('table_id').is_in(pools)).join(predictions.select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        inputs = family_columns(held, 'predicted_behavior')
        scores = {}
        for name, features, params in specs:
            start = time.perf_counter()
            if fold == 'screen' and name != 'xendcg':
                location = {'v6_control': ROOT / 'reports/evidence-capacity-experiment/v6_control.joblib',
                            'capacity': ROOT / 'reports/evidence-capacity-experiment/larger_regularized.joblib',
                            'equity': ROOT / 'reports/equity-evidence-experiment/equity.joblib'}[name]
                bundle = joblib.load(location)
                assert bundle['features'] == features
                model = bundle['model']
            elif fold == 'confirmation' and name == 'v6_control':
                model = joblib.load(parent / 'sequence-ranker.joblib')['model']
            else:
                model = fit_variant(train, features, params)
                joblib.dump({'model': model, 'features': features}, REPORT / f'{fold}-{name}.joblib')
            scores[name] = model.predict(matrix(inputs, features), num_threads=2)
            print(json.dumps({'fold': fold, 'model': name, 'seconds': time.perf_counter() - start}), flush=True)
        scores['rrf3'] = fusion(held, scores, ['v6_control', 'capacity', 'equity'])
        scores['rrf4'] = fusion(held, scores, ['v6_control', 'capacity', 'equity', 'xendcg'])
        held.select('pair_id', 'hand_id', 'table_id').with_columns(*[pl.Series(name, score) for name, score in scores.items()]).write_parquet(REPORT / f'{fold}-scores.parquet')
        for name, score in scores.items():
            rows = records(held, score, truths, fold, name)
            allrows.extend(rows)
            result = {'fold': fold, 'candidate': name, **summarize(rows)}
            results.append(result)
            print(json.dumps(result), flush=True)
    comparisons = {name: compare(allrows, name) for name in ['capacity', 'equity', 'xendcg', 'rrf3', 'rrf4']}
    lookup = {(r['fold'], r['candidate']): r['map5'] for r in results}
    delta = {name: {fold: lookup[fold, name] - lookup[fold, 'v6_control'] for fold in ['screen', 'confirmation']}
             for name in ['xendcg', 'rrf3', 'rrf4']}
    passing = [name for name in delta if comparisons[name]['delta'] >= .005 and min(delta[name].values()) >= -.005]
    selected = max(passing, key=lambda name: comparisons[name]['delta']) if passing else None
    pl.DataFrame(allrows).write_csv(REPORT / 'pairs.csv')
    payload = {'results': results, 'comparisons': comparisons, 'split_gains': delta, 'selected': selected,
               'limitations': 'Exploratory validation reused for research; confidence intervals do not account for all model-selection history.'}
    (REPORT / 'metrics.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'selected': selected, 'split_gains': delta}), flush=True)


if __name__ == '__main__':
    main()
