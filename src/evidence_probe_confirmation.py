"""Fixed follow-up of both evidence probes on the already-used 100-pool split."""
from __future__ import annotations
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import json
import time
from datetime import datetime, timezone

import joblib
import polars as pl
from evidence_capacity_experiment import ROOT, load_sequence, fit_variant, compare, VARIANTS
from equity_evidence_features import FEATURES
from rank_evidence import PARAMS, context, family_columns, matrix
from relative_evidence_experiment import records, summarize, sha

REPORT = ROOT / 'reports/evidence-probe-confirmation'
PARENT = ROOT / 'reports/sequence-evidence-experiment/confirmation-v2'


def main():
    REPORT.mkdir(exist_ok=False)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'split': str(PARENT / 'start.json'),
            'split_sha256': sha(PARENT / 'start.json'), 'variants': ['v6_control', 'larger_regularized', 'equity'],
            'source_sha256': sha(__file__), 'scope': 'Previously used 100-pool split, exploratory confirmation, no tuning.',
            'intent': 'Measure the same two frozen gameplay candidates on another split before public calibration.'}
    (REPORT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    data, current, truths, _, _ = load_sequence()
    extra = pl.read_parquet(ROOT / 'reports/equity-evidence-experiment/development.parquet')
    data = context(data.join(extra, on=['pair_id', 'hand_id'], validate='1:1'), FEATURES)
    candidate = current + FEATURES + ['context_' + c for c in FEATURES]
    held_pools = json.loads((PARENT / 'start.json').read_text())['held_pools']
    pairpred = pl.read_csv(PARENT / 'full-pair-predictions.csv')
    train = data.filter(~pl.col('table_id').is_in(held_pools))
    held = data.filter(pl.col('table_id').is_in(held_pools)).join(pairpred.select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
    assert not set(train['table_id']) & set(held['table_id'])
    results, allrows = [], []
    for name, features, params in [('v6_control', current, PARAMS),
                                    ('larger_regularized', current, VARIANTS['larger_regularized']),
                                    ('equity', candidate, PARAMS)]:
        start = time.perf_counter()
        if name == 'v6_control':
            bundle = joblib.load(PARENT / 'sequence-ranker.joblib')
            assert bundle['features'] == features
            model = bundle['model']
        else:
            model = fit_variant(train, features, params)
        scores = model.predict(matrix(family_columns(held, 'predicted_behavior'), features), num_threads=2)
        rows = records(held, scores, truths, 'reused_100_pool', name)
        allrows.extend(rows)
        result = {'candidate': name, 'seconds': time.perf_counter() - start, **summarize(rows)}
        results.append(result)
        print(json.dumps(result), flush=True)
    comparisons = {name: compare(allrows, name) for name in ['larger_regularized', 'equity']}
    pl.DataFrame(allrows).write_csv(REPORT / 'pairs.csv')
    payload = {'results': results, 'comparisons': comparisons, 'limitations': 'Previously used split; public calibration has not occurred.'}
    (REPORT / 'metrics.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    main()
