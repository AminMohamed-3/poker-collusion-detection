"""Apply the frozen three-model reciprocal-rank ensemble to cached candidate hands."""
from __future__ import annotations
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import gc
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import polars as pl
from evidence_ensemble_experiment import fusion, RRF_K
from evidence_public_calibration import ROOT, REPORT, SOURCE, TOP_PAIRS, EQ, SEQ, rank
from rank_evidence import EVIDENCE_COLUMNS, family_columns, context, matrix
from relative_evidence_experiment import sha
from sequence_evidence_candidate import write_json

OUT = ROOT / 'outputs_v9_ensemble'


def main():
    evidence = json.loads((ROOT / 'reports/evidence-ensemble-experiment/metrics.json').read_text())
    assert evidence['selected'] == 'rrf3'
    source = pl.read_csv(SOURCE, infer_schema=False)
    paths = sorted((REPORT).glob('batch_*-features.parquet'))
    assert len(paths) == 20
    models = {'v6_control': ROOT / 'outputs_v6/sequence_ranker.joblib',
              'capacity': ROOT / 'outputs_v7_capacity/ranker.joblib', 'equity': ROOT / 'outputs_v8_equity/ranker.joblib'}
    bundles = {name: joblib.load(path) for name, path in models.items()}
    OUT.mkdir(exist_ok=True)
    cache = OUT / 'rank_batches'
    cache.mkdir(exist_ok=True)
    identity = {str(path.relative_to(ROOT)): sha(path) for path in [*models.values(), SOURCE,
                ROOT / 'src/evidence_public_ensemble.py', ROOT / 'src/evidence_ensemble_experiment.py',
                ROOT / 'reports/evidence-ensemble-experiment/metrics.json', *paths]}
    plan_path = OUT / 'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text())['inputs'] == identity
    else:
        write_json(plan_path, {'created_at': datetime.now(timezone.utc).isoformat(), 'inputs': identity,
                              'k': RRF_K, 'members': list(models), 'weights': 'equal', 'learned_meta_model': False,
                              'scope': f'The same {TOP_PAIRS} v6 highest-risk pairs as both standalone probes.'})
    for index, path in enumerate(paths):
        batch_name = f'batch_{index:03d}'
        destination = cache / (batch_name + '.parquet')
        receipt = cache / (batch_name + '.json')
        if receipt.exists():
            meta = json.loads(receipt.read_text())
            assert meta['plan_sha256'] == sha(plan_path) and meta['output_sha256'] == sha(destination)
            continue
        feature_receipt = json.loads((REPORT / (batch_name + '.json')).read_text())
        assert sha(path) == feature_receipt['files'][str(path.relative_to(ROOT))]
        raw = pl.read_parquet(ROOT / 'outputs_v2/hands_evaluation' / (batch_name + '.parquet'))
        added = pl.read_parquet(path)
        pools = sorted(raw['table_id'].unique().to_list())
        pieces = []
        for offset in range(0, len(pools), 5):
            part = raw.filter(pl.col('table_id').is_in(pools[offset:offset + 5])).sort('pair_id', 'hand_id')
            part = part.with_columns(pl.Series('tie', np.random.default_rng(931207 + index * 100 + offset).random(part.height)))
            frame = part.join(added, on=['pair_id', 'hand_id'], validate='1:1').join(
                source.select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
            if not frame.height:
                continue
            frame = family_columns(context(frame, [c for c in bundles['v6_control']['raw_features'] if c.startswith('v2_')] + SEQ + EQ), 'predicted_behavior')
            scores = {name: bundle['model'].predict(matrix(frame, bundle['features']), num_threads=2) for name, bundle in bundles.items()}
            fused = fusion(frame, scores, list(models))
            assert np.isfinite(fused).all()
            pieces.append(rank(frame, fused))
            del part, frame, scores, fused
            gc.collect()
        pl.concat(pieces).write_parquet(destination)
        write_json(receipt, {'plan_sha256': sha(plan_path), 'output_sha256': sha(destination)})
        print(f'Ensemble batch {index + 1}/20 ranked', flush=True)
        del raw, added, pieces
        gc.collect()
    ranked = pl.read_parquet(sorted(cache.glob('batch_*.parquet')))
    assert ranked.height == ranked['pair_id'].n_unique() == TOP_PAIRS
    expanded = ranked.select('pair_id', *[pl.col('hands').list.get(i, null_on_oob=True).fill_null('NO_EVIDENCE').alias(c + '_new') for i, c in enumerate(EVIDENCE_COLUMNS)])
    result = source.join(expanded, on='pair_id', how='left', maintain_order='left').with_columns(
        *[pl.coalesce(c + '_new', c).alias(c) for c in EVIDENCE_COLUMNS]).select(source.columns)
    fixed = ['pair_id', 'risk_score', 'predicted_behavior']
    selected = pl.read_parquet(REPORT / 'selected-pairs.parquet')
    assert result.select(fixed).equals(source.select(fixed))
    assert result.join(selected, on='pair_id', how='anti').equals(source.join(selected, on='pair_id', how='anti'))
    destination = OUT / 'submission_ensemble.csv'
    if destination.exists():
        assert pl.read_csv(destination, infer_schema=False).equals(result)
    else:
        result.write_csv(destination)
    assert identity == {p: sha(ROOT / p) for p in identity}
    changed = sum(a != b for a, b in zip(result.select(EVIDENCE_COLUMNS).rows(), source.select(EVIDENCE_COLUMNS).rows()))
    write_json(OUT / 'provenance.json', {'inputs': identity, 'submission_sha256': sha(destination), 'rows': result.height,
                                       'reranked_pairs': TOP_PAIRS, 'changed_evidence_rows': changed,
                                       'preserved_exactly': fixed, 'other_evidence_unchanged': True, 'validation_required': True})
    print(json.dumps({'file': str(destination), 'changed_evidence_rows': changed}), flush=True)


if __name__ == '__main__':
    main()
