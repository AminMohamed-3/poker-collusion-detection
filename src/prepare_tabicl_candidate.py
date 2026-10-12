"""Prepare full-data inputs for the unchanged pretrained-model recipe."""
import os
for key in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import gc
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import polars as pl
from tail_pair_experiment import tails
from robust_pair_experiment import ROOT, LABELS, matrix, sha
from sequence_evidence_candidate import verify_source_cache

OUT = ROOT / 'outputs_v10_tabicl'
SOURCE = ROOT / 'outputs_v8_equity/submission_equity.csv'


def main():
    confirmation = json.loads((ROOT / 'reports/tabicl-confirmation/metrics.json').read_text())
    if confirmation['selected'] is None:
        raise ValueError('No confirmed candidate selected')
    OUT.mkdir(exist_ok=False)
    verify_source_cache('development')
    verify_source_cache('evaluation')
    selected = confirmation['selected']
    protected = [SOURCE, ROOT / 'outputs_v6/submission_sequence_evidence.csv', ROOT / 'outputs_v5/pair_model_event_tails.joblib']
    paths = protected + [ROOT / 'src/prepare_tabicl_candidate.py', ROOT / 'src/run_tabicl_candidate.py',
                        ROOT / 'reports/tabicl-confirmation/metrics.json', ROOT / 'reports/tabicl-screen/checkpoint.json']
    identity = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    bundle = joblib.load(ROOT / 'outputs_v5/pair_model_event_tails.joblib')
    features = bundle['features']
    chosen = [features[i] for i in np.argsort(-bundle['model'].booster_.feature_importance('gain'), kind='stable')[:128]]
    labels = pl.read_csv(ROOT / 'data/development_labels.csv')
    raw = pl.read_parquet(sorted((ROOT / 'outputs_v2/hands_development').glob('batch_*.parquet')), columns=['pair_id', 'hand_id', 'table_id', *bundle['events']])
    train = pl.read_parquet(ROOT / 'outputs_v2/pairs_development.parquet').join(tails(raw, bundle['events']), on=['pair_id', 'table_id']).join(labels, on='pair_id').sort('pair_id')
    assert train.height == 1860
    arrays = {'x_train': matrix(train, chosen), 'y_train': np.array([LABELS[f] for f in train['behavior_family']])}
    del raw
    pairs = pl.read_parquet(ROOT / 'outputs_v2/pairs_evaluation.parquet')
    pieces = []
    for path in sorted((ROOT / 'outputs_v2/hands_evaluation').glob('batch_*.parquet')):
        raw = pl.read_parquet(path, columns=['pair_id', 'hand_id', 'table_id', *bundle['events']])
        extra = tails(raw, bundle['events'])
        frame = pairs.join(extra, on=['pair_id', 'table_id'])
        assert frame.height == extra.height
        pieces.append(frame.select('pair_id', *chosen))
        print(f'Prepared {path.name}', flush=True)
        del raw, extra, frame
        gc.collect()
    evaluation = pl.concat(pieces).sort('pair_id')
    assert evaluation.height == evaluation['pair_id'].n_unique() == 112540
    arrays['x_evaluation'] = matrix(evaluation, chosen)
    np.savez_compressed(OUT / 'arrays.npz', **arrays)
    evaluation.select('pair_id').write_csv(OUT / 'evaluation-order.csv')
    payload = {'created_at': datetime.now(timezone.utc).isoformat(), 'inputs': identity,
               'selected': selected, 'features': chosen, 'feature_selection': 'All-label v5 LightGBM gain, identical train-only procedure to validation.',
               'training_pairs': train.height, 'evaluation_pairs': evaluation.height,
               'array_sha256': sha(OUT / 'arrays.npz'), 'order_sha256': sha(OUT / 'evaluation-order.csv'),
               'parameters': {'n_estimators': 2, 'batch_size': 1, 'kv_cache': True, 'use_amp': False,
                              'random_state': 7329, 'device': 'cpu', 'n_jobs': 2},
               'source': str(SOURCE.relative_to(ROOT)), 'changed_column': 'risk_score',
               'inference_chunk_rows': 4096}
    assert identity == {p: sha(ROOT / p) for p in identity}
    (OUT / 'plan.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'prepared': selected, 'training_pairs': train.height, 'evaluation_pairs': evaluation.height}), flush=True)


if __name__ == '__main__':
    main()
