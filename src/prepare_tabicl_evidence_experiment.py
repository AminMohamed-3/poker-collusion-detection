"""Prepare a private pretrained hand-evidence classifier and fixed rank fusion."""
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile
import joblib
import numpy as np
import polars as pl
from evidence_capacity_experiment import ROOT, load_sequence
from equity_evidence_features import FEATURES as EQ
from rank_evidence import context, family_columns, matrix

OUT = ROOT / 'reports/tabicl-evidence-experiment'


def main():
    OUT.mkdir(exist_ok=False)
    data, current, truths, pairpred, pools0 = load_sequence()
    data = context(data.join(pl.read_parquet(ROOT / 'reports/equity-evidence-experiment/development.parquet'), on=['pair_id', 'hand_id'], validate='1:1'), EQ)
    features = current + EQ + ['context_' + c for c in EQ]
    previous = ROOT / 'reports/sequence-evidence-experiment/confirmation-v2'
    pools1 = json.loads((previous / 'start.json').read_text())['held_pools']
    assert not set(pools0) & set(pools1)
    jobs = [('screen', pools0, pairpred, ROOT / 'reports/equity-evidence-experiment/equity.joblib'),
            ('confirmation', pools1, pl.read_csv(previous / 'full-pair-predictions.csv'), ROOT / 'reports/evidence-ensemble-experiment/confirmation-equity.joblib')]
    for name, pools, pair_prediction, model_path in jobs:
        folder = OUT / 'inputs' / name
        folder.mkdir(parents=True)
        train = family_columns(data.filter(~pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id'), 'behavior_family')
        held = data.filter(pl.col('table_id').is_in(pools)).join(pair_prediction.select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        inputs = family_columns(held, 'predicted_behavior')
        model = joblib.load(model_path)
        assert model['features'] == features
        selected = [features[i] for i in np.argsort(-model['model'].booster_.feature_importance('gain'), kind='stable')[:128]]
        train = train.with_row_index('row').with_columns(pl.Series('selection_score', model['model'].predict(matrix(train, features), num_threads=2)))
        rng = np.random.default_rng(238451)
        chosen = []
        for _, group in train.group_by('pair_id', maintain_order=True):
            positives = group.filter(pl.col('relevance') == 1)['row'].to_list()
            unlisted = group.filter(pl.col('relevance') == 0)
            hard = unlisted.sort('selection_score', descending=True).head(4)['row'].to_list()
            remaining = unlisted.filter(~pl.col('row').is_in(hard))['row'].to_numpy()
            random_rows = rng.choice(remaining, min(4, len(remaining)), replace=False).tolist()
            chosen.extend(positives + hard + random_rows)
        selected_train = train[sorted(chosen)]
        assert len(chosen) == len(set(chosen))
        assert selected_train['relevance'].sum() == train['relevance'].sum()
        baseline = model['model'].predict(matrix(inputs, features), num_threads=2)
        prior = pl.read_parquet(ROOT / f'reports/evidence-ensemble-experiment/{name}-scores.parquet')
        np.testing.assert_array_equal(prior['equity'].to_numpy(), baseline)
        assert prior['pair_id'].to_list() == held['pair_id'].to_list() and prior['hand_id'].to_list() == held['hand_id'].to_list()
        np.savez_compressed(folder / 'arrays.npz', x_train=matrix(selected_train, selected), y_train=selected_train['relevance'].to_numpy(),
                            x_held=matrix(inputs, selected), equity=baseline, sequence=prior['v6_control'].to_numpy())
        selected_train.select('pair_id', 'hand_id', 'table_id', 'relevance').write_csv(folder / 'context-rows.csv')
        held.select('pair_id', 'hand_id', 'table_id', 'behavior_family', 'relevance').write_csv(folder / 'held-rows.csv')
        (folder / 'features.json').write_text(json.dumps({'features': selected, 'context_rows': selected_train.height,
             'listed_context_rows': int(selected_train['relevance'].sum()), 'held_rows': held.height,
             'equity_control_reproduced_exactly': True}, indent=2) + '\n')
        print(json.dumps({'job': name, 'context_rows': selected_train.height, 'held_rows': held.height}), flush=True)
    previous_plan = json.loads((ROOT / 'reports/gpu-objective-experiment/plan.json').read_text())
    plan = {k: v for k, v in previous_plan.items() if k.startswith('checkpoint_')}
    files = {p.relative_to(OUT).as_posix(): p for p in (OUT / 'inputs').rglob('*') if p.is_file()}
    files['run_tabicl_evidence_experiment.py'] = ROOT / 'src/run_tabicl_evidence_experiment.py'
    plan.update({'created_at': datetime.now(timezone.utc).isoformat(), 'jobs': ['screen', 'confirmation'],
        'parameters': {'n_estimators': 2, 'batch_size': 1, 'device': 'cuda', 'n_jobs': 4, 'random_state': 7329, 'kv_cache': True, 'use_amp': False, 'use_fa3': False},
        'context': 'All listed training evidence plus at most four high-ranked unlisted and four randomly sampled other unlisted hands per training pair. Unlisted rows have zero ranking relevance, not confirmed-negative status.',
        'feature_selection': 'Top 128 gameplay features by the equity ranker fitted on the training pools; held behavior comes from out-of-pool pair predictions.',
        'candidates': ['tabicl', 'rrf_equity_tabicl'], 'rrf_k': 10, 'inference_chunk_rows': 2048,
        'selection': 'Relative to equity control, pooled pair MAP@5 gain >=.01, neither split below -.005, and no family pooled decline below -.025. Highest passing pooled MAP is selected for review; no automatic submission.',
        'limitations': 'Two reused disjoint held-pool sets; exploratory and adaptively chosen method. No test labels, ID features, human evidence edits, or evaluation predictions.',
        'files': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items()}})
    plan_path = OUT / 'plan.json'; plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    archive = OUT / 'evidence-bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        t.add(plan_path, arcname='plan.json')
        for name, path in files.items(): t.add(path, arcname=name)
    receipt = {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size}
    (OUT / 'bundle-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
