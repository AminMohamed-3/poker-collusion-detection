"""Confirm a post-hoc soft-play evidence gate on the remaining pool partition."""
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import tarfile
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from evidence_capacity_experiment import ROOT, load_sequence, fit_variant
from equity_evidence_features import FEATURES as EQ
from rank_evidence import context, family_columns, matrix, PARAMS, FAMILIES
from robust_pair_experiment import LABELS, TRAIN_VIEWS, MODEL_PARAMS, thin, aggregate

OUT = ROOT / 'reports/soft-gate-confirmation'


def main():
    OUT.mkdir(exist_ok=False)
    data, current, truths, _, pools0 = load_sequence()
    pools1 = json.loads((ROOT / 'reports/sequence-evidence-experiment/confirmation-v2/start.json').read_text())['held_pools']
    training_pools = sorted(set(pools0) | set(pools1))
    labels = pl.read_csv(ROOT / 'data/development_labels.csv')
    pairs = pl.read_parquet(ROOT / 'outputs_v2/pairs_development.parquet').join(labels, on='pair_id').sort('pair_id')
    third_pools = sorted(set(pairs['table_id']) - set(training_pools))
    assert not set(pools0) & set(pools1) and not set(third_pools) & set(training_pools)
    (OUT / 'preparation-plan.json').write_text(json.dumps({'created_at': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'Apply fixed equal reciprocal-rank fusion only to predicted soft_play; retain equity elsewhere.',
        'training_pools': training_pools, 'held_pools': third_pools,
        'gate': 'Third-partition MAP gain at least .005, no family decline exceeding .025; do not change recipe after outcomes.',
        'scope': 'Held pools are disjoint from the two evidence-screen partitions, but broader prior research used some of these pools. Not an untouched external test.',
        'pair_behavior': 'Fit the original 653-feature, augmented-history family recipe on training pools only. No all-label pair model for this validation.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    base_features = json.loads((ROOT / 'reports/tail-pair-experiment/metrics.json').read_text())['base_features']
    assert len(base_features) == 653
    raw = pl.read_parquet(sorted((ROOT / 'outputs_v2/hands_development').glob('batch_*.parquet')))
    hand_features = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    training = pairs.filter(pl.col('table_id').is_in(training_pools))
    augmented = [training]
    for fraction, seed in TRAIN_VIEWS:
        frame = aggregate(thin(raw, fraction, seed), hand_features).join(labels, on='pair_id').sort('pair_id')
        augmented.append(frame.filter(pl.col('table_id').is_in(training_pools)).select(training.columns))
    augmented = pl.concat(augmented)
    pair_model = lgb.LGBMClassifier(**MODEL_PARAMS)
    pair_model.fit(matrix(augmented, base_features), np.array([LABELS[f] for f in augmented['behavior_family']]), sample_weight=np.full(augmented.height, 1/3))
    held_pairs = pairs.filter(pl.col('table_id').is_in(third_pools))
    probability = pair_model.predict_proba(matrix(held_pairs, base_features))
    assert np.array_equal(pair_model.classes_, np.arange(4))
    behavior = held_pairs.select('pair_id').with_columns(pl.Series('predicted_behavior', np.array(FAMILIES)[probability[:, 1:].argmax(axis=1)]))
    behavior.write_csv(OUT / 'held-pair-behaviors.csv')
    joblib.dump({'model': pair_model, 'features': base_features}, OUT / 'pair-model.joblib')
    data = context(data.join(pl.read_parquet(ROOT / 'reports/equity-evidence-experiment/development.parquet'), on=['pair_id', 'hand_id'], validate='1:1'), EQ)
    features = current + EQ + ['context_' + c for c in EQ]
    train = data.filter(pl.col('table_id').is_in(training_pools)).sort('pair_id', 'hand_id')
    held = data.filter(pl.col('table_id').is_in(third_pools)).join(behavior, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
    assert train['pair_id'].n_unique() + held['pair_id'].n_unique() == 372
    eq = fit_variant(train, features, PARAMS)
    seq = fit_variant(train, current, PARAMS)
    joblib.dump({'model': eq, 'features': features}, OUT / 'equity-model.joblib')
    joblib.dump({'model': seq, 'features': current}, OUT / 'sequence-model.joblib')
    selected = [features[i] for i in np.argsort(-eq.booster_.feature_importance('gain'), kind='stable')[:128]]
    train = family_columns(train, 'behavior_family').with_row_index('row')
    train = train.with_columns(pl.Series('selection_score', eq.predict(matrix(train, features), num_threads=2)))
    rng = np.random.default_rng(238451); chosen = []
    for _, group in train.group_by('pair_id', maintain_order=True):
        positives = group.filter(pl.col('relevance') == 1)['row'].to_list()
        unlisted = group.filter(pl.col('relevance') == 0)
        hard = unlisted.sort('selection_score', descending=True).head(4)['row'].to_list()
        remaining = unlisted.filter(~pl.col('row').is_in(hard))['row'].to_numpy()
        chosen.extend(positives + hard + rng.choice(remaining, min(4, len(remaining)), replace=False).tolist())
    selected_train = train[sorted(chosen)]
    assert len(chosen) == len(set(chosen)) and selected_train['relevance'].sum() == train['relevance'].sum()
    held_features = family_columns(held, 'predicted_behavior')
    folder = OUT / 'inputs/third'; folder.mkdir(parents=True)
    np.savez_compressed(folder / 'arrays.npz', x_train=matrix(selected_train, selected), y_train=selected_train['relevance'].to_numpy(),
        x_held=matrix(held_features, selected), equity=eq.predict(matrix(held_features, features), num_threads=2),
        sequence=seq.predict(matrix(held_features, current), num_threads=2))
    held.select('pair_id', 'hand_id', 'table_id', 'behavior_family', 'relevance', 'predicted_behavior').write_csv(folder / 'held-rows.csv')
    selected_train.select('pair_id', 'hand_id', 'table_id', 'relevance').write_csv(folder / 'context-rows.csv')
    (folder / 'features.json').write_text(json.dumps({'features': selected, 'context_rows': selected_train.height,
        'held_rows': held.height, 'training_pairs': train['pair_id'].n_unique(), 'held_pairs': held['pair_id'].n_unique()}, indent=2) + '\n')
    source = (ROOT / 'src/run_tabicl_evidence_experiment.py').read_text()
    source = source.replace("scores['rrf_equity_tabicl'] = fusion", "scores['rrf_equity_tabicl'] = fusion\n        scores['soft_gate'] = np.where(np.array([r['predicted_behavior'] for r in rows]) == 'soft_play', fusion, scores['equity'])")
    source = source.replace('np.mean(deltas) >= .01', 'np.mean(deltas) >= .005')
    script = OUT / 'run_soft_gate_confirmation.py'; script.write_text(source)
    files = {p.relative_to(OUT).as_posix(): p for p in (OUT / 'inputs').rglob('*') if p.is_file()}
    files[script.name] = script; files['preparation-plan.json'] = OUT / 'preparation-plan.json'
    parent = json.loads((ROOT / 'reports/tabicl-evidence-experiment/plan.json').read_text())
    plan = {k: v for k, v in parent.items() if k.startswith('checkpoint_') or k in ['parameters','context','inference_chunk_rows','rrf_k']}
    plan.update({'jobs': ['third'], 'candidates': ['soft_gate'], 'selection': 'Third MAP gain >=.005 and no family loss beyond .025.',
                 'files': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items()}})
    plan_path = OUT / 'plan.json'; plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    archive = OUT / 'confirmation-bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        t.add(plan_path, arcname='plan.json')
        for name, path in files.items(): t.add(path, arcname=name)
    receipt = {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size}
    (OUT / 'bundle-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'bundle': receipt, 'context_rows': selected_train.height, 'held_rows': held.height, 'held_pairs': held['pair_id'].n_unique()}), flush=True)


if __name__ == '__main__':
    main()
