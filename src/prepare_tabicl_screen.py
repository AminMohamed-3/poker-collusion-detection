"""Export one train-only feature selection for a pretrained-model screen."""
import os
for key in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import json
from datetime import datetime, timezone
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit
from tail_pair_experiment import tails
from robust_pair_experiment import ROOT, LABELS, TRAIN_VIEWS, HELDOUT_VIEW, MODEL_PARAMS, thin, aggregate, assert_disjoint, matrix, sha

OUT = ROOT / 'reports/tabicl-screen'


def main():
    OUT.mkdir(exist_ok=False)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
            'checkpoint': 'tabicl-classifier-v2-20260212.ckpt', 'n_estimators': 2, 'batch_size': 1,
            'feature_selection': 'Top 128 LightGBM gain features fitted on training pools only.',
            'context': 'Original training pairs only, four trusted classes; no unknown-pair negative labels.',
            'validation': 'Reused fold0 full and 65% histories; no feature/model tuning on these results.',
            'comparisons': ['Unchanged v5 control risk', 'TabICL risk', 'Equal arithmetic mean of both risks'],
            'next_gate': 'Advance if shortened sparse-prior AP improves without either ordinary AP losing more than .003.',
            'limitations': ['Cached thinning retains original player baselines.', 'Known-family split cannot establish hidden-family accuracy.']}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    labels = pl.read_csv(ROOT / 'data/development_labels.csv')
    raw = pl.read_parquet(sorted((ROOT / 'outputs_v2/hands_development').glob('batch_*.parquet')))
    names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    events = [c for c in names if c.startswith('v2_')]
    meta = json.loads((ROOT / 'reports/tail-pair-experiment/metrics.json').read_text())
    features = meta['base_features'] + meta['extra_features']
    full = pl.read_parquet(ROOT / 'outputs_v2/pairs_development.parquet').join(tails(raw, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').sort('pair_id')
    split = GroupShuffleSplit(n_splits=3, test_size=.25, random_state=2901)
    train_ids, held_ids = next(split.split(full, full['label'], full['table_id']))
    train, held = full[train_ids.tolist()], full[held_ids.tolist()]
    assert_disjoint(train, held)
    augmented = [train]
    for fraction, seed in TRAIN_VIEWS:
        sampled = thin(raw.filter(pl.col('table_id').is_in(train['table_id'].unique().to_list())), fraction, seed)
        augmented.append(aggregate(sampled, names).join(tails(sampled, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').select(full.columns))
    augmented = pl.concat(augmented)
    target = np.array([LABELS[f] for f in augmented['behavior_family']])
    model = lgb.LGBMClassifier(**MODEL_PARAMS)
    model.fit(matrix(augmented, features), target, sample_weight=np.full(augmented.height, 1 / 3))
    chosen_indices = np.argsort(-model.booster_.feature_importance('gain'), kind='stable')[:128]
    chosen = [features[i] for i in chosen_indices]
    sampled = thin(raw.filter(pl.col('table_id').is_in(held['table_id'].unique().to_list())), *HELDOUT_VIEW)
    short = aggregate(sampled, names).join(tails(sampled, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').sort('pair_id')
    assert held['pair_id'].to_list() == short['pair_id'].to_list()
    arrays = {'x_train': matrix(train, chosen), 'y_train': np.array([LABELS[f] for f in train['behavior_family']]),
              'y_held': np.array([LABELS[f] for f in held['behavior_family']])}
    old = pl.read_csv(ROOT / 'reports/tail-pair-experiment/predictions.csv').filter(
        (pl.col('fold') == 0) & (pl.col('candidate') == 'augmented_with_event_tails'))
    for history, frame in [('full', held), ('shortened', short)]:
        probabilities = model.predict_proba(matrix(frame, features))
        expected = old.filter(pl.col('history') == history).sort('pair_id')
        np.testing.assert_array_equal(expected['risk'].to_numpy(), 1 - probabilities[:, 0])
        arrays['x_' + history] = matrix(frame, chosen)
        arrays['control_' + history] = 1 - probabilities[:, 0]
    np.savez_compressed(OUT / 'arrays.npz', **arrays)
    held.select('pair_id', 'table_id', 'behavior_family').write_csv(OUT / 'held-pairs.csv')
    (OUT / 'features.json').write_text(json.dumps({'features': chosen, 'selection': 'train-only gain',
        'train_pairs': train.height, 'held_pairs': held.height, 'features_total': len(features),
        'control_risks_reproduced_exactly': True, 'array_sha256': sha(OUT / 'arrays.npz')}, indent=2) + '\n')
    print(json.dumps({'train': train.height, 'held': held.height, 'features': len(chosen), 'v5_control_exact': True}))


if __name__ == '__main__':
    main()
