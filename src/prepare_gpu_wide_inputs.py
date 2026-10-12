"""Rebuild train-only feature rankings for a fixed wider-feature GPU comparison."""
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

OUT = ROOT / 'reports/gpu-wide-experiment'


def main():
    OUT.mkdir(exist_ok=False)
    (OUT / 'preparation-plan.json').write_text(json.dumps({'created_at': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(__file__), 'feature_counts': [256, 765],
        'objective': 'Four-class TabICL with two estimators, changing only the number of train-selected gameplay features.',
        'validation': 'Reuse all six outer pool/family splits. Confirm first 128 reconstructed feature columns and v5 controls against immutable prior arrays before inference.',
        'inner_finetune': 'Additionally prepare a 15-percent inner pool holdout, with a separate 128-feature selector fitted without inner validation labels. Future fine-tuning must use inner validation, never the outer held labels.'}, indent=2) + '\n')
    labels = pl.read_csv(ROOT / 'data/development_labels.csv')
    raw = pl.read_parquet(sorted((ROOT / 'outputs_v2/hands_development').glob('batch_*.parquet')))
    names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    events = [c for c in names if c.startswith('v2_')]
    meta = json.loads((ROOT / 'reports/tail-pair-experiment/metrics.json').read_text())
    features = meta['base_features'] + meta['extra_features']
    full = pl.read_parquet(ROOT / 'outputs_v2/pairs_development.parquet').join(tails(raw, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').sort('pair_id')
    views = {}
    for fraction, seed in TRAIN_VIEWS + [HELDOUT_VIEW]:
        sampled = thin(raw, fraction, seed)
        views[(fraction, seed)] = aggregate(sampled, names).join(tails(sampled, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').select(full.columns).sort('pair_id')
    splits = list(GroupShuffleSplit(n_splits=3, test_size=.25, random_state=2901).split(full, full['label'], full['table_id']))
    jobs = [('fold' + str(i), i, None) for i in range(3)] + [('omit_' + f, 0, f) for f in list(LABELS)[1:]]
    for job, fold, omitted in jobs:
        directory = OUT / 'inputs' / job
        directory.mkdir(parents=True)
        train_ids, held_ids = splits[fold]
        train, held = full[train_ids.tolist()], full[held_ids.tolist()]
        train_pools, held_pools = train['table_id'].unique().to_list(), held['table_id'].unique().to_list()
        augmented = pl.concat([train] + [views[k].filter(pl.col('table_id').is_in(train_pools)) for k in TRAIN_VIEWS])
        short = views[HELDOUT_VIEW].filter(pl.col('table_id').is_in(held_pools)).sort('pair_id')
        if omitted:
            train = train.filter(pl.col('behavior_family') != omitted)
            augmented = augmented.filter(pl.col('behavior_family') != omitted)
            held = held.filter(pl.col('behavior_family').is_in(['none', omitted]))
            short = short.filter(pl.col('behavior_family').is_in(['none', omitted]))
        assert_disjoint(augmented, held)
        selector = lgb.LGBMClassifier(**MODEL_PARAMS)
        selector.fit(matrix(augmented, features), np.array([LABELS[f] for f in augmented['behavior_family']]), sample_weight=np.full(augmented.height, 1/3))
        ranked = [features[i] for i in np.argsort(-selector.booster_.feature_importance('gain'), kind='stable')]
        arrays = {'x_train': matrix(train, ranked), 'y_train': np.array([LABELS[f] for f in train['behavior_family']]),
                  'y_held': np.array([LABELS[f] for f in held['behavior_family']])}
        reference_dir = ROOT / ('reports/tabicl-screen' if job == 'fold0' else 'reports/tabicl-confirmation/' + job)
        old = np.load(reference_dir / 'arrays.npz', allow_pickle=False)
        old_predictions = np.load(reference_dir / 'predictions.npz', allow_pickle=False)
        np.testing.assert_array_equal(arrays['x_train'][:, :128], old['x_train'])
        np.testing.assert_array_equal(arrays['y_train'], old['y_train'])
        np.testing.assert_array_equal(arrays['y_held'], old['y_held'])
        for history, frame in [('full', held), ('shortened', short)]:
            arrays['x_' + history] = matrix(frame, ranked)
            arrays['control_' + history] = 1 - selector.predict_proba(matrix(frame, features))[:, 0]
            arrays['cpu_' + history] = old_predictions[history]
            np.testing.assert_array_equal(arrays['control_' + history], old['control_' + history])
            np.testing.assert_array_equal(arrays['x_' + history][:, :128], old['x_' + history])
        inner_train_ids, inner_val_ids = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=8741).split(train, groups=train['table_id']))
        inner_train, inner_val = train[inner_train_ids.tolist()], train[inner_val_ids.tolist()]
        inner_aug = augmented.filter(pl.col('table_id').is_in(inner_train['table_id'].unique()))
        assert_disjoint(inner_aug, inner_val)
        inner_selector = lgb.LGBMClassifier(**MODEL_PARAMS)
        inner_selector.fit(matrix(inner_aug, features), np.array([LABELS[f] for f in inner_aug['behavior_family']]), sample_weight=np.full(inner_aug.height, 1/3))
        inner_features = [features[i] for i in np.argsort(-inner_selector.booster_.feature_importance('gain'), kind='stable')[:128]]
        for suffix, frame in [('train', inner_train), ('val', inner_val), ('full', held), ('shortened', short)]:
            arrays['ft_x_' + suffix] = matrix(frame, inner_features)
            if suffix in ['train', 'val']:
                arrays['ft_y_' + suffix] = np.array([LABELS[f] for f in frame['behavior_family']])
        np.savez_compressed(directory / 'arrays.npz', **arrays)
        held.select('pair_id', 'table_id', 'behavior_family').write_csv(directory / 'held-pairs.csv')
        (directory / 'features.json').write_text(json.dumps({'features': ranked, 'inner_features': inner_features,
            'inner_train_pools': sorted(inner_train['table_id'].unique()), 'inner_val_pools': sorted(inner_val['table_id'].unique()),
            'fold': fold, 'omitted': omitted, 'array_sha256': sha(directory / 'arrays.npz'), 'first_128_and_controls_reproduced_exactly': True}, indent=2) + '\n')
        print(json.dumps({'prepared': job, 'training_pairs': train.height, 'features': len(ranked)}), flush=True)


if __name__ == '__main__':
    main()
