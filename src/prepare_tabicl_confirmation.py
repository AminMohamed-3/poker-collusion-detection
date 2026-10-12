"""Two more fixed pool folds and three omitted-family tests, with train-only selection."""
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

OUT = ROOT / 'reports/tabicl-confirmation'


def main():
    OUT.mkdir(exist_ok=False)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
            'checkpoint': 'tabicl-classifier-v2-20260212.ckpt', 'estimators': 2, 'feature_count': 128,
            'jobs': ['fold1', 'fold2', 'omit_directed_transfer', 'omit_soft_play', 'omit_coordinated_isolation'],
            'gate': 'For a risk-only candidate, no known fold/history AP decline >0.003; positive average sparse-prior gain; omitted-family mean sparse gain nonnegative and no omitted sparse decline >0.03.',
            'candidate_options': ['tabicl', 'equal_blend'],
            'selection': 'Among passing options choose the largest average shortened sparse-prior AP across known folds.',
            'scope': 'Previously used pool splits; omit a family from feature selection and model context entirely, testing it against confirmed-none pairs.'}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
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
    jobs = [('fold1', 1, None), ('fold2', 2, None)] + [('omit_' + f, 0, f) for f in list(LABELS)[1:]]
    old = pl.read_csv(ROOT / 'reports/tail-pair-experiment/predictions.csv').filter(pl.col('candidate') == 'augmented_with_event_tails')
    for name, fold, omitted in jobs:
        directory = OUT / name
        directory.mkdir()
        train_ids, held_ids = splits[fold]
        train, held = full[train_ids.tolist()], full[held_ids.tolist()]
        train_pools, held_pools = train['table_id'].unique().to_list(), held['table_id'].unique().to_list()
        augmented = pl.concat([train] + [views[key].filter(pl.col('table_id').is_in(train_pools)) for key in TRAIN_VIEWS])
        short = views[HELDOUT_VIEW].filter(pl.col('table_id').is_in(held_pools)).sort('pair_id')
        if omitted:
            train = train.filter(pl.col('behavior_family') != omitted)
            augmented = augmented.filter(pl.col('behavior_family') != omitted)
            held = held.filter(pl.col('behavior_family').is_in(['none', omitted]))
            short = short.filter(pl.col('behavior_family').is_in(['none', omitted]))
        assert_disjoint(augmented, held)
        assert held['pair_id'].to_list() == short['pair_id'].to_list()
        model = lgb.LGBMClassifier(**MODEL_PARAMS)
        model.fit(matrix(augmented, features), np.array([LABELS[f] for f in augmented['behavior_family']]), sample_weight=np.full(augmented.height, 1 / 3))
        assert model.classes_[0] == 0
        chosen = [features[i] for i in np.argsort(-model.booster_.feature_importance('gain'), kind='stable')[:128]]
        arrays = {'x_train': matrix(train, chosen), 'y_train': np.array([LABELS[f] for f in train['behavior_family']]),
                  'y_held': np.array([LABELS[f] for f in held['behavior_family']])}
        for history, frame in [('full', held), ('shortened', short)]:
            control = 1 - model.predict_proba(matrix(frame, features))[:, 0]
            if not omitted:
                expected = old.filter((pl.col('fold') == fold) & (pl.col('history') == history)).sort('pair_id')
                np.testing.assert_array_equal(control, expected['risk'].to_numpy())
            arrays['x_' + history] = matrix(frame, chosen)
            arrays['control_' + history] = control
        np.savez_compressed(directory / 'arrays.npz', **arrays)
        held.select('pair_id', 'table_id', 'behavior_family').write_csv(directory / 'held-pairs.csv')
        (directory / 'features.json').write_text(json.dumps({'features': chosen, 'fold': fold, 'omitted': omitted,
            'training_pairs': train.height, 'held_pairs': held.height, 'array_sha256': sha(directory / 'arrays.npz')}, indent=2) + '\n')
        print(json.dumps({'prepared': name, 'training': train.height, 'held': held.height}), flush=True)


if __name__ == '__main__':
    main()
