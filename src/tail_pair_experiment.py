"""Fixed, private comparison of repeated extreme events against current pair features.

No unknown pair labels, final model, or submission is produced. All augmented
views of a pair remain in the same player-disjoint pool partition.
"""
from __future__ import annotations

import os
for variable in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '2'

import json
from datetime import datetime, timezone
from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit
from robust_pair_experiment import (ROOT, SOURCE, DATA, LABELS, FAMILIES, KEYS,
    TRAIN_VIEWS, HELDOUT_VIEW, MODEL_PARAMS, thin, aggregate, matrix, metrics,
    assert_disjoint, sha, stable_ap)

REPORT = ROOT / 'reports' / 'tail-pair-experiment'


def tails(frame, columns):
    return frame.group_by('pair_id', 'table_id').agg(
        *[pl.col(c).top_k(3).mean().alias(c + '_top3_mean') for c in columns],
        *[(pl.col(c) > 0).mean().alias(c + '_nonzero_fraction') for c in columns])


def evaluate(frame, probabilities):
    y = np.array([LABELS[name] for name in frame['behavior_family']])
    risk = 1 - probabilities[:, 0]
    prediction = np.argmax(probabilities[:, 1:], axis=1) + 1
    result = metrics(y, risk)
    result['behavior_map'] = float(np.mean([
        stable_ap(y == i, np.where(prediction == i, risk, 0)) for i in range(1, 4)]))
    result['families'] = {}
    for i, name in enumerate(FAMILIES[1:], 1):
        selected = (y == 0) | (y == i)
        result['families'][name] = metrics(y[selected], risk[selected])
    return result


def run():
    REPORT.mkdir(exist_ok=True)
    if (REPORT / 'metrics.json').exists():
        raise FileExistsError('Experiment already completed; do not overwrite results')
    # Verify aggregation against a direct, independently computed tiny example.
    example = pl.DataFrame({'pair_id': ['a'] * 5 + ['b'],
        'table_id': ['t1'] * 5 + ['t2'], 'v2_event': [0., 1., 9., 2., 0., 4.]})
    check = tails(example, ['v2_event']).sort('pair_id')
    np.testing.assert_allclose(check['v2_event_top3_mean'], [4., 4.])
    np.testing.assert_allclose(check['v2_event_nonzero_fraction'], [.6, 1.])
    assert check.equals(tails(example.reverse(), ['v2_event']).sort('pair_id'))
    labels = pl.read_csv(DATA / 'development_labels.csv')
    assert labels['pair_id'].n_unique() == labels.height
    assert not labels.filter(~pl.col('label_status').is_in(
        ['confirmed_target', 'confirmed_non_target'])).height
    full = pl.read_parquet(SOURCE / 'pairs_development.parquet').join(labels, on='pair_id').sort('pair_id')
    raw = pl.concat([pl.read_parquet(path) for path in
                    sorted((SOURCE / 'hands_development').glob('batch_*.parquet'))])
    names = [c for c in raw.columns if c not in KEYS]
    selected = [c for c in names if c.startswith('v2_')]
    assert len(selected) == 56
    assert not raw.select('pair_id', 'table_id').unique().join(
        full.select('pair_id', 'table_id'), on=['pair_id', 'table_id'], how='anti').height
    original_counts = raw.group_by('pair_id').len().join(full.select('pair_id', 'shared_hands'), on='pair_id')
    assert original_counts.height == labels.height
    assert not original_counts.filter(pl.col('len') != pl.col('shared_hands')).height
    base_features = [c for c in full.columns if c not in KEYS +
        ['player_1', 'player_2', 'label', 'label_status', 'behavior_family']]
    extra = tails(raw, selected)
    tail_features = [c for c in extra.columns if c not in KEYS]
    full = full.join(extra, on=['pair_id', 'table_id']).sort('pair_id')
    views = {}
    for fraction, seed in TRAIN_VIEWS + [HELDOUT_VIEW]:
        sample = thin(raw, fraction, seed)
        view = aggregate(sample, names).join(tails(sample, selected), on=['pair_id', 'table_id'])
        views[(fraction, seed)] = view.join(labels, on='pair_id').select(full.columns).sort('pair_id')
    y = np.array([LABELS[name] for name in full['behavior_family']])
    assert np.array_equal(y > 0, full['label'].to_numpy() == 1)
    splitter = GroupShuffleSplit(n_splits=3, test_size=.25, random_state=2901)
    results, predictions = [], []
    for fold, (train_ids, held_ids) in enumerate(splitter.split(full, y, full['table_id'])):
        train = full[train_ids.tolist()]
        held = full[held_ids.tolist()]
        train_tables = train['table_id'].unique().to_list()
        held_tables = held['table_id'].unique().to_list()
        augmented = pl.concat([train] + [views[key].filter(pl.col('table_id').is_in(train_tables))
                                        for key in TRAIN_VIEWS])
        assert_disjoint(augmented, held)
        assert not augmented.group_by('pair_id').len().filter(pl.col('len') != 3).height
        validation = {'full': held,
            'shortened': views[HELDOUT_VIEW].filter(pl.col('table_id').is_in(held_tables)).sort('pair_id')}
        for candidate, features in [('current_augmented', base_features),
                                    ('augmented_with_event_tails', base_features + tail_features)]:
            model = lgb.LGBMClassifier(**MODEL_PARAMS)
            target = np.array([LABELS[name] for name in augmented['behavior_family']])
            model.fit(matrix(augmented, features), target,
                      sample_weight=np.full(augmented.height, 1 / 3))
            assert np.array_equal(model.classes_, np.arange(4))
            result = {'fold': fold, 'candidate': candidate, 'feature_count': len(features),
                      'train_pairs': train.height, 'held_pairs': held.height}
            for history, frame in validation.items():
                probabilities = model.predict_proba(matrix(frame, features))
                result[history] = evaluate(frame, probabilities)
                predictions.append(frame.select('pair_id', 'table_id', 'behavior_family', 'shared_hands')
                    .with_columns(pl.lit(fold).alias('fold'), pl.lit(candidate).alias('candidate'),
                        pl.lit(history).alias('history'), pl.Series('risk', 1-probabilities[:, 0]),
                        pl.Series('predicted_class', np.argmax(probabilities[:, 1:], axis=1)+1)))
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
        'model_params': MODEL_PARAMS, 'base_features': base_features, 'extra_features': tail_features,
        'self_check_passed': True, 'rows': raw.height, 'labeled_pairs': full.height,
        'pair_table_player_overlap': False, 'split_seed': 2901,
        'training_views': [[1, None], *TRAIN_VIEWS], 'validation_short_view': HELDOUT_VIEW,
        'results': results,
        'limitations': ['Exploratory reused, overlapping group holdouts, not independent final test sets.',
            'Unknown development pairs remain unknown and are excluded from targets.',
            'One-percent prevalence metric is a sensitivity assumption, not an estimated test prior.',
            'Thinning retains positive pair labels even if sampled hands omit the suspicious behavior.',
            'No public score was used to choose the feature formula; no final model or submission is produced.']}
    (REPORT / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    pl.concat(predictions).write_csv(REPORT / 'predictions.csv')


if __name__ == '__main__':
    run()
