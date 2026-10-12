"""One fixed partner-selectivity candidate versus a matched robust fold-0 control."""
import os
for key in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'

import gc
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit
from novelty_experiment import score_report
from robust_pair_experiment import (FAMILIES, LABELS, KEYS, MODEL_PARAMS, TRAIN_VIEWS,
    HELDOUT_VIEW, aggregate, thin, matrix, assert_disjoint, sha, peak_memory_bytes, stable_ap)

ROOT = Path(__file__).resolve().parents[1]
DATA, BASE, REPORT = ROOT/'data', ROOT/'outputs_v2', ROOT/'reports'
SCOPES = ['all', 'postflop']
MEASURES = ['fold', 'call', 'raise', 'cheap_fold', 'call_commit']
PSEUDO = 5.


def response_stats(keys, actions):
    """Only actual decisions facing a prior opponent aggressor count as trials."""
    actions = actions.sort('hand_id', 'action_no').with_columns(
        (pl.col('action').is_in(['bet', 'raise', 'all_in']) & (pl.col('amount') > pl.col('to_call'))).alias('aggressive'))
    actions = actions.with_columns(pl.when('aggressive').then('player_id').otherwise(None)
        .forward_fill().shift(1).over('hand_id', 'street').alias('previous_aggressor'))
    actions = actions.filter((pl.col('to_call') > 0) & pl.col('previous_aggressor').is_not_null() &
                              (pl.col('previous_aggressor') != pl.col('player_id')))
    directions = []
    for side, other in [(1, 2), (2, 1)]:
        mapping = keys.select('pair_id', 'hand_id', pl.col(f'player_{side}').alias('player_id'),
                              pl.col(f'player_{other}').alias('partner_id'))
        events = mapping.join(actions, on=['hand_id', 'player_id'], how='inner')
        call = pl.col('action').is_in(['call', 'all_in']) & ~pl.col('aggressive')
        expressions = []
        for scope in SCOPES:
            scope_mask = pl.lit(True) if scope == 'all' else pl.col('street') != 'preflop'
            for relation in ['partner', 'other']:
                same = pl.col('previous_aggressor') == pl.col('partner_id')
                mask = scope_mask & (same if relation == 'partner' else ~same)
                prefix = f'{scope}_{relation}_'
                values = {'trials': pl.lit(1.), 'fold': (pl.col('action') == 'fold').cast(pl.Float64),
                          'call': call.cast(pl.Float64), 'raise': pl.col('aggressive').cast(pl.Float64),
                          'cheap_fold': (pl.col('action') == 'fold').cast(pl.Float64) *
                              (1-pl.col('to_call')/(pl.col('pot_before')+pl.col('to_call')).clip(lower_bound=1)),
                          'call_commit': call.cast(pl.Float64) *
                              (pl.col('amount')/pl.col('stack_before').clip(lower_bound=1)).clip(0, 1)}
                expressions += [pl.when(mask).then(value).otherwise(0.).sum().alias(prefix+name)
                                for name, value in values.items()]
        grouped = events.group_by('pair_id', 'hand_id').agg(expressions).with_columns(pl.lit(side).alias('side'))
        directions.append(grouped)
    return pl.concat(directions)


def selectivity(stats, hand_keys):
    """Recompute both comparisons using only the supplied retained pair hands."""
    retained = stats.join(hand_keys.select('pair_id', 'hand_id'), on=['pair_id', 'hand_id'], how='inner')
    numeric = [c for c in stats.columns if c not in ['pair_id', 'hand_id', 'side']]
    summed = retained.group_by('pair_id', 'side').agg(pl.col(numeric).sum())
    complete = pl.concat([hand_keys.select('pair_id').unique().with_columns(pl.lit(side).alias('side'))
                          for side in [1, 2]])
    summed = complete.join(summed, on=['pair_id', 'side'], how='left').fill_null(0)
    contrasts = []
    for scope in SCOPES:
        n, m = pl.col(f'{scope}_partner_trials'), pl.col(f'{scope}_other_trials')
        for measure in MEASURES:
            a, b = pl.col(f'{scope}_partner_{measure}'), pl.col(f'{scope}_other_{measure}')
            pooled = (a+b+.5)/(n+m+1)
            delta = (a+PSEUDO*pooled)/(n+PSEUDO) - (b+PSEUDO*pooled)/(m+PSEUDO)
            contrasts.append(pl.when((n > 0) & (m > 0)).then(delta).otherwise(0.)
                             .alias(f'selectivity_{scope}_{measure}'))
    columns = [e.meta.output_name() for e in contrasts]
    return summed.with_columns(contrasts).group_by('pair_id').agg(
        *[getattr(pl.col(c), operation)().alias(c+'_'+operation) for c in columns
          for operation in ['mean', 'min', 'max']]).sort('pair_id')


def self_test():
    keys = pl.DataFrame({'pair_id': ['p'], 'hand_id': ['h'], 'player_1': ['a'], 'player_2': ['b']})
    actions = pl.DataFrame({'hand_id': ['h']*5, 'action_no': list(range(5)), 'street': ['preflop']*5,
        'player_id': ['b', 'a', 'c', 'b', 'a'], 'action': ['raise', 'call', 'raise', 'all_in', 'fold'],
        'amount': [4, 4, 12, 8, 0], 'to_call': [2, 4, 4, 8, 8],
        'pot_before': [3, 7, 11, 23, 31], 'stack_before': [100, 100, 100, 8, 96]})
    stats = response_stats(keys, actions)
    actor = stats.filter(pl.col('side') == 1).to_dicts()[0]
    partner = stats.filter(pl.col('side') == 2).to_dicts()[0]
    assert actor['all_partner_call'] == 1 and actor['all_other_fold'] == 1
    assert partner['all_other_call'] == 1 and partner['all_other_raise'] == 0
    result = selectivity(stats, keys)
    swapped = keys.with_columns(pl.col('player_1').alias('player_2'), pl.col('player_2').alias('player_1'))
    np.testing.assert_allclose(result.select(pl.exclude('pair_id')).to_numpy(),
        selectivity(response_stats(swapped, actions), swapped).select(pl.exclude('pair_id')).to_numpy(), atol=1e-12)
    assert np.isfinite(result.select(pl.exclude('pair_id')).to_numpy()).all()
    assert (result.select(pl.col('^selectivity_postflop_.*$')).to_numpy() == 0).all()


def run():
    start = time.perf_counter()
    self_test()
    paths = sorted((BASE/'hands_development').glob('batch_*.parquet'))
    protected = [BASE/'pair_model_robust.joblib', BASE/'hand_model.joblib', BASE/'submission_advanced_robust.csv',
                  ROOT/'outputs_v3/submission_evidence_ranker.csv', ROOT/'outputs_v3/evidence_ranker.joblib']
    input_paths = [DATA/'development_labels.csv', BASE/'pairs_development.parquet', BASE/'validation_pairs.csv', *paths]
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in [*input_paths, *protected]}
    raw_metadata = {name: {'size': (DATA/name).stat().st_size, 'mtime_ns': (DATA/name).stat().st_mtime_ns}
                    for name in ['actions.parquet', 'hands.parquet']}
    labels = pl.read_csv(DATA/'development_labels.csv')
    assert labels.height == labels['pair_id'].n_unique() == 1860
    assert set(labels['label_status']) == {'confirmed_target', 'confirmed_non_target'}
    full = pl.read_parquet(BASE/'pairs_development.parquet').join(labels, on='pair_id').sort('pair_id')
    excluded = set(KEYS+['player_1', 'player_2', 'label', 'label_status', 'behavior_family'])
    features = [c for c in full.columns if c not in excluded]
    assert len(features) == 653
    y = np.array([LABELS[f] for f in full['behavior_family']])
    assert np.array_equal(y > 0, full['label'].to_numpy() == 1)
    ti, hi = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=2901).split(full, y, full['table_id']))
    training, held = full[ti.tolist()], full[hi.tolist()]
    assert_disjoint(training, held)
    assert set(held['pair_id']) == set(pl.read_csv(BASE/'validation_pairs.csv')['pair_id'])
    raw = pl.read_parquet(paths)
    raw_features = [c for c in raw.columns if c not in KEYS]
    assert len(raw_features) == 163 and raw.select(pl.struct('pair_id', 'hand_id').n_unique()).item() == raw.height
    assert set(raw['pair_id']) == set(labels['pair_id'])
    keys = raw.select('pair_id', 'hand_id').join(labels.select('pair_id', 'player_1', 'player_2'), on='pair_id')
    actions = pl.scan_parquet(DATA/'actions.parquet').join(keys.select('hand_id').unique().lazy(),
        on='hand_id', how='inner').collect(engine='streaming')
    stats = response_stats(keys, actions)
    del actions, keys
    gc.collect()
    print(json.dumps({'response_stat_pair_hand_sides': stats.height, 'raw_pair_hands': raw.height}), flush=True)
    scopes = {'train_full': raw.filter(pl.col('table_id').is_in(training['table_id'].unique().to_list())),
              'held_full': raw.filter(pl.col('table_id').is_in(held['table_id'].unique().to_list()))}
    scopes['train_0.7'] = thin(scopes['train_full'], *TRAIN_VIEWS[0])
    scopes['train_0.4'] = thin(scopes['train_full'], *TRAIN_VIEWS[1])
    scopes['held_0.65'] = thin(scopes['held_full'], *HELDOUT_VIEW)
    views = {}
    for name, frame in scopes.items():
        baseline = (training if name == 'train_full' else held) if name.endswith('_full') else aggregate(frame, raw_features).join(labels, on='pair_id')
        extra = selectivity(stats, frame)
        views[name] = baseline.join(extra, on='pair_id', how='left').sort('pair_id')
        assert views[name].height == (training.height if name.startswith('train') else held.height)
        assert not views[name].select(pl.col('^selectivity_.*$')).null_count().sum_horizontal().sum()
    extras = [c for c in views['train_full'].columns if c.startswith('selectivity_')]
    assert len(extras) == 30
    augmented = pl.concat([views[n].select(views['train_full'].columns) for n in ['train_full', 'train_0.7', 'train_0.4']])
    assert_disjoint(augmented, views['held_full'])
    assert augmented.group_by('pair_id').len().filter(pl.col('len') != 3).height == 0
    truth = np.array([LABELS[f] for f in augmented['behavior_family']])
    weights = np.full(augmented.height, 1/3)
    results, predictions, importance = [], [], []
    for name, columns in [('robust_control', features), ('robust_selectivity', features+extras)]:
        model = lgb.LGBMClassifier(**MODEL_PARAMS)
        model.fit(matrix(augmented, columns), truth, sample_weight=weights)
        assert model.classes_.tolist() == [0, 1, 2, 3] and model.booster_.current_iteration() == 226
        result = {'candidate': name, 'features': len(columns), 'validation': {}}
        for history in ['held_full', 'held_0.65']:
            frame = views[history]
            actual = np.array([LABELS[f] for f in frame['behavior_family']])
            prob = model.predict_proba(matrix(frame, columns))
            risk = 1-prob[:, 0]
            family = 1+np.argmax(prob[:, 1:], axis=1)
            metrics = score_report(actual, risk)
            metrics['behavior_map'] = float(np.mean([stable_ap(actual == k, np.where(family == k, risk, 0)) for k in [1, 2, 3]]))
            result['validation'][history] = metrics
            predictions.append(frame.select('pair_id', 'table_id', 'label', 'behavior_family', 'shared_hands').with_columns(
                pl.lit(name).alias('candidate'), pl.lit(history).alias('history'), pl.Series('risk_score', risk)))
        results.append(result)
        importance += [{'candidate': name, 'feature': feature, 'gain': float(gain)} for feature, gain in
                       zip(columns, model.booster_.feature_importance(importance_type='gain')) if feature in extras]
        print(json.dumps(result), flush=True)
        del model
        gc.collect()
    for relative, fingerprint in fingerprints.items():
        assert sha(ROOT/relative) == fingerprint, relative
    for name, before in raw_metadata.items():
        path = DATA/name
        assert {'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} == before
    output = {'script_sha256': sha(__file__), 'input_and_protected_sha256': fingerprints, 'raw_file_metadata': raw_metadata,
        'protected_artifacts_unchanged': True, 'fitted_models': 2, 'features_added': extras, 'model_parameters': MODEL_PARAMS,
        'pseudo_responses': PSEUDO, 'training_pairs': training.height, 'heldout_pairs': held.height,
        'heldout_targets': int(held['label'].sum()), 'unlisted_training_pairs': 0, 'pair_table_player_overlap': False,
        'augmentation': [{'fraction': 1., 'weight': 1/3}, *[{'fraction': f, 'seed': seed, 'weight': 1/3} for f, seed in TRAIN_VIEWS]],
        'heldout_thinning': {'fraction': HELDOUT_VIEW[0], 'seed': HELDOUT_VIEW[1]},
        'selectivity_definition': 'For each player, compare actual response outcomes to the partner versus other prior aggressors in retained shared hands. Both means shrink toward the pooled rate by 5 pseudo-responses; no opportunity on either side means contrast zero. Symmetric mean/min/max over the two players.',
        'self_tests': {'opponent_attribution': True, 'all_in_call_not_raise': True, 'pair_symmetry': True, 'no_postflop_opportunities_zero': True},
        'results': results, 'new_feature_importance': sorted(importance, key=lambda x: -x['gain']),
        'elapsed_seconds': time.perf_counter()-start, 'peak_resident_memory_bytes': peak_memory_bytes(),
        'limitations': ['One fixed development holdout; no full-evaluation prediction or public-score tuning.',
            'Existing base features retain player-baseline residuals when histories are thinned; selectivity is recomputed on retained hands.',
            'Other opponents supply unlabeled behavioral context only, never negative training labels.',
            'The 1% prevalence diagnostic does not estimate evaluation prevalence.',
            'Response comparisons are not matched on hand strength or opponent range; existing features may absorb some confounding.']}
    (REPORT/'pair-diagnostic.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    pl.concat(predictions).write_csv(REPORT/'pair-diagnostic.predictions.csv')
    print(json.dumps({'elapsed_seconds': output['elapsed_seconds'], 'peak_MiB': output['peak_resident_memory_bytes']/2**20,
                      'protected_artifacts_unchanged': True}), flush=True)


if __name__ == '__main__':
    run()
