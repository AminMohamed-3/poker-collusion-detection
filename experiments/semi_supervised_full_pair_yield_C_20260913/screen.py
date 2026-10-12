"""Frozen full-corpus adapter for the first-salt pair-yield teachers.

This file deliberately has no fit path.  It may only load the five saved
teachers from semi_supervised_pair_yield_20260913, reproduce their old
sample predictions exactly, and (after that replay passes) score the larger
deterministic corpus.  No evidence labels, truth files, or student models are
read here.
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '1'

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parent
# Claude copy (Amin request, 2026-09-13); inputs read from the Codex tree read-only; outputs stay in OUT.
RESUME = Path('C:/Data/Projects/Money Making/work/poker/Codex Resume')
ROOT = RESUME.parent
FULL_CACHE = RESUME / 'research/semi_supervised_full_features_20260913'
OLD_CACHE = RESUME / 'research/semi_supervised_features_20260913'
PARENT = RESUME / 'experiments/semi_supervised_pair_yield_20260913'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(ROOT / 'src'))
from robust_pair_experiment import aggregate, matrix, thin  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402

IDENTITY = ['pair_id', 'table_id', 'player_1', 'player_2', 'label', 'behavior_family']
VIEWS = {'full': (1.0, None), 'v70': (.7, 7307)}
THRESHOLD = .95


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, value):
    path = Path(path)
    assert not path.exists(), f'refusing to overwrite {path}'
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def assert_sources(plan):
    for name, receipt in plan['frozen_sources'].items():
        path = Path(receipt['path'])
        assert path.exists(), f'missing frozen source {name}: {path}'
        assert sha(path) == receipt['sha256'], f'frozen source changed: {name}'


def selected(scores):
    return scores.filter(
        (pl.col('risk_full') >= THRESHOLD)
        & (pl.col('risk_v70') >= THRESHOLD)
        & (pl.col('family_full') == pl.col('family_v70'))
    ).sort(['table_id', 'selection_score', 'pair_id'], descending=[False, True, False]).group_by(
        'table_id', maintain_order=True
    ).head(2).sort('pair_id')


def score_frame(model, query, query70, features):
    assert query['pair_id'].equals(query70['pair_id'])
    full_prob = model.predict(matrix(query, features))
    short_prob = model.predict(matrix(query70, features))
    # Models are four-class teachers; class 0 is the non-positive family.
    assert full_prob.shape == short_prob.shape and full_prob.shape[1] == 4
    risk = 1.0 - full_prob[:, 0]
    risk70 = 1.0 - short_prob[:, 0]
    family = np.argmax(full_prob[:, 1:], axis=1) + 1
    family70 = np.argmax(short_prob[:, 1:], axis=1) + 1
    return query.select('pair_id', 'table_id').with_columns(
        pl.Series('risk_full', risk), pl.Series('risk_v70', risk70),
        pl.Series('family_full', family), pl.Series('family_v70', family70),
        pl.Series('selection_score', np.minimum(risk, risk70)),
    )


def freeze():
    """Freeze all mutable inputs before preparing or interpreting full scores."""
    assert not (OUT / 'plan.json').exists(), 'adapter already frozen'
    old_plan = load(PARENT / 'plan.json')
    old_frames = load(PARENT / 'frames-frozen.json')
    old_result = load(PARENT / 'yield-result.json')
    assert old_result['teacher_fits'] == 5
    assert old_frames['feature_count'] == 765
    assert old_plan['view_seeds']['70'] == 7307
    assert old_plan['filter']['full_positive_score_at_least'] == THRESHOLD
    assert old_plan['filter']['view70_positive_score_at_least'] == THRESHOLD
    assert old_plan['filter']['max_selected_per_pool'] == 2
    frozen = {
        'adapter_script': OUT / 'screen.py',
        'old_plan': PARENT / 'plan.json',
        'old_frames': PARENT / 'frames-frozen.json',
        'old_result': PARENT / 'yield-result.json',
        'old_selected_pairs': PARENT / 'selected-all-folds.parquet',
        'full_feature_plan': FULL_CACHE / 'feature-plan.json',
        'full_selected_pairs': FULL_CACHE / 'selected-pairs.parquet',
        'old_feature_plan': OLD_CACHE / 'feature-plan.json',
        'robust_pair_functions': ROOT / 'src/robust_pair_experiment.py',
        'tail_pair_functions': ROOT / 'src/tail_pair_experiment.py',
    }
    for fold in range(5):
        frozen[f'model_{fold}'] = PARENT / f'fold{fold}/pair-teacher.txt'
        frozen[f'old_scores_{fold}'] = PARENT / f'fold{fold}/unknown-scores.parquet'
        frozen[f'old_selected_{fold}'] = PARENT / f'fold{fold}/selected.parquet'
    assert all(path.exists() for path in frozen.values()), 'parent study is incomplete'
    old_predictions = []
    for fold in range(5):
        fit = load(PARENT / f'fold{fold}/fit.json')
        assert sha(PARENT / f'fold{fold}/pair-teacher.txt') == fit['model_sha256']
        assert sha(PARENT / f'fold{fold}/unknown-scores.parquet') == fit['scores_sha256']
        assert sha(PARENT / f'fold{fold}/selected.parquet') == fit['selected_sha256']
        old_predictions.append({
            'fold': fold, 'model_sha256': fit['model_sha256'],
            'scores_sha256': fit['scores_sha256'], 'selected_sha256': fit['selected_sha256'],
            'query_unknown_pairs': fit['query_unknown_pairs'],
            'selected_pairs': fit['selected_pairs'], 'selected_pools': fit['selected_pools'],
        })
    plan = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Scale only the unknown corpus while preserving the frozen five first-salt pair teachers.',
        'prohibitions': [
            'No pair-teacher fitting or parameter search.', 'No evidence or truth reads.',
            'No student fitting, quality evaluation, or submission.',
            'No threshold, family-agreement, or per-pool-cap change.',
        ],
        'teacher_source': str(PARENT),
        'full_cache': str(FULL_CACHE),
        'old_sample_cache': str(OLD_CACHE),
        'views': {'full': {'fraction': 1.0, 'seed': None}, 'v70': {'fraction': .7, 'seed': 7307}},
        'selection': {
            'positive_score_at_least_both_views': THRESHOLD,
            'same_winning_positive_family': True,
            'maximum_per_pool': 2,
            'tie_order': 'selection_score descending, pair_id ascending within table_id',
        },
        'gate': {'minimum_distinct_pairs': 15, 'minimum_distinct_pools': 10},
        'expected_full_cache': {'batches': 85, 'selected_pairs': 66521, 'known_pairs': 787,
                                'unknown_pairs': 65734, 'raw_feature_count': 163},
        'old_predictions': old_predictions,
        'frozen_sources': {name: {'path': str(path), 'sha256': sha(path)} for name, path in frozen.items()},
        'original_salt': old_plan['salt'],
        'excluded_old_outer_pools': old_plan['excluded_old_outer_pools'],  # reviewer fix: top-level in the parent plan
        'copied_from': {'path': str(PARENT.parent / 'semi_supervised_full_pair_yield_20260913/screen.py'), 'sha256': 'cbe6325331d8ece4fcdc5c93a14f13ccdc45ab09eeae1c4008913942e93aec14'},
        'old_frame_features_sha256': hashlib.sha256(
            json.dumps(old_frames['features'], separators=(',', ':')).encode('utf-8')
        ).hexdigest(),
        'replay_requirement': 'replay-old must report exact scores and selections for every old sample query before score is allowed',
    }
    write_once(OUT / 'plan.json', plan)
    print(json.dumps({'plan_frozen': True, 'teacher_fits': 0, 'model_reuse_count': 5}), flush=True)


def cache_receipts(plan):
    feature_plan = load(FULL_CACHE / 'feature-plan.json')
    expected = plan['expected_full_cache']
    assert len(feature_plan['batch_pools']) == expected['batches']
    assert feature_plan['selected_pairs'] == expected['selected_pairs']
    assert feature_plan['known_pairs'] == expected['known_pairs']
    assert feature_plan['unknown_pairs'] == expected['unknown_pairs']
    assert feature_plan['feature_count_expected'] == expected['raw_feature_count']
    receipts = []
    for index in range(expected['batches']):
        receipt_path = FULL_CACHE / f'batch_{index:03}.json'
        hand_path = FULL_CACHE / f'batch_{index:03}-hands.parquet'
        pair_path = FULL_CACHE / f'batch_{index:03}-pairs.parquet'
        assert receipt_path.exists() and hand_path.exists() and pair_path.exists(), f'missing batch {index:03}'
        receipt = load(receipt_path)
        assert receipt['hand_file'] == hand_path.name and receipt['pair_file'] == pair_path.name
        assert sha(hand_path) == receipt['hand_sha256']
        assert sha(pair_path) == receipt['pair_sha256']
        assert receipt['raw_features'] == expected['raw_feature_count']
        assert receipt['exact_raw_parity'] is True and receipt['prior_sample_raw_parity'] is True
        assert receipt['max_abs_difference'] == 0.0 and receipt['unequal_columns'] == []
        receipts.append({'batch': index, 'receipt_sha256': sha(receipt_path),
                         'hand_sha256': receipt['hand_sha256'], 'pair_sha256': receipt['pair_sha256'],
                         'known_pairs': receipt['known_pairs'], 'prior_sample_hand_rows': receipt['prior_sample_hand_rows']})
    return feature_plan, receipts


def prepare():
    """Create only deterministic full and 70%-history pair views; never score them."""
    plan = load(OUT / 'plan.json')
    assert_sources(plan)
    assert not (OUT / 'frames-frozen.json').exists(), 'frames already frozen'
    feature_plan, receipts = cache_receipts(plan)
    old_frames = load(PARENT / 'frames-frozen.json')
    features = old_frames['features']
    assert hashlib.sha256(json.dumps(features, separators=(',', ':')).encode('utf-8')).hexdigest() == plan['old_frame_features_sha256']
    meta = pl.read_parquet(FULL_CACHE / 'selected-pairs.parquet').sort('pair_id')
    assert meta.height == 66521 and meta['pair_id'].n_unique() == meta.height
    assert meta.filter(pl.col('label') >= 0).height == 787
    assert meta.filter(pl.col('label') < 0).height == 65734
    assert not set(meta['table_id']) & set(plan['excluded_old_outer_pools'])
    pieces = {name: [] for name in VIEWS}
    raw_columns = None
    for index in range(85):
        raw = pl.read_parquet(FULL_CACHE / f'batch_{index:03}-hands.parquet')
        names = [column for column in raw.columns if column not in ['pair_id', 'hand_id', 'table_id']]
        events = [column for column in names if column.startswith('v2_')]
        assert len(names) == 163 and len(events) == 56
        if raw_columns is None:
            raw_columns = names
        else:
            assert names == raw_columns, f'raw feature order changed in batch {index:03}'
        for name, (fraction, seed) in VIEWS.items():
            sampled = raw if fraction == 1.0 else thin(raw, fraction, seed)
            base = pl.read_parquet(FULL_CACHE / f'batch_{index:03}-pairs.parquet') if fraction == 1.0 else aggregate(sampled, names)
            frame = base.join(tails(sampled, events), on=['pair_id', 'table_id'], validate='1:1').join(
                meta.select('pair_id', 'player_1', 'player_2', 'label', 'behavior_family'),
                on='pair_id', validate='1:1'
            ).select(['pair_id', 'table_id', *features, 'player_1', 'player_2', 'label', 'behavior_family']).sort('pair_id')
            assert frame['pair_id'].n_unique() == frame.height
            pieces[name].append(frame)
        if index % 10 == 0:
            print(json.dumps({'prepared_batches': index + 1, 'total_batches': 85}), flush=True)
    outputs = {}
    for name, chunks in pieces.items():
        frame = pl.concat(chunks).sort('pair_id')
        assert frame.height == 66521 and frame['pair_id'].n_unique() == frame.height
        assert frame.select(features).columns == features
        destination = OUT / f'{name}.parquet'
        assert not destination.exists(), f'refusing to overwrite {destination}'
        frame.write_parquet(destination)
        outputs[name] = {'rows': frame.height, 'sha256': sha(destination)}
    # A physical hand may occur for more than one pair, but never in more than one original pool.
    raw_scan = pl.scan_parquet(str(FULL_CACHE / 'batch_*-hands.parquet')).select('hand_id', 'table_id').unique()
    overlap = raw_scan.group_by('hand_id').agg(pl.col('table_id').n_unique().alias('pools')).filter(pl.col('pools') > 1).collect()
    assert overlap.height == 0, 'physical hand crosses original pools'
    receipt = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'features': features,
        'feature_count': len(features), 'outputs': outputs, 'full_cache_receipts': receipts,
        'full_feature_plan_sha256': sha(FULL_CACHE / 'feature-plan.json'),
        'plan_sha256': sha(OUT / 'plan.json'), 'known_raw_parity_all_85': True,
        'prior_sample_raw_parity_all_85': True, 'physical_hand_cross_pool_overlap': 0,
    }
    write_once(OUT / 'frames-frozen.json', receipt)
    print(json.dumps({'frames_frozen': True, 'features': len(features), 'outputs': outputs}), flush=True)


def replay_old():
    """Exact preflight on the old sample cache; this gates all full-corpus scoring."""
    plan = load(OUT / 'plan.json')
    assert_sources(plan)
    assert not (OUT / 'old-replay.json').exists(), 'old replay already recorded'
    old_frames = load(PARENT / 'frames-frozen.json')
    features = old_frames['features']
    frames = {name: pl.read_parquet(PARENT / f'{name}.parquet').sort('pair_id') for name in VIEWS}
    mapping = plan['original_salt']['mapping']
    reports = []
    for fold in range(5):
        held = {pool for pool, value in mapping.items() if value == fold}
        train_pools = set(frames['full']['table_id']) - held
        query = frames['full'].filter(pl.col('table_id').is_in(sorted(train_pools)) & (pl.col('label') < 0)).sort('pair_id')
        query70 = frames['v70'].join(query.select('pair_id'), on='pair_id', how='semi').sort('pair_id')
        model_path = PARENT / f'fold{fold}/pair-teacher.txt'
        model = lgb.Booster(model_file=str(model_path))
        replay = score_frame(model, query, query70, features).sort('pair_id')
        reload_scores = score_frame(lgb.Booster(model_file=str(model_path)), query, query70, features).sort('pair_id')
        assert replay.equals(reload_scores), f'model reload changed old fold {fold}'
        saved = pl.read_parquet(PARENT / f'fold{fold}/unknown-scores.parquet').sort('pair_id')
        assert replay.equals(saved), f'old score replay mismatch in fold {fold}'
        rerun_selected = selected(replay)
        saved_selected = pl.read_parquet(PARENT / f'fold{fold}/selected.parquet').sort('pair_id')
        assert rerun_selected.equals(saved_selected), f'old selected output mismatch in fold {fold}'
        expected = plan['old_predictions'][fold]
        assert replay.height == expected['query_unknown_pairs']
        assert sha(PARENT / f'fold{fold}/unknown-scores.parquet') == expected['scores_sha256']
        assert sha(PARENT / f'fold{fold}/selected.parquet') == expected['selected_sha256']
        reports.append({'fold': fold, 'query_unknown_pairs': replay.height,
                        'selected_pairs': rerun_selected.height, 'selected_pools': rerun_selected['table_id'].n_unique(),
                        'exact_scores': True, 'exact_selection': True, 'exact_model_reload': True})
    write_once(OUT / 'old-replay.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
                                         'all_old_queries_exact': True, 'all_old_selections_exact': True,
                                         'folds': reports, 'plan_sha256': sha(OUT / 'plan.json')})
    print(json.dumps({'old_replay_passed': True, 'folds': 5}), flush=True)


def full_boundary_check(meta, held, train_pools):
    held_rows = meta.filter(pl.col('table_id').is_in(sorted(held)))
    query_rows = meta.filter(pl.col('table_id').is_in(sorted(train_pools)) & (pl.col('label') < 0))
    held_players = set(held_rows['player_1']) | set(held_rows['player_2'])
    query_players = set(query_rows['player_1']) | set(query_rows['player_2'])
    assert not held_players & query_players, 'held-pool player in full unknown query'
    assert not set(held_rows['pair_id']) & set(query_rows['pair_id'])
    return {'held_pools': len(held), 'query_pools': len(set(query_rows['table_id'])),
            'held_query_player_overlap': 0, 'physical_hand_exclusion_by_original_pool': True}


def score():
    """Score full unknown training pools with the frozen models after exact old replay."""
    plan = load(OUT / 'plan.json')
    assert_sources(plan)
    frames_receipt = load(OUT / 'frames-frozen.json')
    replay = load(OUT / 'old-replay.json')
    assert replay['plan_sha256'] == sha(OUT / 'plan.json')
    assert replay['all_old_queries_exact'] and replay['all_old_selections_exact']
    assert frames_receipt['plan_sha256'] == sha(OUT / 'plan.json')
    assert frames_receipt['known_raw_parity_all_85'] and frames_receipt['prior_sample_raw_parity_all_85']
    assert frames_receipt['physical_hand_cross_pool_overlap'] == 0
    assert not (OUT / 'yield-result.json').exists() and not (OUT / 'selected-all-folds.parquet').exists()
    for name, receipt in frames_receipt['outputs'].items():
        assert sha(OUT / f'{name}.parquet') == receipt['sha256']
    features = frames_receipt['features']
    frames = {name: pl.read_parquet(OUT / f'{name}.parquet').sort('pair_id') for name in VIEWS}
    meta = pl.read_parquet(FULL_CACHE / 'selected-pairs.parquet').sort('pair_id')
    mapping = plan['original_salt']['mapping']
    reports, all_selected = [], []
    all_pools = set(frames['full']['table_id'])
    assert len(all_pools) == 169 and len(mapping) == 104
    for fold in range(5):
        destination = OUT / f'fold{fold}'
        assert not destination.exists(), f'refusing to overwrite {destination}'
        held = {pool for pool, value in mapping.items() if value == fold}
        train_pools = all_pools - held
        boundary = full_boundary_check(meta, held, train_pools)
        query = frames['full'].filter(pl.col('table_id').is_in(sorted(train_pools)) & (pl.col('label') < 0)).sort('pair_id')
        query70 = frames['v70'].join(query.select('pair_id'), on='pair_id', how='semi').sort('pair_id')
        assert query['pair_id'].equals(query70['pair_id'])
        model_path = PARENT / f'fold{fold}/pair-teacher.txt'
        assert sha(model_path) == plan['old_predictions'][fold]['model_sha256']
        prediction = score_frame(lgb.Booster(model_file=str(model_path)), query, query70, features).sort('pair_id')
        reloaded = score_frame(lgb.Booster(model_file=str(model_path)), query, query70, features).sort('pair_id')
        assert prediction.equals(reloaded), f'model reload changed full fold {fold}'
        chosen = selected(prediction)
        destination.mkdir()
        prediction.write_parquet(destination / 'unknown-scores.parquet')
        chosen.write_parquet(destination / 'selected.parquet')
        report = {'fold': fold, 'query_unknown_pairs': prediction.height,
                  'selected_pairs': chosen.height, 'selected_pools': chosen['table_id'].n_unique(),
                  'selected_family_counts': chosen.group_by('family_full').len().to_dicts(),
                  'reused_model_sha256': sha(model_path), 'exact_saved_model_reload': True,
                  'scores_sha256': sha(destination / 'unknown-scores.parquet'),
                  'selected_sha256': sha(destination / 'selected.parquet'), **boundary,
                  'no_evidence_labels_read': True, 'no_quality_metrics': True, 'no_teacher_fit': True}
        write_once(destination / 'score.json', report)
        reports.append(report)
        all_selected.append(chosen.with_columns(pl.lit(fold).alias('fold')))
        print(json.dumps(report), flush=True)
    appearances = pl.concat(all_selected)
    distinct = appearances.unique('pair_id')
    passed = distinct.height >= 15 and distinct['table_id'].n_unique() >= 10
    appearances.write_parquet(OUT / 'selected-all-folds.parquet')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'YIELD_PASS' if passed else 'INSUFFICIENT_YIELD_STOP',
              'teacher_fits': 0, 'reused_saved_pair_teachers': 5,
              'distinct_selected_pairs': distinct.height, 'distinct_selected_pools': distinct['table_id'].n_unique(),
              'selected_appearances': appearances.height, 'passes_yield_gate': passed, 'folds': reports,
              'next_action': ('Pass selected-full pairs, with fold identity, to a separately frozen reuse of the existing '
                              'evidence teachers; do not refit them.' if passed else
                              'Stop before evidence-teacher or student work; preserve this insufficient-yield result.')}
    write_once(OUT / 'yield-result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'prepare', 'replay-old', 'score'])
    args = parser.parse_args()
    {'freeze': freeze, 'prepare': prepare, 'replay-old': replay_old, 'score': score}[args.mode]()
