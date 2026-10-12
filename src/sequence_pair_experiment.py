"""One fixed, private screen of sequence aggregates for pair-risk prediction."""
from __future__ import annotations

import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from robust_pair_experiment import (ROOT, DATA, SOURCE, KEYS, LABELS, FAMILIES,
    TRAIN_VIEWS, HELDOUT_VIEW, MODEL_PARAMS, thin, aggregate, matrix, metrics,
    assert_disjoint, sha, peak_memory_bytes)
from tail_pair_experiment import tails
from sequence_evidence_fast import extract
from sequence_evidence_features import NAMES, extract_pair_hand

REPORT = ROOT / 'reports' / 'sequence-pair-experiment'
FEATURES = ['seq_' + name for name in NAMES]


def sequence_aggregate(frame, columns=FEATURES):
    return frame.group_by('pair_id', 'table_id').agg(
        *[pl.col(c).mean().alias(c + '_mean') for c in columns],
        *[pl.col(c).top_k(3).mean().alias(c + '_top3_mean') for c in columns],
        *[(pl.col(c) > 0).mean().alias(c + '_nonzero_fraction') for c in columns])


def aggregation_check():
    raw = pl.DataFrame({'pair_id': ['a'] * 5 + ['b'],
        'table_id': ['t1'] * 5 + ['t2'], 'seq_example': [0., 1., 9., 2., 0., 4.]})
    actual = sequence_aggregate(raw, ['seq_example']).sort('pair_id')
    np.testing.assert_allclose(actual['seq_example_mean'], [2.4, 4.])
    np.testing.assert_allclose(actual['seq_example_top3_mean'], [4., 4.])
    np.testing.assert_allclose(actual['seq_example_nonzero_fraction'], [.6, 1.])
    assert actual.equals(sequence_aggregate(raw.reverse(), ['seq_example']).sort('pair_id'))


def source_frames(keys):
    ids = keys.select('hand_id').unique().lazy()
    hands = (pl.scan_parquet(DATA / 'hands.parquet')
        .filter(pl.col('phase') == 'development').join(ids, on='hand_id')
        .select('hand_id', 'big_blind').collect())
    seats = (pl.scan_parquet(DATA / 'seats.parquet').join(ids, on='hand_id')
        .select('hand_id', 'player_id', 'total_contribution', 'net_chips').collect())
    actions = (pl.scan_parquet(DATA / 'actions.parquet').join(ids, on='hand_id')
        .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount',
                'to_call', 'pot_before').collect())
    assert hands.height == keys['hand_id'].n_unique()
    return hands, seats, actions


def scalar_check(keys, features):
    hands, seats, actions = source_frames(keys)
    blind = dict(hands.iter_rows())
    seat_lookup, action_lookup = {}, {}
    for h, p, contribution, net in seats.iter_rows():
        seat_lookup.setdefault(h, {})[p] = (contribution, net)
    for row in actions.iter_rows(named=True):
        h = row.pop('hand_id')
        action_lookup.setdefault(h, []).append(row)
    keys = keys.sort('pair_id', 'hand_id')
    expected = np.asarray([extract_pair_hand(row['player_1'], row['player_2'],
        blind[row['hand_id']], seat_lookup[row['hand_id']],
        action_lookup.get(row['hand_id'], [])) for row in keys.iter_rows(named=True)],
        dtype=np.float32)
    actual = (keys.select('pair_id', 'hand_id').join(features,
        on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
        .select(FEATURES).to_numpy())
    np.testing.assert_array_equal(actual, expected)
    return {'pair_hands': keys.height, 'cells': int(expected.size),
            'scalar_vector_exact': True, 'sample_seed': 202609091,
            'sample_scope': 'confirmed-negative training rows only'}


def evaluate(frame, risk, omitted=None):
    y = np.array([LABELS[name] for name in frame['behavior_family']])
    result = metrics(y, risk)
    if omitted is None:
        result['families'] = {}
        for family in FAMILIES[1:]:
            selected = (y == 0) | (y == LABELS[family])
            result['families'][family] = metrics(y[selected], risk[selected])
    return result


def known_gate(base, new):
    gains = {history: {
        'pair_ap': new[history]['pair_ap'] - base[history]['pair_ap'],
        'hypothetical_1pct_ap': new[history]['pair_ap_at_hypothetical_1pct_prevalence']
            - base[history]['pair_ap_at_hypothetical_1pct_prevalence'],
        'family_ap': {f: new[history]['families'][f]['pair_ap']
            - base[history]['families'][f]['pair_ap'] for f in FAMILIES[1:]}
        } for history in ['full', 'shortened']}
    checks = {
        'full_ap_min_minus_001': gains['full']['pair_ap'] >= -.001,
        'short_ap_min_002': gains['shortened']['pair_ap'] >= .002,
        'sparse_prior_nonnegative': all(g['hypothetical_1pct_ap'] >= 0 for g in gains.values()),
        'family_regression_max_015': all(v >= -.015 for g in gains.values() for v in g['family_ap'].values())}
    return {'gains': gains, 'checks': checks, 'passed': all(checks.values())}


def run():
    REPORT.mkdir(exist_ok=True)
    if (REPORT / 'start.json').exists():
        raise FileExistsError('Preserve the existing fixed run; do not refit or overwrite')
    started = time.perf_counter()
    aggregation_check()
    protected_paths = [ROOT / 'outputs_v5/submission_event_tails.csv',
        ROOT / 'outputs_v5/pair_model_event_tails.joblib',
        ROOT / 'outputs_v6/submission_sequence_evidence.csv',
        ROOT / 'outputs_v6/sequence_ranker.joblib', ROOT / 'outputs_v6/ready.json']
    helper_paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ['robust_pair_experiment.py', 'tail_pair_experiment.py',
         'sequence_evidence_fast.py', 'sequence_evidence_features.py']]]
    protected = {str(p.relative_to(ROOT)): sha(p) for p in protected_paths + helper_paths}
    large_sources = {str(p.relative_to(ROOT)): {'size': p.stat().st_size,
        'mtime_ns': p.stat().st_mtime_ns} for p in [DATA / f'{name}.parquet'
        for name in ['hands', 'seats', 'actions']]}
    raw_paths = sorted((SOURCE / 'hands_development').glob('batch_*.parquet'))
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [DATA / 'development_labels.csv',
        SOURCE / 'pairs_development.parquet', *raw_paths,
        ROOT / 'reports/sequence-evidence-experiment/development.parquet']}
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(REPORT / 'PLAN.md'), 'protected_sha256': protected,
        'input_sha256': inputs, 'raw_source_identity': large_sources,
        'raw_source_identity_kind': 'size and nanosecond mtime, not content hash',
        'model_params': MODEL_PARAMS, 'split_seed': 2901, 'fold': 0,
        'training_views': [[1, None], *TRAIN_VIEWS], 'heldout_view': HELDOUT_VIEW,
        'fit_budget': '2 known, then 6 omitted only if known gate passes'}
    (REPORT / 'start.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    labels = pl.read_csv(DATA / 'development_labels.csv')
    assert labels['pair_id'].n_unique() == labels.height
    assert not labels.filter(~pl.col('label_status').is_in(
        ['confirmed_target', 'confirmed_non_target'])).height
    raw = pl.concat([pl.read_parquet(p) for p in raw_paths])
    assert raw.select(pl.struct('pair_id', 'hand_id').n_unique()).item() == raw.height
    full = (pl.read_parquet(SOURCE / 'pairs_development.parquet')
        .join(labels, on='pair_id', validate='1:1').sort('pair_id'))
    assert full.height == labels.height == 1860
    counts = raw.group_by('pair_id', 'table_id').len().join(
        full.select('pair_id', 'table_id', 'shared_hands'), on=['pair_id', 'table_id'], validate='1:1')
    assert counts.height == full.height and not counts.filter(pl.col('len') != pl.col('shared_hands')).height
    assert counts['len'].sum() == raw.height
    raw_names = [c for c in raw.columns if c not in KEYS]
    events = [c for c in raw_names if c.startswith('v2_')]
    bundle = joblib.load(ROOT / 'outputs_v5/pair_model_event_tails.joblib')
    control_features = bundle['features']
    assert len(control_features) == 765 and len(events) == 56
    y = np.array([LABELS[name] for name in full['behavior_family']])
    assert np.array_equal(y > 0, full['label'].to_numpy() == 1)
    train_ids, held_ids = next(GroupShuffleSplit(n_splits=3, test_size=.25,
        random_state=2901).split(full, y, full['table_id']))
    train_pools = full[train_ids.tolist()]['table_id'].unique().sort().to_list()
    held_pools = full[held_ids.tolist()]['table_id'].unique().sort().to_list()
    (REPORT / 'split.json').write_text(json.dumps({'train_pools': train_pools,
        'held_pools': held_pools}, indent=2) + '\n', encoding='utf-8')
    keys = raw.select(KEYS).join(labels.select('pair_id', 'player_1', 'player_2',
        'label_status'), on='pair_id', validate='m:1')
    pools = keys['table_id'].unique().sort().to_list()
    feature_parts = []
    for i in range(0, len(pools), 20):
        batch = keys.filter(pl.col('table_id').is_in(pools[i:i + 20]))
        hands, seats, actions = source_frames(batch)
        part = extract(batch, hands, seats, actions)
        assert part.height == batch.height and np.isfinite(part.select(FEATURES).to_numpy()).all()
        feature_parts.append(part)
        del batch, hands, seats, actions
        gc.collect()
        print(json.dumps({'sequence_pools': min(i + 20, len(pools)), 'total_pools': len(pools),
            'pair_hands_built': sum(p.height for p in feature_parts),
            'seconds': round(time.perf_counter() - started, 2)}), flush=True)
    features = pl.concat(feature_parts).sort('pair_id', 'hand_id')
    assert features.height == raw.height == 223749
    features.write_parquet(REPORT / 'development-sequences.parquet')
    negative_keys = keys.filter((pl.col('label_status') == 'confirmed_non_target')
        & pl.col('table_id').is_in(train_pools)).sort('pair_id', 'hand_id')
    sample_idx = np.random.default_rng(202609091).choice(negative_keys.height, 50, replace=False)
    scalar_result = scalar_check(negative_keys[sample_idx.tolist()], features)
    old = pl.read_parquet(ROOT / 'reports/sequence-evidence-experiment/development.parquet').sort('pair_id', 'hand_id')
    new_targets = features.join(old.select('pair_id', 'hand_id'), on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
    assert old.height == new_targets.height == 45129
    np.testing.assert_array_equal(old.select(FEATURES).to_numpy(), new_targets.select(FEATURES).to_numpy())
    raw = raw.join(features, on=['pair_id', 'hand_id'], validate='1:1')
    extra = sequence_aggregate(raw)
    extras = [c for c in extra.columns if c not in KEYS]
    candidate_features = control_features + extras
    assert len(candidate_features) == len(set(candidate_features)) == 834
    full = full.join(tails(raw, events), on=['pair_id', 'table_id'], validate='1:1').join(
        extra, on=['pair_id', 'table_id'], validate='1:1').sort('pair_id')
    views = {}
    for fraction, seed in TRAIN_VIEWS + [HELDOUT_VIEW]:
        sample = thin(raw, fraction, seed)
        view = aggregate(sample, raw_names).join(tails(sample, events), on=['pair_id', 'table_id'], validate='1:1')
        view = view.join(sequence_aggregate(sample), on=['pair_id', 'table_id'], validate='1:1')
        views[fraction, seed] = view.join(labels, on='pair_id', validate='1:1').select(full.columns).sort('pair_id')
    train = full.filter(pl.col('table_id').is_in(train_pools))
    held = full.filter(pl.col('table_id').is_in(held_pools))
    augmented = pl.concat([train] + [views[key].filter(pl.col('table_id').is_in(train_pools)) for key in TRAIN_VIEWS])
    assert not augmented.group_by('pair_id').len().filter(pl.col('len') != 3).height
    assert_disjoint(augmented, held)
    validation = {'full': held, 'shortened': views[HELDOUT_VIEW].filter(pl.col('table_id').is_in(held_pools))}
    assert all(np.isfinite(matrix(f, candidate_features)).all() for f in [full, *views.values()])
    results, predictions = [], []
    gate = None
    for omitted in [None, *FAMILIES[1:]]:
        if omitted is not None and not gate['passed']:
            break
        tag = 'known' if omitted is None else 'omit_' + omitted
        fitting = augmented if omitted is None else augmented.filter(pl.col('behavior_family') != omitted)
        target = np.array([LABELS[f] for f in fitting['behavior_family']])
        for name, columns in [('v5_control', control_features), ('sequence_aggregates', candidate_features)]:
            model = lgb.LGBMClassifier(**MODEL_PARAMS)
            model.fit(matrix(fitting, columns), target, sample_weight=np.full(fitting.height, 1 / 3))
            result = {'experiment': tag, 'candidate': name, 'features': len(columns),
                'train_pairs': fitting['pair_id'].n_unique(), 'train_rows': fitting.height}
            for history, frame in validation.items():
                if omitted is not None:
                    frame = frame.filter(pl.col('behavior_family').is_in(['none', omitted]))
                frame = frame.sort('pair_id')
                probabilities = model.predict_proba(matrix(frame, columns))
                zero = int(np.flatnonzero(model.classes_ == 0)[0])
                risk = 1 - probabilities[:, zero]
                result[history] = evaluate(frame, risk, omitted)
                predictions.append(frame.select('pair_id', 'table_id', 'behavior_family', 'shared_hands')
                    .with_columns(pl.lit(tag).alias('experiment'), pl.lit(name).alias('candidate'),
                        pl.lit(history).alias('history'), pl.Series('risk', risk)))
            results.append(result)
            (REPORT / f'{tag}-{name}.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(result), flush=True)
        if omitted is None:
            gate = known_gate(results[0], results[1])
            print(json.dumps({'known_gate': gate}), flush=True)
    omitted_gate = None
    if gate['passed']:
        lookup = {(r['experiment'], r['candidate']): r for r in results}
        omitted_gains = {h: {f: lookup['omit_' + f, 'sequence_aggregates'][h]['pair_ap']
            - lookup['omit_' + f, 'v5_control'][h]['pair_ap'] for f in FAMILIES[1:]}
            for h in validation}
        omitted_gate = {'gains': omitted_gains, 'passed': all(
            np.mean(list(g.values())) >= .02 and min(g.values()) >= -.03 for g in omitted_gains.values())}
    assert protected == {str(p.relative_to(ROOT)): sha(p) for p in protected_paths + helper_paths}
    assert large_sources == {str(p.relative_to(ROOT)): {'size': p.stat().st_size,
        'mtime_ns': p.stat().st_mtime_ns} for p in [DATA / f'{name}.parquet'
        for name in ['hands', 'seats', 'actions']]}
    pl.concat(predictions).write_csv(REPORT / 'predictions.csv')
    payload = {'plan': plan, 'scalar_check': scalar_result, 'aggregation_check': True,
        'target_cache_exact_cells': old.height * len(FEATURES),
        'labeled_pairs': full.height, 'labeled_pair_hands': raw.height,
        'train_pools': len(train_pools), 'held_pools': len(held_pools),
        'held_pairs': held.height, 'pair_pool_player_overlap': False,
        'candidate_features': candidate_features, 'results': results,
        'known_gate': gate, 'omitted_gate': omitted_gate,
        'screen_passed': bool(gate['passed'] and omitted_gate and omitted_gate['passed']),
        'protected_unchanged': True, 'seconds': time.perf_counter() - started,
        'peak_memory_bytes': peak_memory_bytes(),
        'limitations': ['Exploratory reused split; no independent confirmation or new holdout.',
            'One-percent prevalence is a sensitivity assumption, not an estimated test prior.',
            'Old player baseline features remain fixed during hand thinning.',
            'Retained positive labels may outlive all suspicious hands under thinning.',
            'Only pair-risk metrics are reported; these are not combined competition scores.',
            'No evaluation prediction, final model or submission was produced.']}
    (REPORT / 'screen.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'screen_passed': payload['screen_passed'], 'fits': len(results),
        'seconds': payload['seconds'], 'peak_memory_bytes': payload['peak_memory_bytes']}), flush=True)


if __name__ == '__main__':
    run()
