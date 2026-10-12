"""Step 20: full-stack pair features for EVERY player pair of a held fold's pools (PU audit, pair side).

Why: the public pair AP cannot be measured locally (1,488 labelled negatives, local AP .9995) while the
evaluation set has 112,540 pairs at ~0.3-0.5% prevalence. This builds the exact production feature stack
(features.py base -> advanced_features.py v2 events -> conditional-action policy residuals) for all
C(30,2)=435 pairs of each table in a held fold, so the frozen fold model (trained without those pools) can
be scored at realistic prevalence. Unlabelled pairs are UNKNOWN, not negative (competition glossary), so the
resulting AP is a lower bound and high-scoring unlabelled pairs must be inspected, not assumed wrong.

Reproduction guard: the 1,860 labelled pairs that fall in the processed pools must reproduce Codex's saved
fold predictions (reports/conditional-action-confirmation/results/fold{F}/pair-predictions.csv) exactly.

Outputs (Claude Version/experiments/pu_pairs_20260911/fold{F}/): batch_{i:03}-pairs.parquet (pair-level:
shared_hands + base/v2 aggregates + tails + conditional aggregates), batch_{i:03}.json receipt, pairs.csv
(pair_id, player_1, player_2, table_id, label if known).
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys
import gc
import json
import time
import hashlib
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
sys.path.insert(0, str(ORIGINAL / 'src'))
import numpy as np
import polars as pl
import joblib
from advanced_features import strength_table, directional_events  # noqa: E402  (ORIGINAL/src, read-only)
from conditional_inference_worker import extract_chunk  # noqa: E402
from run_conditional_policy_screen import hand_features  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402
from robust_pair_experiment import aggregate  # noqa: E402
from evaluate_conditional_action_screen import extra_aggregate  # noqa: E402
from features import RANKS  # noqa: E402

CONF = ORIGINAL / 'reports' / 'conditional-action-confirmation'
STYLE = ORIGINAL / 'reports' / 'conditional-action-experiment' / 'public-action-style-counts.parquet'
EXP = ORIGINAL / 'Claude Version' / 'experiments' / 'pu_pairs_20260911'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_policy(fold):
    return CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'held-models.joblib'


def fold_pools(fold):
    return sorted(pl.read_csv(ORIGINAL / 'reports' / 'tabicl-confirmation' / f'fold{fold}' / 'held-pairs.csv')['table_id'].unique())


def all_pairs(pools, hands):
    """All unordered player pairs per table (players are unique to one table), labelled where known."""
    seats = pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet').select('hand_id', 'player_id').join(hands.lazy().select('hand_id', 'table_id'), on='hand_id')
    players = seats.select('table_id', 'player_id').unique().collect()
    a = players.rename({'player_id': 'player_1'})
    b = players.rename({'player_id': 'player_2'})
    pairs = a.join(b, on='table_id').filter(pl.col('player_1') < pl.col('player_2'))
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'player_1', 'player_2', 'label', 'behavior_family')
    pairs = pairs.join(labels, on=['player_1', 'player_2'], how='left')
    synthetic = pl.concat_str([pl.lit('X'), pl.concat_str(['player_1', 'player_2']).map_elements(lambda s: hashlib.sha1(s.encode()).hexdigest()[:12].upper(), return_dtype=pl.String)])
    pairs = pairs.with_columns(pl.coalesce([pl.col('pair_id'), synthetic]).alias('pair_id'), pl.col('label').fill_null(-1), pl.col('behavior_family').fill_null('unknown'))
    assert pairs['pair_id'].n_unique() == pairs.height
    return pairs.select('pair_id', 'player_1', 'player_2', 'table_id', 'label', 'behavior_family').sort('pair_id')


def base_features(h, p):
    """Verbatim port of the per-batch body of ORIGINAL/src/features.py (pairs frame injected)."""
    s = pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet').join(
        h.lazy().select('hand_id', 'big_blind', 'final_pot', 'players_at_showdown'), on='hand_id', how='inner').collect(engine='streaming')
    a = pl.scan_parquet(ORIGINAL / 'data' / 'actions.parquet').join(
        h.lazy().select('hand_id', 'big_blind'), on='hand_id', how='inner').collect(engine='streaming').sort('hand_id', 'action_no')
    a = a.with_columns(
        (pl.col('action').is_in(['bet', 'raise', 'all_in']) & (pl.col('amount') > pl.col('to_call'))).alias('aggressive'),
        (pl.col('amount') / pl.col('big_blind')).alias('amount_bb'),
        (pl.col('to_call') / (pl.col('pot_before') + pl.col('to_call')).clip(lower_bound=1)).alias('pot_odds'),
    ).with_columns(
        pl.when('aggressive').then(pl.col('player_id')).otherwise(None).forward_fill().shift(1).over('hand_id', 'street').alias('previous_aggressor'))
    action_stats = a.group_by('hand_id', 'player_id').agg(
        pl.len().alias('n_actions'),
        pl.col('aggressive').sum().alias('n_aggressive'),
        (pl.col('action') == 'check').sum().alias('n_check'),
        (pl.col('action') == 'call').sum().alias('n_call'),
        ((pl.col('action') == 'check') & (pl.col('street') != 'preflop')).sum().alias('late_checks'),
        pl.when('aggressive').then('amount_bb').otherwise(0).sum().alias('agg_bb'),
        pl.when(pl.col('action').is_in(['call', 'all_in']) & ~pl.col('aggressive')).then('amount_bb').otherwise(0).sum().alias('call_bb'),
        pl.when(pl.col('action') == 'fold').then('pot_odds').otherwise(0).max().alias('fold_pot_odds'),
        pl.col('players_active').min().alias('min_active'),
    )
    interactions = a.filter(pl.col('previous_aggressor').is_not_null()).group_by('hand_id', 'player_id', 'previous_aggressor').agg(
        (pl.col('action') == 'fold').sum().alias('opp_fold'),
        pl.col('aggressive').sum().alias('opp_raise'),
        (~pl.col('aggressive') & pl.col('action').is_in(['call', 'all_in'])).sum().alias('opp_call'),
        pl.when(~pl.col('aggressive')).then('amount_bb').otherwise(0).sum().alias('opp_call_bb'),
        pl.when(pl.col('action') == 'fold').then('pot_odds').otherwise(0).max().alias('opp_fold_odds'),
    )
    s = s.join(action_stats, on=['hand_id', 'player_id'], how='left').with_columns(
        (pl.col('total_contribution') / pl.col('big_blind')).alias('contribution_bb'),
        (pl.col('net_chips') / pl.col('big_blind')).alias('net_bb'),
        (pl.col('starting_stack') / pl.col('big_blind')).alias('stack_bb'),
        pl.col('hole_card_1').str.slice(0, 1).replace_strict(RANKS).cast(pl.Float64).alias('rank1'),
        pl.col('hole_card_2').str.slice(0, 1).replace_strict(RANKS).cast(pl.Float64).alias('rank2'),
        (pl.col('hole_card_1').str.slice(1) == pl.col('hole_card_2').str.slice(1)).cast(pl.Float64).alias('suited'),
    ).with_columns(
        ((pl.max_horizontal('rank1', 'rank2') + pl.min_horizontal('rank1', 'rank2') * .5) / 21 +
         (pl.col('rank1') == pl.col('rank2')).cast(pl.Float64) * .7 + pl.col('suited') * .1).alias('hole_strength'))
    player_base = s.group_by('player_id').agg(
        pl.col('n_aggressive').mean().alias('baseline_aggressive'),
        pl.col('n_check').mean().alias('baseline_check'),
        pl.col('contribution_bb').mean().alias('baseline_contribution'),
        pl.col('net_bb').mean().alias('baseline_net'),
        pl.col('folded').mean().alias('baseline_fold'),
    )
    s = s.join(player_base, on='player_id').with_columns(
        (pl.col('n_aggressive') - pl.col('baseline_aggressive')).alias('aggression_residual'),
        (pl.col('n_check') - pl.col('baseline_check')).alias('check_residual'),
        (pl.col('contribution_bb') - pl.col('baseline_contribution')).alias('contribution_residual'),
        (pl.col('net_bb') - pl.col('baseline_net')).alias('net_residual'),
    )
    columns = ['contribution_bb', 'net_bb', 'stack_bb', 'folded', 'went_to_showdown',
               'won_share', 'n_actions', 'n_aggressive', 'n_check', 'n_call', 'late_checks',
               'agg_bb', 'call_bb', 'fold_pot_odds', 'min_active', 'hole_strength',
               'aggression_residual', 'check_residual', 'contribution_residual', 'net_residual']
    s1 = s.select('hand_id', 'player_id', *columns).rename({'player_id': 'player_1', **{c: c + '_1' for c in columns}})
    s2 = s.select('hand_id', 'player_id', *columns).rename({'player_id': 'player_2', **{c: c + '_2' for c in columns}})
    joined = p.join(s1, on='player_1').join(s2, on=['player_2', 'hand_id']).join(
        h.select('hand_id', 'big_blind', 'final_pot', 'players_at_showdown'), on='hand_id')
    oppcols = ['opp_fold', 'opp_raise', 'opp_call', 'opp_call_bb', 'opp_fold_odds']
    for side, other in [(1, 2), (2, 1)]:
        inter = interactions.rename({'player_id': f'player_{side}', 'previous_aggressor': f'player_{other}', **{c: c + f'_{side}' for c in oppcols}})
        joined = joined.join(inter, on=['hand_id', 'player_1', 'player_2'], how='left')
    joined = joined.with_columns(pl.col(oppcols[0] + '_1').fill_null(0))
    symmetric = []
    for c in columns + oppcols:
        c1, c2 = pl.col(c + '_1').cast(pl.Float64).fill_null(0), pl.col(c + '_2').cast(pl.Float64).fill_null(0)
        symmetric += [(c1 + c2).alias(c + '_sum'), (c1 - c2).abs().alias(c + '_gap'), pl.min_horizontal(c1, c2).alias(c + '_min'), pl.max_horizontal(c1, c2).alias(c + '_max')]
    joined = joined.with_columns(symmetric).with_columns(
        (pl.col('final_pot') / pl.col('big_blind')).alias('pot_bb'),
        ((pl.col('net_bb_1') > 0) & (pl.col('net_bb_2') < 0) | (pl.col('net_bb_2') > 0) & (pl.col('net_bb_1') < 0)).cast(pl.Float64).alias('opposite_outcomes'),
        (pl.col('folded_1') & pl.col('folded_2')).cast(pl.Float64).alias('both_fold'),
        ((pl.col('won_share_1') + pl.col('won_share_2'))).alias('joint_win_share'),
        (pl.col('hole_strength_1') * pl.col('opp_fold_1').fill_null(0) + pl.col('hole_strength_2') * pl.col('opp_fold_2').fill_null(0)).alias('strong_fold_to_partner'),
        (pl.col('hole_strength_1') * pl.col('late_checks_1') + pl.col('hole_strength_2') * pl.col('late_checks_2')).alias('strong_late_checks'),
    )
    names = [e.meta.output_name() for e in symmetric]
    extra = ['pot_bb', 'players_at_showdown', 'opposite_outcomes', 'both_fold', 'joint_win_share', 'strong_fold_to_partner', 'strong_late_checks']
    result = joined.select('pair_id', 'hand_id', 'table_id', *names, *extra)
    return result.with_columns(pl.exclude('pair_id', 'hand_id', 'table_id').cast(pl.Float32).fill_null(0))


def v2_features(raw, pairs):
    """Verbatim port of the per-batch body of ORIGINAL/src/advanced_features.py build()."""
    keys = raw.select('pair_id', 'hand_id').join(pairs.select('pair_id', 'player_1', 'player_2'), on='pair_id')
    ids = keys.select('hand_id').unique()
    h = pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet').join(ids.lazy(), on='hand_id').collect()
    s = pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet').join(ids.lazy(), on='hand_id').collect()
    a = pl.scan_parquet(ORIGINAL / 'data' / 'actions.parquet').join(ids.lazy(), on='hand_id').collect().sort('hand_id', 'action_no')
    a = a.join(h.select('hand_id', 'big_blind'), on='hand_id').with_columns(
        (pl.col('action').is_in(['bet', 'raise', 'all_in']) & (pl.col('amount') > pl.col('to_call'))).alias('aggressive'))
    a = a.with_columns(pl.when('aggressive').then(pl.col('player_id')).otherwise(None).forward_fill().shift(1).over('hand_id', 'street').alias('previous_aggressor'))
    folds = a.filter(pl.col('action') == 'fold').group_by('hand_id', 'player_id').agg(pl.col('action_no').min().alias('fold_no'))
    si = strength_table(s, h).join(s.select('hand_id', 'player_id', 'net_chips'), on=['hand_id', 'player_id'])
    si = si.join(folds, on=['hand_id', 'player_id'], how='left').with_columns(pl.col('fold_no').fill_null(1000000))
    first = directional_events(keys, a, si, 'player_1', 'player_2')
    second = directional_events(keys, a, si, 'player_2', 'player_1')
    cols = [c for c in first.columns if c.startswith('v2_')]
    joined = first.join(second, on=['pair_id', 'hand_id'], how='full', coalesce=True, suffix='_other').fill_null(0)
    added = joined.select('pair_id', 'hand_id', *[(pl.col(c) + pl.col(c + '_other')).alias(c + '_sum') for c in cols], *[pl.max_horizontal(c, c + '_other').alias(c + '_max') for c in cols])
    result = raw.join(added, on=['pair_id', 'hand_id'], how='left')
    newcols = [c for c in result.columns if c.startswith('v2_')]
    result = result.with_columns(pl.col(newcols).fill_null(0).cast(pl.Float32))
    assert result.height == raw.height
    return result


def conditional_features(keys, hands, policy, counts, caf):
    """Port of the per-pool body of ORIGINAL/src/conditional_inference_worker.py main()."""
    from conditional_action_features import card_features
    ids = keys.select('hand_id').unique()
    h = hands.join(ids, on='hand_id', how='semi')
    s = pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet').join(ids.lazy(), on='hand_id', how='semi').collect()
    a = pl.scan_parquet(ORIGINAL / 'data' / 'actions.parquet').join(ids.lazy(), on='hand_id', how='semi').collect()
    frame = extract_chunk(keys, h, s, a, counts)
    X = frame.select(policy['inputs']).to_numpy().astype(np.float32)
    ca = hand_features(frame, policy['classifier'].predict_proba(X, num_threads=1), policy['regressor'].predict(X, num_threads=1))
    got = [c for c in ca.columns if c.startswith('ca_')]
    assert got == caf, (got[:3], caf[:3])
    missing = keys.select('pair_id', 'hand_id').join(ca.select('pair_id', 'hand_id'), on=['pair_id', 'hand_id'], how='anti').height
    ca = keys.select('pair_id', 'hand_id', 'table_id').join(ca.drop('table_id'), on=['pair_id', 'hand_id'], how='left', validate='1:1').with_columns(pl.col(caf).fill_null(0))
    assert ca.height == keys.height
    card_features.cache_clear()
    return ca, frame.height, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--pools-per-batch', type=int, default=2)
    ap.add_argument('--batches', type=int, nargs='*', default=None, help='batch indices to build (default all)')
    args = ap.parse_args()
    out = EXP / f'fold{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    pools = fold_pools(args.fold)
    hands_all = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').filter((pl.col('phase') == 'development') & pl.col('table_id').is_in(pools))
    pairs = all_pairs(pools, hands_all)
    pairs.write_csv(out / 'pairs.csv')
    policy = joblib.load(fold_policy(args.fold))
    caf = [c for c in pl.read_parquet(CONF / f'conditional-fold{args.fold}-v1' / 'policy-screen' / 'hand-features.parquet', n_rows=1).columns if c.startswith('ca_')]
    counts = {r[0]: np.array(r[1:], dtype=np.int64) for r in pl.read_parquet(STYLE).iter_rows()}
    identity = {'fold': args.fold, 'pools': pools, 'pools_per_batch': args.pools_per_batch, 'policy_sha256': sha(fold_policy(args.fold)), 'style_counts_sha256': sha(STYLE),
                'code_sha256': sha(ORIGINAL / 'Claude Version' / 'src' / 'pu_pairs' / 'build_allpairs_fold.py'), 'pairs': pairs.height, 'labelled_pairs': int((pairs['label'] >= 0).sum())}
    write_json(out / 'identity.json', identity)
    batches = list(range(0, len(pools), args.pools_per_batch))
    for i, start in enumerate(batches):
        if args.batches is not None and i not in args.batches:
            continue
        dest = out / f'batch_{i:03}-pairs.parquet'
        receipt = out / f'batch_{i:03}.json'
        if receipt.exists() and dest.exists():
            print(json.dumps({'cached': i}), flush=True)
            continue
        t0 = time.perf_counter()
        these = pools[start:start + args.pools_per_batch]
        h = hands_all.filter(pl.col('table_id').is_in(these))
        p = pairs.filter(pl.col('table_id').is_in(these)).select('pair_id', 'player_1', 'player_2', 'table_id')
        raw = base_features(h, p)
        p = p.join(raw.select('pair_id').unique(), on='pair_id', how='semi')  # pairs never dealt together have no rows
        t1 = time.perf_counter()
        raw = v2_features(raw, p)
        t2 = time.perf_counter()
        names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
        events = [c for c in names if c.startswith('v2_')]
        keys = raw.select('pair_id', 'hand_id', 'table_id').join(p.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        cas = []
        actions_total = 0
        missing_total = 0
        for pool in these:
            ca, n_act, miss = conditional_features(keys.filter(pl.col('table_id') == pool), h.select('hand_id', 'table_id', 'board_cards', 'button_seat', 'big_blind'), policy, counts, caf)
            cas.append(ca)
            actions_total += n_act
            missing_total += miss
            gc.collect()
        ca = pl.concat(cas)
        t3 = time.perf_counter()
        pairframe = aggregate(raw, names).join(tails(raw, events), on=['pair_id', 'table_id'], validate='1:1').join(extra_aggregate(ca, caf), on=['pair_id', 'table_id'], validate='1:1').sort('pair_id')
        assert pairframe.height == p.height, (pairframe.height, p.height)
        pairframe.write_parquet(dest)
        rep = {'batch': i, 'pools': these, 'pairs': pairframe.height, 'pair_hands': raw.height, 'action_rows': actions_total, 'pair_hands_without_eligible_action': missing_total,
               'seconds': {'base': t1 - t0, 'v2': t2 - t1, 'conditional': t3 - t2, 'total': time.perf_counter() - t0}, 'sha256': sha(dest), 'identity_sha256': sha(out / 'identity.json')}
        write_json(receipt, rep)
        print(json.dumps(rep), flush=True)
        del raw, ca, cas, pairframe, keys
        gc.collect()


if __name__ == '__main__':
    main()
