"""Street-specific card strength and opponent-aware gameplay features.

Uses only disclosed gameplay, with the board prefix available at that street.
Original baseline files are immutable inputs; augmented features live separately.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import polars as pl
from phevaluator import Card, evaluate_cards

ROOT = Path(__file__).resolve().parents[1]
DATA, BASE, OUT = ROOT / 'data', ROOT / 'outputs', ROOT / 'outputs_v2'
CARDS = {r + s: Card.to_id(r + s) for r in '23456789TJQKA' for s in 'cdhs'}
STREETS = ['preflop', 'flop', 'turn', 'river']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflop_strength(c1, c2):
    """A bounded rank/suit/connectivity feature, not an equity probability."""
    r1, r2 = c1 // 4 + 2, c2 // 4 + 2
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return .55 + hi / 32
    return min(.95, max(.05, (hi + lo) / 32 - max(hi-lo-1, 0) / 40
                        + .07 * (c1 % 4 == c2 % 4)))


def strength_table(seats, hands):
    boards = {h: [CARDS[c] for c in (b or '').split()]
              for h, b in hands.select('hand_id', 'board_cards').iter_rows()}
    rows = []
    for h, player, c1, c2 in seats.select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2').iter_rows():
        first, second = CARDS[c1], CARDS[c2]
        board = boards[h]
        ranks = [7463 - int(preflop_strength(first, second) * 7462)]
        for n in [3, 4, 5]:
            ranks.append(evaluate_cards(first, second, *board[:n]) if len(board) >= n else None)
        rows.append((h, player, *ranks))
    return pl.DataFrame(rows, schema=['hand_id', 'player_id', *[f'rank_{s}' for s in STREETS]],
                        orient='row', schema_overrides={f'rank_{s}': pl.Int32 for s in STREETS})


def street_rank(prefix=''):
    return pl.coalesce(*[pl.when(pl.col('street') == s).then(pl.col(f'{prefix}rank_{s}'))
                         for s in STREETS])


def directional_events(keys, actions, seatinfo, actor, partner):
    # Join only the actor's actual actions, then attach this partner's state.
    mapping = keys.select('pair_id', 'hand_id', pl.col(actor).alias('player_id'),
                          pl.col(partner).alias('partner_id'))
    events = mapping.join(actions, on=['hand_id', 'player_id'], how='inner')
    events = events.join(seatinfo, on=['hand_id', 'player_id'], how='left')
    partnerinfo = seatinfo.rename({c: ('partner_id' if c == 'player_id' else 'partner_' + c)
                                  for c in seatinfo.columns if c != 'hand_id'})
    events = events.join(partnerinfo, on=['hand_id', 'partner_id'], how='left')
    events = events.with_columns(street_rank().alias('actor_rank'),
                                street_rank('partner_').alias('partner_rank'))
    if events['actor_rank'].null_count() or events['partner_rank'].null_count():
        raise ValueError('A betting street has no corresponding board prefix')
    events = events.with_columns(
        ((7463-pl.col('actor_rank')) / 7462).alias('strength'),
        ((7463-pl.col('partner_rank')) / 7462).alias('partner_strength'),
        (pl.col('previous_aggressor') == pl.col('partner_id')).fill_null(False).alias('to_partner'),
        (pl.col('action_no') < pl.col('partner_fold_no')).alias('partner_active'),
        (pl.col('actor_rank') > pl.col('partner_rank')).alias('behind_partner'),
        (pl.col('to_call') / (pl.col('pot_before') + pl.col('to_call')).clip(lower_bound=1)).alias('odds'),
        (pl.col('amount') / pl.col('big_blind')).alias('cost_bb'),
        (pl.col('amount') / pl.col('stack_before').clip(lower_bound=1)).alias('commitment'),
    )
    call = pl.col('action').is_in(['call', 'all_in']) & ~pl.col('aggressive')
    fold, check, agg = pl.col('action') == 'fold', pl.col('action') == 'check', pl.col('aggressive')
    to, live, behind = pl.col('to_partner'), pl.col('partner_active'), pl.col('behind_partner')
    ahead = pl.col('actor_rank') < pl.col('partner_rank')
    late, river = pl.col('street').is_in(['turn', 'river']), pl.col('street') == 'river'
    multi = pl.col('players_active') >= 3
    strength, weak = pl.col('strength'), 1 - pl.col('strength')
    cost, commit = pl.col('cost_bb'), pl.col('commitment')
    price = 1 - pl.col('odds')
    lost = (pl.col('net_chips') < 0) & (pl.col('partner_net_chips') > 0)
    own_value = (-pl.col('net_chips') / pl.col('big_blind')).clip(lower_bound=0)
    event_exprs = {
        'weak_call': call.cast(pl.Float32) * weak * cost,
        'weak_partner_call': call.cast(pl.Float32) * to * weak * cost,
        'weak_other_call': call.cast(pl.Float32) * ~to * weak * cost,
        'behind_partner_call': call.cast(pl.Float32) * to * behind * cost,
        'river_losing_call': call.cast(pl.Float32) * river * behind * live * cost,
        'river_partner_call': call.cast(pl.Float32) * river * to * behind * cost,
        'weak_raise': agg.cast(pl.Float32) * live * weak * cost,
        'weak_raise_behind': agg.cast(pl.Float32) * live * behind * weak * cost,
        'commit_behind': (call | agg).cast(pl.Float32) * live * behind * commit,
        'partner_commit_behind': (call | agg).cast(pl.Float32) * to * behind * commit,
        'lost_weak_call': call.cast(pl.Float32) * to * weak * lost * cost,
        'soft_favorable_fold': fold.cast(pl.Float32) * to * price,
        'soft_strength_fold': fold.cast(pl.Float32) * to * price * strength,
        'soft_ahead_fold': fold.cast(pl.Float32) * to * price * ahead,
        'soft_costly_fold': fold.cast(pl.Float32) * to * price * own_value,
        'other_favorable_fold': fold.cast(pl.Float32) * ~to * price,
        'other_strength_fold': fold.cast(pl.Float32) * ~to * price * strength,
        'soft_late_check': check.cast(pl.Float32) * late * live * strength,
        'soft_heads_up_check': check.cast(pl.Float32) * late * live * ~multi * strength,
        'soft_ahead_check': check.cast(pl.Float32) * late * live * ahead * strength,
        'isolation_reraise': agg.cast(pl.Float32) * to * multi * live,
        'isolation_weak_reraise': agg.cast(pl.Float32) * to * multi * live * weak * cost.log1p(),
        'isolation_both_weak': agg.cast(pl.Float32) * to * multi * live * weak * (1-pl.col('partner_strength')),
        'isolation_other_raise': agg.cast(pl.Float32) * ~to * multi * live * weak,
        'late_aggressive_strength': agg.cast(pl.Float32) * late * strength,
        'partner_call_count': call.cast(pl.Float32) * to,
        'partner_fold_count': fold.cast(pl.Float32) * to,
        'partner_ahead_call_count': call.cast(pl.Float32) * to * ahead,
    }
    result = events.group_by('pair_id', 'hand_id').agg(
        *[expr.cast(pl.Float32).sum().alias('v2_' + name) for name, expr in event_exprs.items()])
    return result


def build(phase, limit_batches=None):
    OUT.mkdir(exist_ok=True)
    cache = OUT / f'hands_{phase}'
    cache.mkdir(exist_ok=True)
    paths = sorted((BASE / f'hands_{phase}').glob('batch_*.parquet'))
    source = 'development_labels.csv' if phase == 'development' else 'evaluation_pairs.csv'
    identity = {'source_sha': sha(__file__), 'phevaluator': importlib.metadata.version('phevaluator'),
                'polars': pl.__version__, 'base': {p.name: sha(p) for p in paths},
                'inputs': {name: sha(DATA/name) for name in
                           ['hands.parquet', 'seats.parquet', 'actions.parquet', source]}}
    manifest = cache / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Advanced feature source/input mismatch; use a new output directory')
    else:
        if list(cache.glob('*.parquet')):
            raise ValueError('Untracked advanced feature cache')
        manifest.write_text(json.dumps(identity, indent=2) + '\n')
    pairs = pl.read_csv(DATA/source).select('pair_id', 'player_1', 'player_2')
    for i, path in enumerate(paths):
        if limit_batches is not None and i >= limit_batches:
            break
        dest = cache/path.name
        if dest.exists():
            print(f'{phase} {i}: cached', flush=True)
            continue
        raw = pl.read_parquet(path)
        keys = raw.select('pair_id', 'hand_id').join(pairs, on='pair_id')
        ids = keys.select('hand_id').unique()
        h = pl.scan_parquet(DATA/'hands.parquet').join(ids.lazy(), on='hand_id').collect()
        s = pl.scan_parquet(DATA/'seats.parquet').join(ids.lazy(), on='hand_id').collect()
        a = pl.scan_parquet(DATA/'actions.parquet').join(ids.lazy(), on='hand_id').collect().sort('hand_id', 'action_no')
        a = a.join(h.select('hand_id', 'big_blind'), on='hand_id').with_columns(
            (pl.col('action').is_in(['bet', 'raise', 'all_in']) &
             (pl.col('amount') > pl.col('to_call'))).alias('aggressive'))
        a = a.with_columns(pl.when('aggressive').then(pl.col('player_id')).otherwise(None)
                           .forward_fill().shift(1).over('hand_id', 'street').alias('previous_aggressor'))
        folds = a.filter(pl.col('action') == 'fold').group_by('hand_id', 'player_id').agg(pl.col('action_no').min().alias('fold_no'))
        si = strength_table(s, h).join(s.select('hand_id', 'player_id', 'net_chips'), on=['hand_id', 'player_id'])
        si = si.join(folds, on=['hand_id', 'player_id'], how='left').with_columns(pl.col('fold_no').fill_null(1000000))
        first = directional_events(keys, a, si, 'player_1', 'player_2')
        second = directional_events(keys, a, si, 'player_2', 'player_1')
        cols = [c for c in first.columns if c.startswith('v2_')]
        joined = first.join(second, on=['pair_id', 'hand_id'], how='full', coalesce=True, suffix='_other').fill_null(0)
        added = joined.select('pair_id', 'hand_id',
            *[(pl.col(c)+pl.col(c+'_other')).alias(c+'_sum') for c in cols],
            *[pl.max_horizontal(c, c+'_other').alias(c+'_max') for c in cols])
        result = raw.join(added, on=['pair_id', 'hand_id'], how='left')
        newcols = [c for c in result.columns if c.startswith('v2_')]
        result = result.with_columns(pl.col(newcols).fill_null(0).cast(pl.Float32))
        if result.height != raw.height:
            raise ValueError('Advanced joins changed pair-hand multiplicity')
        result.write_parquet(dest)
        print(f'{phase} {i}: {result.height} pair-hands, {len(newcols)} new features', flush=True)
        del raw, keys, h, s, a, si, first, second, joined, added, result
        gc.collect()
    expected = [cache/p.name for p in paths]
    if not all(p.exists() for p in expected):
        return
    raw = pl.scan_parquet(expected)
    cols = [c for c in raw.collect_schema().names() if c not in ['pair_id', 'hand_id', 'table_id']]
    result = raw.group_by('pair_id', 'table_id').agg(
        pl.len().alias('shared_hands'),
        *[pl.col(c).mean().alias(c+'_mean') for c in cols],
        *[pl.col(c).std().fill_null(0).alias(c+'_std') for c in cols],
        *[pl.col(c).max().alias(c+'_max') for c in cols],
        *[pl.col(c).quantile(.95).alias(c+'_q95') for c in cols],
    ).collect(engine='streaming')
    result.write_parquet(OUT/f'pairs_{phase}.parquet')
    print(f'{phase}: {result.height} pairs, {len(result.columns)-2} features', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['development', 'evaluation'])
    p.add_argument('--limit-batches', type=int)
    args = p.parse_args()
    build(args.phase, args.limit_batches)
