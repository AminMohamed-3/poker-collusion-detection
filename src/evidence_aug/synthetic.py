"""Sequence / equity / conditional-action features for pair-hands of any gameplay directory.

Pure re-implementation of the three original builders with the data path as a
parameter (the originals hard-code the project data directory). The extractors
themselves are imported unchanged from the copied src/.
"""
import sys
from collections import defaultdict
from evidence_aug.paths import SRC, SCREEN
sys.path.insert(0, str(SRC))
import numpy as np
import polars as pl
from sequence_evidence_features import extract_pair_hand as seq_extract
from equity_evidence_features import CARDS, extract_pair_hand as eq_extract, _canonical_equity
from conditional_action_features import INPUTS, action_class, card_features
from conditional_inference_worker import extract_chunk
from run_conditional_policy_screen import hand_features
from evidence_aug.frame import SEQ, EQ

KEYS3 = ['pair_id', 'hand_id', 'table_id']


def gameplay(data, ids):
    """hands/seats/actions restricted to the given hand ids (development phase only)."""
    hands = pl.scan_parquet(data / 'hands.parquet').filter(pl.col('phase') == 'development').join(ids.lazy(), on='hand_id').collect()
    assert hands.height == ids.height
    seats = pl.scan_parquet(data / 'seats.parquet').join(ids.lazy(), on='hand_id').collect()
    actions = pl.scan_parquet(data / 'actions.parquet').join(ids.lazy(), on='hand_id').collect().sort('hand_id', 'action_no')
    return hands, seats, actions


def sequence_features(keys, hands, seats, actions):
    keys = keys.sort('hand_id', 'pair_id')
    blind = dict(hands.select('hand_id', 'big_blind').iter_rows())
    seat_lookup = {}
    for h, p, contribution, net in seats.select('hand_id', 'player_id', 'total_contribution', 'net_chips').iter_rows():
        seat_lookup.setdefault(h, {})[p] = (contribution, net)
    action_lookup = {}
    for row in actions.select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call', 'pot_before').iter_rows(named=True):
        action_lookup.setdefault(row.pop('hand_id'), []).append(row)
    out = [(r['pair_id'], r['hand_id'], *seq_extract(r['player_1'], r['player_2'], blind[r['hand_id']], seat_lookup[r['hand_id']], action_lookup.get(r['hand_id'], [])))
           for r in keys.iter_rows(named=True)]
    return pl.DataFrame(out, schema=['pair_id', 'hand_id', *SEQ], orient='row').with_columns(pl.col(SEQ).cast(pl.Float32))


def equity_features(keys, hands, seats, actions):
    keys = keys.sort('hand_id', 'pair_id')
    hand_lookup = {h: (bb, [CARDS[c] for c in (board or '').split()]) for h, bb, board in hands.select('hand_id', 'big_blind', 'board_cards').iter_rows()}
    seat_lookup = {}
    for h, p, c1, c2 in seats.select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2').iter_rows():
        seat_lookup.setdefault(h, {})[p] = (CARDS[c1], CARDS[c2])
    action_lookup = {}
    for row in actions.select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call', 'pot_before', 'stack_before').iter_rows(named=True):
        action_lookup.setdefault(row.pop('hand_id'), []).append(row)
    rows = []
    for pair, hand, first, second in keys.select('pair_id', 'hand_id', 'player_1', 'player_2').iter_rows():
        blind, board = hand_lookup[hand]
        rows.append((pair, hand, *eq_extract(first, second, blind, seat_lookup[hand], board, action_lookup.get(hand, []))))
    return pl.DataFrame(rows, schema=['pair_id', 'hand_id', *EQ], orient='row').with_columns(pl.col(EQ).cast(pl.Float32))


def style_counts(data):
    """Per-player action-class totals over every development hand of the population (public counts)."""
    ids = pl.scan_parquet(data / 'hands.parquet').filter(pl.col('phase') == 'development').select('hand_id')
    cls = pl.when(pl.col('action') == 'fold').then(0).when(pl.col('action') == 'check').then(1).when(
        (pl.col('action') == 'call') | ((pl.col('action') == 'all_in') & (pl.col('amount') <= pl.col('to_call')))).then(2).otherwise(3)
    totals = pl.scan_parquet(data / 'actions.parquet').join(ids, on='hand_id', how='semi').with_columns(cls.alias('kind')).group_by('player_id').agg(
        *[(pl.col('kind') == k).sum().alias('n' + str(k)) for k in range(4)]).collect()
    return {r[0]: np.array(r[1:], dtype=np.int64) for r in totals.iter_rows()}


def conditional_features(keys, hands, seats, actions, counts, policy):
    """Action rows -> frozen reference residuals -> 33 ca_ hand features, left-joined to keys with zero fill."""
    hands = hands.select('hand_id', 'table_id', 'board_cards', 'button_seat', 'big_blind')
    frame = extract_chunk(keys.select('pair_id', 'hand_id', 'player_1', 'player_2'), hands, seats, actions, counts)
    assert policy['inputs'] == INPUTS
    X = frame.select(INPUTS).to_numpy().astype(np.float32)
    ca = hand_features(frame, policy['classifier'].predict_proba(X, num_threads=2), policy['regressor'].predict(X, num_threads=2))
    caf = [c for c in ca.columns if c.startswith('ca_')]
    missing = keys.select('pair_id', 'hand_id').join(ca.select('pair_id', 'hand_id'), on=['pair_id', 'hand_id'], how='anti').height
    ca = keys.select(KEYS3).join(ca.drop('table_id'), on=['pair_id', 'hand_id'], how='left', validate='1:1').with_columns(pl.col(caf).fill_null(0))
    assert ca.height == keys.height
    card_features.cache_clear()
    return ca.sort('pair_id', 'hand_id'), caf, {'action_rows': frame.height, 'pair_hands_without_eligible_action': missing}


def clear_caches():
    card_features.cache_clear()
    _canonical_equity.cache_clear()
