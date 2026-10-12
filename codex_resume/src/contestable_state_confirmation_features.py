"""Nested-fold wrapper for frozen contestable state mechanics.

Callers supply explicit keys and a pair-id/player-id lookup. This module never
opens label, evidence, or evaluation files; all state mechanics remain in the
frozen contestable_state_features module.
"""
import os
import sys
from pathlib import Path

for key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl

from contestable_state_features import FEATURES, state_features

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
DATA = ROOT / 'data'
KEYS = ('pair_id', 'hand_id')
PAIR_COLUMNS = ('pair_id', 'player_1', 'player_2')


def build_for_keys(keys, pair_players):
    """Build the frozen 29 features for explicit unique pair/hand keys.

    `pair_players` must be supplied by the caller and contain only the named
    pair and player identifiers. The only final-state field reached by the
    delegated mechanics is `seats.total_contribution`, used exclusively for
    its final contribution accounting assertion.
    """
    if not isinstance(keys, pl.DataFrame) or not isinstance(pair_players, pl.DataFrame):
        raise TypeError('keys and pair_players must be Polars DataFrames')
    if not set(KEYS).issubset(keys.columns) or not set(PAIR_COLUMNS).issubset(pair_players.columns):
        raise ValueError('missing explicit key or pair-player columns')
    selected = keys.select(*KEYS).sort(*KEYS)
    if selected.height == 0 or selected.unique().height != selected.height:
        raise ValueError('keys must be non-empty and unique by pair_id/hand_id')
    supplied = pair_players.select(*PAIR_COLUMNS).unique().sort('pair_id')
    if supplied.height != pair_players.select('pair_id').n_unique():
        raise ValueError('pair_players must have one player pair per pair_id')
    selected_pairs = selected.select('pair_id').unique()
    supplied = supplied.join(selected_pairs, on='pair_id', how='semi')
    if supplied.height != selected_pairs.height:
        raise ValueError('pair_players does not cover every selected pair')
    pair_map = {row['pair_id']: (row['player_1'], row['player_2']) for row in supplied.iter_rows(named=True)}
    hand_ids = selected['hand_id'].unique().to_list()
    hands = pl.read_parquet(DATA / 'hands.parquet', columns=['hand_id', 'button_seat', 'small_blind', 'big_blind', 'players_dealt'])
    hand_map = {row['hand_id']: row for row in hands.filter(pl.col('hand_id').is_in(hand_ids)).iter_rows(named=True)}
    seat_data = pl.read_parquet(DATA / 'seats.parquet').filter(pl.col('hand_id').is_in(hand_ids))
    seat_map = {key[0]: frame.to_dicts() for key, frame in seat_data.group_by('hand_id')}
    action_data = (pl.scan_parquet(DATA / 'actions.parquet').filter(pl.col('hand_id').is_in(hand_ids))
                   .sort('hand_id', 'action_no').collect())
    action_map = {}
    for row in action_data.iter_rows(named=True):
        action_map.setdefault(row['hand_id'], []).append(row)
    output, counts = [], {'pot_before_checks': 0, 'stack_before_checks': 0, 'final_contribution_checks': 0,
                          'pair_decisions': 0, 'pair_decisions_partner_allin': 0}
    for pair_id, hand_id in selected.iter_rows():
        if hand_id not in hand_map or hand_id not in seat_map or hand_id not in action_map:
            raise RuntimeError(f'missing public state input for {pair_id}/{hand_id}')
        features, parity = state_features(hand_map[hand_id], seat_map[hand_id], action_map[hand_id], pair_map[pair_id])
        output.append({'pair_id': pair_id, 'hand_id': hand_id, **features})
        for name, value in parity.items():
            counts[name] += value
    frame = pl.DataFrame(output).sort(*KEYS)
    if not frame.select(*KEYS).equals(selected):
        raise AssertionError('feature key ordering changed')
    if frame.columns != list(KEYS) + list(FEATURES):
        raise AssertionError('feature manifest changed')
    if not np.isfinite(frame.select(FEATURES).to_numpy()).all():
        raise AssertionError('non-finite feature value')
    receipt = {'keys': {'pair_hands': selected.height, 'pairs': selected['pair_id'].n_unique(),
                        'distinct_hands': selected['hand_id'].n_unique()},
               'action_rows_loaded': action_data.height, 'state_parity_counts': counts,
               'feature_count': len(FEATURES),
               'final_contribution_verification_only': True}
    return frame, receipt
