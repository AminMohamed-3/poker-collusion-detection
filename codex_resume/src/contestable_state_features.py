"""Build label-free public contestable-pot state features for the current audit top20.

The reconstruction uses only pre-action public stacks, blinds, and action amounts.
Final seat contributions are read solely as an accounting assertion after every hand.
"""
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'research' / 'contestable_state_20260912'
PREV = HERE / 'experiments' / 'symmetry_censor_20260912'
DATA = ROOT / 'data'
KEYS = ('pair_id', 'hand_id')

FEATURES = [
    'cs_decisions', 'cs_decisions_partner_allin', 'cs_decisions_outsider_allin',
    'cs_decisions_any_allin', 'cs_decisions_both_can_wager',
    'cs_effective_remaining_bb_mean', 'cs_effective_remaining_bb_max',
    'cs_effective_remaining_pot_mean', 'cs_effective_remaining_pot_max',
    'cs_contribution_gap_partner_bb_mean', 'cs_contribution_gap_partner_bb_max',
    'cs_actor_unreachable_frac_mean', 'cs_actor_unreachable_frac_max',
    'cs_partner_unreachable_frac_mean', 'cs_partner_unreachable_frac_max',
    'cs_live_opponents_can_bet_mean', 'cs_live_opponents_can_bet_max',
    'cs_fold_partner_active_count', 'cs_fold_partner_allin_count',
    'cs_check_partner_active_count', 'cs_check_partner_allin_count',
    'cs_call_partner_active_count', 'cs_call_partner_allin_count',
    'cs_call_partner_active_bb_sum', 'cs_call_partner_allin_bb_sum',
    'cs_aggr_partner_active_count', 'cs_aggr_partner_allin_count',
    'cs_aggr_partner_active_bb_sum', 'cs_aggr_partner_allin_bb_sum',
]
assert len(FEATURES) == 29


class MechanicalError(RuntimeError):
    def __init__(self, kind, hand_id, pair_id, action_no, expected, observed):
        self.detail = {'kind': kind, 'hand_id': hand_id, 'pair_id': pair_id,
                       'action_no': action_no, 'expected': expected, 'observed': observed}
        super().__init__(json.dumps(self.detail, sort_keys=True))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def zero_features():
    return {name: 0.0 for name in FEATURES}


def fail(kind, hand, pair, event, expected, observed):
    raise MechanicalError(kind, hand['hand_id'], pair[0] + '|' + pair[1],
                          int(event.get('action_no', -1)), expected, observed)


def state_features(hand, seat_rows, events, pair):
    """Return symmetric decision-state aggregates and accounting parity counts.

    `total_contribution` is intentionally consulted only after all actions to
    prove reconstruction parity. It is never consulted while deriving features.
    """
    p1, p2 = pair
    hand_id = hand['hand_id']
    big = int(hand['big_blind'])
    small = int(hand['small_blind'])
    button = int(hand['button_seat'])
    if big <= 0 or small < 0:
        raise MechanicalError('invalid_blinds', hand_id, p1 + '|' + p2, -1, 'positive big blind', [small, big])
    if len(seat_rows) != 6 or int(hand['players_dealt']) != 6:
        raise MechanicalError('six_player_assumption', hand_id, p1 + '|' + p2, -1, 6,
                              {'seat_rows': len(seat_rows), 'players_dealt': hand['players_dealt']})
    by_player = {row['player_id']: row for row in seat_rows}
    if len(by_player) != 6 or p1 not in by_player or p2 not in by_player:
        raise MechanicalError('seat_or_pair_coverage', hand_id, p1 + '|' + p2, -1, 'six unique seats containing pair', len(by_player))
    seats = {int(row['seat_no']) for row in seat_rows}
    if seats != set(range(6)) or button not in seats:
        raise MechanicalError('six_seat_numbering', hand_id, p1 + '|' + p2, -1, list(range(6)), sorted(seats))
    seat_player = {int(row['seat_no']): row['player_id'] for row in seat_rows}
    sb_player = seat_player[(button + 1) % 6]
    bb_player = seat_player[(button + 2) % 6]
    starting = {player: int(row['starting_stack']) for player, row in by_player.items()}
    contribution = {player: 0 for player in by_player}
    if starting[sb_player] < small or starting[bb_player] < big:
        raise MechanicalError('blind_exceeds_starting_stack', hand_id, p1 + '|' + p2, -1,
                              {'sb': small, 'bb': big}, {'sb_stack': starting[sb_player], 'bb_stack': starting[bb_player]})
    contribution[sb_player] = small
    contribution[bb_player] = big
    folded = set()
    all_in = {player for player in by_player if starting[player] == contribution[player] and starting[player] > 0}
    feature = zero_features()
    sums = {name: 0.0 for name in ('effective_bb', 'effective_pot', 'gap_bb', 'actor_unreachable', 'partner_unreachable', 'live_opponents')}
    maxima = {name: 0.0 for name in sums}
    parity = {'pot_before_checks': 0, 'stack_before_checks': 0, 'final_contribution_checks': 0,
              'pair_decisions': 0, 'pair_decisions_partner_allin': 0}

    for event in events:
        actor = event['player_id']
        if actor not in by_player:
            fail('action_actor_missing_seat', hand, pair, event, 'seat player', actor)
        if actor in folded:
            fail('action_after_fold', hand, pair, event, 'unfolded actor', actor)
        if actor in all_in or starting[actor] == contribution[actor]:
            fail('action_after_all_in', hand, pair, event, 'positive remaining stack', actor)
        pot_before = int(event['pot_before'])
        expected_pot = sum(contribution.values())
        if pot_before != expected_pot:
            fail('pot_before', hand, pair, event, expected_pot, pot_before)
        parity['pot_before_checks'] += 1
        stack_before = int(event['stack_before'])
        expected_stack = starting[actor] - contribution[actor]
        if stack_before != expected_stack:
            fail('stack_before', hand, pair, event, expected_stack, stack_before)
        parity['stack_before_checks'] += 1
        amount = int(event['amount'])
        if amount < 0 or amount > stack_before:
            fail('action_amount_bounds', hand, pair, event, '[0, stack_before]', {'amount': amount, 'stack_before': stack_before})

        if actor in pair:
            partner = p2 if actor == p1 else p1
            if partner not in folded:
                partner_allin = partner in all_in
                outsiders_allin = any(player not in pair and player not in folded for player in all_in)
                any_allin = bool((all_in - folded))
                actor_remaining = stack_before
                partner_remaining = starting[partner] - contribution[partner]
                both_can_wager = actor_remaining > 0 and partner_remaining > 0 and not partner_allin
                effective = min(actor_remaining, partner_remaining) if both_can_wager else 0
                call_paid = min(max(int(event['to_call']), 0), actor_remaining)
                actor_target = contribution[actor] + call_paid
                total_after_call = pot_before + call_paid
                contestable_after_call = sum(min(value + (call_paid if player == actor else 0), actor_target)
                                             for player, value in contribution.items())
                actor_unreachable = (total_after_call - contestable_after_call) / max(total_after_call, 1)
                if actor_unreachable < -1e-12 or actor_unreachable > 1 + 1e-12:
                    fail('actor_unreachable_bounds', hand, pair, event, '[0, 1]', actor_unreachable)
                actor_unreachable = min(max(actor_unreachable, 0.0), 1.0)
                if partner_allin:
                    partner_target = contribution[partner]
                    partner_contestable = sum(min(value, partner_target) for value in contribution.values())
                    partner_unreachable = (pot_before - partner_contestable) / max(pot_before, 1)
                    if partner_unreachable < -1e-12 or partner_unreachable > 1 + 1e-12:
                        fail('partner_unreachable_bounds', hand, pair, event, '[0, 1]', partner_unreachable)
                    partner_unreachable = min(max(partner_unreachable, 0.0), 1.0)
                else:
                    partner_unreachable = 0.0
                live_opponents = sum(1 for player in by_player if player != actor and player not in folded
                                     and player not in all_in and starting[player] > contribution[player])
                values = {
                    'effective_bb': effective / big,
                    'effective_pot': effective / max(pot_before, 1),
                    'gap_bb': abs(contribution[actor] - contribution[partner]) / big,
                    'actor_unreachable': actor_unreachable,
                    'partner_unreachable': partner_unreachable,
                    'live_opponents': float(live_opponents),
                }
                feature['cs_decisions'] += 1
                feature['cs_decisions_partner_allin'] += float(partner_allin)
                feature['cs_decisions_outsider_allin'] += float(outsiders_allin)
                feature['cs_decisions_any_allin'] += float(any_allin)
                feature['cs_decisions_both_can_wager'] += float(both_can_wager)
                parity['pair_decisions'] += 1
                parity['pair_decisions_partner_allin'] += int(partner_allin)
                for name, value in values.items():
                    sums[name] += value
                    maxima[name] = max(maxima[name], value)
                suffix = 'partner_allin' if partner_allin else 'partner_active'
                action = event['action']
                if action == 'fold':
                    feature[f'cs_fold_{suffix}_count'] += 1
                elif action == 'check':
                    feature[f'cs_check_{suffix}_count'] += 1
                elif action == 'call' or (action == 'all_in' and amount <= int(event['to_call'])):
                    feature[f'cs_call_{suffix}_count'] += 1
                    feature[f'cs_call_{suffix}_bb_sum'] += amount / big
                elif action in ('bet', 'raise') or (action == 'all_in' and amount > int(event['to_call'])):
                    feature[f'cs_aggr_{suffix}_count'] += 1
                    feature[f'cs_aggr_{suffix}_bb_sum'] += amount / big
                else:
                    fail('unknown_action', hand, pair, event, 'fold/check/call/bet/raise/all_in', action)

        contribution[actor] += amount
        if event['action'] == 'fold':
            folded.add(actor)
        if contribution[actor] == starting[actor] and starting[actor] > 0 and actor not in folded:
            all_in.add(actor)

    decisions = feature['cs_decisions']
    mapping = {
        'effective_bb': 'cs_effective_remaining_bb',
        'effective_pot': 'cs_effective_remaining_pot',
        'gap_bb': 'cs_contribution_gap_partner_bb',
        'actor_unreachable': 'cs_actor_unreachable_frac',
        'partner_unreachable': 'cs_partner_unreachable_frac',
        'live_opponents': 'cs_live_opponents_can_bet',
    }
    for raw, prefix in mapping.items():
        feature[prefix + '_mean'] = sums[raw] / decisions if decisions else 0.0
        feature[prefix + '_max'] = maxima[raw]
    for player, row in by_player.items():
        if contribution[player] != int(row['total_contribution']):
            raise MechanicalError('final_contribution', hand_id, p1 + '|' + p2, -1,
                                  int(row['total_contribution']), contribution[player])
        parity['final_contribution_checks'] += 1
    if set(feature) != set(FEATURES) or not all(math.isfinite(value) for value in feature.values()):
        raise MechanicalError('feature_output', hand_id, p1 + '|' + p2, -1, FEATURES, feature)
    return feature, parity


def target_keys():
    saved = pl.read_parquet(PREV / 'scores.parquet').sort(*KEYS)
    ranks = saved.with_columns(pl.col('stage1_only').rank('ordinal', descending=True).over('pair_id').alias('_rank'))
    keys = ranks.filter(pl.col('_rank') <= 20).select(*KEYS).sort(*KEYS)
    assert keys.height == 3240 and keys['pair_id'].n_unique() == 162
    return keys


def build():
    keys = target_keys()
    pair_rows = (pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
                 .join(keys.select('pair_id').unique(), on='pair_id', how='semi'))
    pair_map = {row['pair_id']: (row['player_1'], row['player_2']) for row in pair_rows.iter_rows(named=True)}
    if len(pair_map) != 162:
        raise RuntimeError('pair-player lookup does not cover every selected pair')
    hand_ids = keys['hand_id'].unique().to_list()
    hands = pl.read_parquet(DATA / 'hands.parquet', columns=['hand_id', 'button_seat', 'small_blind', 'big_blind', 'players_dealt'])
    hands = hands.filter(pl.col('hand_id').is_in(hand_ids))
    hand_map = {row['hand_id']: row for row in hands.iter_rows(named=True)}
    seat_data = pl.read_parquet(DATA / 'seats.parquet').filter(pl.col('hand_id').is_in(hand_ids))
    seat_map = {key[0]: rows for key, rows in seat_data.group_by('hand_id')}
    actions = (pl.scan_parquet(DATA / 'actions.parquet').filter(pl.col('hand_id').is_in(hand_ids))
               .sort('hand_id', 'action_no').collect())
    action_map = {}
    for row in actions.iter_rows(named=True):
        action_map.setdefault(row['hand_id'], []).append(row)
    rows, totals = [], {'pot_before_checks': 0, 'stack_before_checks': 0, 'final_contribution_checks': 0,
                         'pair_decisions': 0, 'pair_decisions_partner_allin': 0}
    for pair_id, hand_id in keys.iter_rows():
        if hand_id not in hand_map or hand_id not in seat_map or hand_id not in action_map:
            raise RuntimeError(f'missing public hand input for {pair_id}/{hand_id}')
        feature, parity = state_features(hand_map[hand_id], seat_map[hand_id].to_dicts(), action_map[hand_id], pair_map[pair_id])
        rows.append({'pair_id': pair_id, 'hand_id': hand_id, **feature})
        for name, value in parity.items():
            totals[name] += value
    output = pl.DataFrame(rows).sort(*KEYS)
    assert output.select(*KEYS).equals(keys)
    assert output.width == len(FEATURES) + 2 and all(dtype.is_numeric() for dtype in output.dtypes[2:])
    assert np.isfinite(output.select(FEATURES).to_numpy()).all()
    return output, totals, keys, actions.height


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        output, parity, keys, action_rows = build()
    except MechanicalError as error:
        write_json(OUT / 'FAILED.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                                         'status': 'stopped_mechanical_assumption_failure', 'detail': error.detail})
        raise
    output.write_parquet(OUT / 'features.parquet')
    manifest = {
        'feature_count': len(FEATURES), 'features': FEATURES,
        'symmetric': 'Every decision is aggregated across either pair member; no A/B orientation is emitted.',
        'decision_scope': 'Actor is a pair member and partner has not folded, evaluated before the action.',
        'money_normalization': 'All amounts are normalized by the hand big blind or contemporaneous pot.',
        'contestable_proxy': 'Unreachable fractions cap every current contribution at the actor or all-in partner contribution. This is public chip accounting, not exact strategic EV.',
        'excluded': 'No IDs, timestamps, cards, board, labels, outcomes, final pot, net chips, or final contribution are features.',
    }
    sources = [Path(__file__), PREV / 'scores.parquet', DATA / 'hands.parquet', DATA / 'seats.parquet', DATA / 'actions.parquet', DATA / 'development_labels.csv']
    receipt = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'complete_label_free_extraction',
        'keys': {'pair_hands': keys.height, 'pairs': keys['pair_id'].n_unique(), 'distinct_hands': keys['hand_id'].n_unique()},
        'action_rows_loaded': action_rows, 'state_parity_counts': parity,
        'six_player_assumption_verified_per_distinct_hand': keys['hand_id'].n_unique(),
        'source_hashes': {str(path): sha(path) for path in sources},
        'feature_manifest': 'feature_manifest.json',
        'limitations': ['Contestable-pot features are public chip-accounting proxies, not exact strategic expected value.',
                        'Final seat contributions are verification-only and never inference inputs.',
                        'No relevance, behavior label, outer evidence or evaluation data was read.']
    }
    write_json(OUT / 'feature_manifest.json', manifest)
    write_json(OUT / 'receipt.json', receipt)
    print(json.dumps({'features': str(OUT / 'features.parquet'), 'shape': output.shape, 'parity': parity}, indent=2))


if __name__ == '__main__':
    main()


