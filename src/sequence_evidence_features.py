"""Pair-symmetric action-sequence features from observed gameplay only.

Tracks third-player responses and the order of pair aggression and capitulation.
No labels, hand-ID structure, timestamps, or future board cards are inputs.
"""
from __future__ import annotations

import math
from collections import defaultdict

NAMES = [
    'outsider_fold_to_pair_n', 'outsider_fold_to_pair_invested_bb',
    'outsider_call_to_pair_bb', 'outsider_raise_to_pair_n',
    'outsider_fold_after_both_raise_n', 'outsider_fold_after_both_raise_invested_bb',
    'outsider_call_after_both_raise_bb', 'pair_reraise_multiway_n',
    'pair_raise_after_partner_multiway_n', 'pair_fold_after_outsider_fold_n',
    'pair_fold_after_outsider_fold_invested_bb', 'pair_fold_after_both_raise_n',
    'pair_fold_after_both_raise_invested_bb', 'heads_up_mutual_check_streets',
    'both_pair_raise_streets', 'pair_raise_after_outsiders_gone_n',
    'partner_call_after_outsiders_gone_bb', 'outsiders_contribution_bb',
    'outsiders_net_loss_bb', 'pair_profit_after_both_raise_bb',
    'max_pair_raise_pot_fraction_multiway', 'max_outsider_fold_odds_after_both_raise',
    'max_same_street_min_pair_raises',
]


def extract_pair_hand(first, second, big_blind, seats, actions):
    """seats: {player: (contribution, net)}; actions: dictionaries in any order."""
    pair = {first, second}
    if len(pair) != 2 or not pair.issubset(seats) or big_blind <= 0:
        raise ValueError('Invalid pair, seat coverage or blind')
    numbers = [a['action_no'] for a in actions]
    if len(numbers) != len(set(numbers)):
        raise ValueError('Duplicate action number within hand')
    actions = sorted(actions, key=lambda a: a['action_no'])
    values = dict.fromkeys(NAMES, 0.)
    live = set(seats)
    investment = defaultdict(float)
    street = None
    previous = None
    raises = {first: 0, second: 0}
    checks = set()
    outsider_folds = 0
    ended_streets = set()

    def finish_street():
        if street is not None:
            values['heads_up_mutual_check_streets'] += len(checks) == 2
            values['both_pair_raise_streets'] += min(raises.values()) > 0
            values['max_same_street_min_pair_raises'] = max(values['max_same_street_min_pair_raises'], min(raises.values()))

    for event in actions:
        current_street = event['street']
        if current_street != street:
            finish_street()
            if current_street in ended_streets:
                raise ValueError('Noncontiguous betting street')
            if street is not None:
                ended_streets.add(street)
            street, previous = current_street, None
            raises, checks, outsider_folds = {first: 0, second: 0}, set(), 0
        player, action = event['player_id'], event['action']
        if player not in live:
            raise ValueError('Player acts after folding or without a seat')
        amount, to_call = event['amount'], event['to_call']
        if amount < 0 or to_call < 0:
            raise ValueError('Negative action amount')
        aggressive = action in ('bet', 'raise', 'all_in') and amount > to_call
        calling = action in ('call', 'all_in') and not aggressive
        outsiders_live = len(live-pair)
        both_raised = min(raises.values()) > 0
        to_pair = previous in pair
        if player not in pair:
            if action == 'fold':
                if to_pair:
                    values['outsider_fold_to_pair_n'] += 1
                    values['outsider_fold_to_pair_invested_bb'] += investment[player]/big_blind
                    if both_raised:
                        values['outsider_fold_after_both_raise_n'] += 1
                        values['outsider_fold_after_both_raise_invested_bb'] += investment[player]/big_blind
                        odds = to_call/max(event['pot_before']+to_call, 1)
                        values['max_outsider_fold_odds_after_both_raise'] = max(values['max_outsider_fold_odds_after_both_raise'], odds)
                outsider_folds += 1
            if to_pair and calling:
                values['outsider_call_to_pair_bb'] += amount/big_blind
                if both_raised:
                    values['outsider_call_after_both_raise_bb'] += amount/big_blind
            if to_pair and aggressive:
                values['outsider_raise_to_pair_n'] += 1
        else:
            partner = second if player == first else first
            partner_live = partner in live
            to_partner = previous == partner and partner_live
            if aggressive and partner_live:
                if outsiders_live:
                    values['pair_reraise_multiway_n'] += to_partner
                    values['pair_raise_after_partner_multiway_n'] += raises[partner] > 0
                    ratio = amount/max(event['pot_before']+to_call, big_blind)
                    values['max_pair_raise_pot_fraction_multiway'] = max(values['max_pair_raise_pot_fraction_multiway'], ratio)
                else:
                    values['pair_raise_after_outsiders_gone_n'] += 1
            if action == 'fold' and to_partner:
                if outsider_folds:
                    values['pair_fold_after_outsider_fold_n'] += 1
                    values['pair_fold_after_outsider_fold_invested_bb'] += investment[player]/big_blind
                if both_raised:
                    values['pair_fold_after_both_raise_n'] += 1
                    values['pair_fold_after_both_raise_invested_bb'] += investment[player]/big_blind
            if not outsiders_live and to_partner and calling:
                values['partner_call_after_outsiders_gone_bb'] += amount/big_blind
            if action == 'check' and not outsiders_live and partner_live:
                checks.add(player)
            if aggressive:
                raises[player] += 1
        investment[player] += amount
        if aggressive:
            previous = player
        if action == 'fold':
            live.remove(player)
    finish_street()
    outsiders = set(seats)-pair
    values['outsiders_contribution_bb'] = sum(seats[p][0] for p in outsiders)/big_blind
    values['outsiders_net_loss_bb'] = max(0., -sum(seats[p][1] for p in outsiders))/big_blind
    values['pair_profit_after_both_raise_bb'] = max(0., sum(seats[p][1] for p in pair))/big_blind if values['both_pair_raise_streets'] else 0.
    if not all(math.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError('Invalid sequence feature')
    return [float(values[name]) for name in NAMES]
