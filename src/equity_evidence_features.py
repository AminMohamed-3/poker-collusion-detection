"""Observer-side heads-up equity proxies using disclosed cards and street prefixes.

Equity averages uniformly over possible unseen board cards. It is NOT an estimate
of a player's beliefs, an opponent range, or multiway/side-pot expected value.
Neither other players' cards nor the eventual runout are used before their street.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import math

import numpy as np
from phevaluator import Card, evaluate_cards

CARDS = {r + s: Card.to_id(r + s) for r in '23456789TJQKA' for s in 'cdhs'}
PREFLOP_SAMPLES = 256
SAMPLE_SEED = 408331
_SAMPLE_INDICES = np.argsort(np.random.default_rng(SAMPLE_SEED).random((PREFLOP_SAMPLES, 48)), axis=1)[:, :5]
NAMES = [
    'partner_call_deficit_bb', 'partner_call_surplus_bb', 'partner_call_equity_cost_bb',
    'partner_fold_surplus_pot_bb', 'partner_fold_equity', 'partner_fold_underdog',
    'late_check_equity', 'heads_up_check_equity', 'partner_raise_equity_cost_bb',
    'multiway_raise_low_equity_logcost', 'partner_call_equity_drop_bb', 'partner_fold_equity_gain',
]
FEATURES = ['eq_' + name + suffix for suffix in ['_sum', '_max'] for name in NAMES]


@lru_cache(maxsize=200000)
def _canonical_equity(first, second, board):
    known = first + second + board
    if len(set(known)) != len(known) or len(board) not in (0, 3, 4, 5):
        raise ValueError('Invalid or duplicated cards')
    remaining = tuple(c for c in range(52) if c not in known)
    if not board:
        runouts = (tuple(remaining[i] for i in row) for row in _SAMPLE_INDICES)
        count = PREFLOP_SAMPLES
    else:
        runouts = combinations(remaining, 5 - len(board))
        count = math.comb(len(remaining), 5 - len(board))
    points = 0.
    for future in runouts:
        runout = board + future
        a = evaluate_cards(*first, *runout)
        b = evaluate_cards(*second, *runout)
        points += float(a < b) + .5 * float(a == b)
    return points / count


def heads_up_equity(first, second, board):
    """Exact postflop; 256 common random runouts preflop. Pair swap complements."""
    first, second, board = tuple(sorted(first)), tuple(sorted(second)), tuple(sorted(board))
    if len(first) != 2 or len(second) != 2:
        raise ValueError('Expected two hole cards per player')
    if first < second:
        return _canonical_equity(first, second, board)
    return 1. - _canonical_equity(second, first, board)


def extract_pair_hand(first, second, big_blind, seats, board, actions):
    """seats maps player IDs to two integer cards; board is the full card list."""
    pair = {first, second}
    if len(pair) != 2 or not pair.issubset(seats) or big_blind <= 0:
        raise ValueError('Invalid pair, seats or blind')
    live = set(seats)
    values = {p: dict.fromkeys(NAMES, 0.) for p in pair}
    street, previous = None, None
    equity = {}
    prior_equity = {}
    ordered = sorted(actions, key=lambda a: a['action_no'])
    if len({a['action_no'] for a in ordered}) != len(ordered):
        raise ValueError('Duplicate action number')
    for action in ordered:
        if action['street'] != street:
            street, previous = action['street'], None
        player = action['player_id']
        if player not in live:
            raise ValueError('Action after fold')
        kind, amount, to_call = action['action'], action['amount'], action['to_call']
        aggressive = kind in ('bet', 'raise', 'all_in') and amount > to_call
        calling = kind in ('call', 'all_in') and not aggressive
        if player in pair and pair.issubset(live):
            partner = second if player == first else first
            to_partner = previous == partner
            # Only compute states used by these gameplay features.
            relevant = (to_partner and (calling or kind == 'fold' or aggressive)) or kind == 'check' or aggressive
            if relevant:
                if street not in equity:
                    n = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}[street]
                    if len(board) < n:
                        raise ValueError('Betting street exceeds disclosed board')
                    equity[street] = heads_up_equity(seats[first], seats[second], board[:n])
                q = equity[street] if player == first else 1 - equity[street]
                cost = min(to_call, action.get('stack_before', to_call))
                pot = action['pot_before']
                price = cost / max(pot + cost, 1.)
                amount_bb = amount / big_blind
                # HU proxy: q*(pot+call)-call. Side pots and future betting excluded.
                surplus = max(0., q * (pot + cost) - cost) / big_blind
                deficit = max(0., cost - q * (pot + cost)) / big_blind
                v = values[player]
                past = prior_equity.get(player, q)
                if to_partner and calling:
                    v['partner_call_deficit_bb'] += deficit
                    v['partner_call_surplus_bb'] += surplus
                    v['partner_call_equity_cost_bb'] += q * amount_bb
                    v['partner_call_equity_drop_bb'] += max(0., past - q) * amount_bb
                if to_partner and kind == 'fold':
                    v['partner_fold_surplus_pot_bb'] += surplus
                    v['partner_fold_equity'] += q * (1 - price)
                    v['partner_fold_underdog'] += (1 - q) * (1 - price)
                    v['partner_fold_equity_gain'] += max(0., q - past)
                if kind == 'check':
                    v['late_check_equity'] += q * (street in ('turn', 'river'))
                    v['heads_up_check_equity'] += q * (len(live) == 2)
                if aggressive:
                    v['partner_raise_equity_cost_bb'] += q * amount_bb * to_partner
                    v['multiway_raise_low_equity_logcost'] += (1 - q) * math.log1p(amount_bb) * (len(live) >= 3)
                prior_equity[player] = q
        if aggressive:
            previous = player
        if kind == 'fold':
            live.remove(player)
    a, b = values[first], values[second]
    result = [a[n] + b[n] for n in NAMES] + [max(a[n], b[n]) for n in NAMES]
    if not all(math.isfinite(v) and v >= 0 for v in result):
        raise ValueError('Invalid equity feature')
    return result
