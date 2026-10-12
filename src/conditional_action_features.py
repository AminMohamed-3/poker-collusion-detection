"""Pre-action public/own-card features and partner-card responses for private research."""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from functools import lru_cache
from advanced_features import CARDS, preflop_strength
from phevaluator import evaluate_cards

STREETS = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
BOARD_LENGTH = [0, 3, 4, 5]
ACTION_NAMES = ['fold', 'check', 'call', 'raise']
INPUTS = [
    'street_index', 'own_strength', 'own_high_rank', 'own_low_rank', 'own_pair', 'own_suited',
    'own_flush_count', 'own_straight_window', 'board_high_rank', 'board_pairs', 'board_flush_count',
    'position_from_button', 'partner_position_from_button', 'price', 'log_pot_bb', 'log_stack_bb',
    'players_active', 'prior_actions', 'own_prior_actions', 'own_prior_aggression',
    'own_street_aggression', 'log_own_prior_amount_bb', 'partner_prior_actions',
    'partner_prior_aggression', 'partner_street_aggression', 'partner_last_action',
    'log_partner_prior_amount_bb', 'outsider_prior_aggression', 'facing_partner',
    *['own_style_' + n for n in ACTION_NAMES], *['partner_style_' + n for n in ACTION_NAMES],
]


def action_class(action):
    name = action['action']
    if name == 'fold': return 0
    if name == 'check': return 1
    if name == 'call' or (name == 'all_in' and action['amount'] <= action['to_call']): return 2
    if name in ['bet', 'raise', 'all_in']: return 3
    raise ValueError('Unexpected voluntary action: ' + name)


@lru_cache(maxsize=100_000)
def card_features(cards, board):
    c1, c2 = cards
    rank1, rank2 = c1 // 4 + 2, c2 // 4 + 2
    rank = preflop_strength(c1, c2) if not board else (7463 - evaluate_cards(c1, c2, *board)) / 7462
    suits = Counter(c % 4 for c in (*cards, *board))
    ranks = {c // 4 + 2 for c in (*cards, *board)}
    if 14 in ranks: ranks.add(1)
    straight = max(sum(r in ranks for r in range(start, start + 5)) for start in range(1, 11))
    return [rank, max(rank1, rank2), min(rank1, rank2), int(rank1 == rank2),
            int(c1 % 4 == c2 % 4), max(suits.values()), straight]


def extract_hand(hand, seats, actions, pairs, styles):
    """Each row uses state before its action; no later board or action enters INPUTS.

    styles is public, leave-current-hand-out behavior from other observed hands.
    It is supplied separately to keep current-hand future actions out of the reference.
    pairs entries are (pair_id, player_1, player_2). IDs are output/grouping keys only.
    """
    board_all = tuple(CARDS[c] for c in (hand['board_cards'] or '').split())
    seat_map = {s['player_id']: s for s in seats}
    cards = {p: tuple(sorted((CARDS[s['hole_card_1']], CARDS[s['hole_card_2']]))) for p, s in seat_map.items()}
    eligible = defaultdict(list)
    for pair, first, second in pairs:
        eligible[first].append((pair, second, 0))
        eligible[second].append((pair, first, 1))
    state = {p: {'folded': False, 'n': 0, 'aggression': 0, 'amount': 0., 'last': -1,
                 'street_aggression': 0} for p in seat_map}
    previous_aggressor = None
    previous_street = None
    blind = hand['big_blind']
    assert blind > 0
    rows = []
    for previous_count, action in enumerate(sorted(actions, key=lambda a: a['action_no'])):
        street = STREETS[action['street']]
        if street != previous_street:
            previous_aggressor = None
            for s in state.values(): s['street_aggression'] = 0
            previous_street = street
        board = board_all[:BOARD_LENGTH[street]]
        assert len(board) == BOARD_LENGTH[street]
        actor = action['player_id']; own = state[actor]
        assert not own['folded']
        kind = action_class(action)
        for pair, partner, direction in eligible.get(actor, []):
            other = state[partner]
            if other['folded']: continue
            own_card = card_features(cards[actor], board)
            partner_card = card_features(cards[partner], board)
            board_ranks = Counter(c // 4 for c in board)
            board_suits = Counter(c % 4 for c in board)
            price = action['to_call'] / max(1, action['pot_before'] + action['to_call'])
            feature = [street, *own_card,
                max((c // 4 + 2 for c in board), default=0), sum(n - 1 for n in board_ranks.values()),
                max(board_suits.values(), default=0),
                (seat_map[actor]['seat_no'] - hand['button_seat']) % 6,
                (seat_map[partner]['seat_no'] - hand['button_seat']) % 6,
                price, math.log1p(action['pot_before'] / blind), math.log1p(action['stack_before'] / blind),
                action['players_active'], previous_count, own['n'], own['aggression'], own['street_aggression'],
                math.log1p(own['amount'] / blind), other['n'], other['aggression'], other['street_aggression'],
                other['last'], math.log1p(other['amount'] / blind),
                sum(s['aggression'] for p, s in state.items() if p not in [actor, partner]),
                int(previous_aggressor == partner), *styles.get(actor, [.25] * 4), *styles.get(partner, [.25] * 4)]
            assert len(feature) == len(INPUTS) and all(math.isfinite(v) for v in feature)
            rows.append({'pair_id': pair, 'hand_id': hand['hand_id'], 'table_id': hand['table_id'],
                'direction': direction, 'action_no': action['action_no'], 'action_class': kind,
                'partner_strength': partner_card[0], 'observed_log_amount_bb': math.log1p(action['amount'] / blind),
                **dict(zip(INPUTS, feature))})
        own['n'] += 1; own['amount'] += action['amount']; own['last'] = kind
        if kind == 0: own['folded'] = True
        if kind == 3:
            own['aggression'] += 1; own['street_aggression'] += 1
            previous_aggressor = actor
    return rows
