import pytest
from equity_evidence_features import CARDS, FEATURES, heads_up_equity, extract_pair_hand


def cards(text):
    return tuple(CARDS[c] for c in text.split())


def test_river_known_win_and_board_tie():
    assert heads_up_equity(cards('As Ad'), cards('Kc Kd'), cards('2c 3h 7s 9d Jc')) == 1
    assert heads_up_equity(cards('2s 3d'), cards('4c 5d'), cards('Ah Kh Qh Jh Th')) == .5


def test_exact_turn_equity_and_pair_order():
    first, second, board = cards('As Ad'), cards('Kc Kd'), cards('2c 3h 7s 9d')
    q = heads_up_equity(first, second, board)
    assert q == pytest.approx(42 / 44)
    assert heads_up_equity(second, first, board) == pytest.approx(1 - q)
    assert heads_up_equity(first[::-1], second[::-1], board[::-1]) == q
    assert heads_up_equity(first, second, ()) + heads_up_equity(second, first, ()) == 1


def event(n, player, action, amount, cost, pot, street='river'):
    return {'action_no': n, 'player_id': player, 'action': action, 'amount': amount,
            'to_call': cost, 'pot_before': pot, 'street': street, 'stack_before': 100}


def test_call_price_pair_symmetry_and_aggressor():
    seats = {'a': cards('As Ad'), 'b': cards('Kc Kd'), 'c': cards('4c 5d')}
    board = cards('2c 3h 7s 9d Jc')
    actions = [event(1, 'a', 'bet', 10, 0, 20), event(2, 'b', 'call', 10, 10, 30)]
    actual = dict(zip(FEATURES, extract_pair_hand('a', 'b', 2, seats, board, actions)))
    assert actual['eq_partner_call_deficit_bb_sum'] == 5
    assert actual['eq_partner_call_surplus_bb_sum'] == 0
    assert list(actual.values()) == extract_pair_hand('b', 'a', 2, seats, board, actions[::-1])
    outsider = [event(1, 'c', 'bet', 10, 0, 20), event(2, 'b', 'call', 10, 10, 30)]
    other = dict(zip(FEATURES, extract_pair_hand('a', 'b', 2, seats, board, outsider)))
    assert other['eq_partner_call_deficit_bb_sum'] == 0


def test_flop_ignores_later_board_and_folded_partner():
    seats = {'a': cards('As Ad'), 'b': cards('Kc Kd'), 'c': cards('4c 5d')}
    actions = [event(1, 'a', 'bet', 10, 0, 20, 'flop'), event(2, 'b', 'call', 10, 10, 30, 'flop')]
    a = extract_pair_hand('a', 'b', 2, seats, cards('2c 3h 7s 9d Jc'), actions)
    b = extract_pair_hand('a', 'b', 2, seats, cards('2c 3h 7s Kh 8c'), actions)
    assert a == b
    folded = [event(1, 'a', 'fold', 0, 0, 20), event(2, 'b', 'check', 0, 0, 20)]
    assert extract_pair_hand('a', 'b', 2, seats, cards('2c 3h 7s 9d Jc'), folded) == [0.] * len(FEATURES)
