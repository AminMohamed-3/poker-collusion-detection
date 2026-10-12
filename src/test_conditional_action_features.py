from copy import deepcopy
from conditional_action_features import extract_hand, INPUTS


def fixture():
    hand = {'hand_id': 'h', 'table_id': 't', 'board_cards': '2c 7d Jh Qs Ac', 'button_seat': 6, 'big_blind': 2}
    seats = [{'player_id': 'a', 'seat_no': 1, 'hole_card_1': 'Kd', 'hole_card_2': 'Kh'},
             {'player_id': 'b', 'seat_no': 2, 'hole_card_1': '8s', 'hole_card_2': '9s'}]
    actions = [{'player_id': 'a', 'action_no': 1, 'street': 'flop', 'action': 'bet', 'amount': 4, 'to_call': 0, 'pot_before': 10, 'stack_before': 100, 'players_active': 2},
               {'player_id': 'b', 'action_no': 2, 'street': 'flop', 'action': 'call', 'amount': 4, 'to_call': 4, 'pot_before': 14, 'stack_before': 100, 'players_active': 2}]
    return hand, seats, actions


def test_future_board_and_actions_do_not_change_earlier_input():
    hand, seats, actions = fixture(); before = extract_hand(hand, seats, actions, [('p', 'a', 'b')], {})[0]
    hand['board_cards'] = '2c 7d Jh 3s 4c'; actions[1]['amount'] = 30; actions[1]['action'] = 'raise'
    after = extract_hand(hand, seats, actions, [('p', 'a', 'b')], {})[0]
    assert {c: before[c] for c in INPUTS} == {c: after[c] for c in INPUTS}


def test_partner_private_cards_are_not_action_model_inputs():
    hand, seats, actions = fixture(); before = extract_hand(hand, seats, actions, [('p', 'a', 'b')], {})[0]
    seats[1].update(hole_card_1='Jd', hole_card_2='Js')
    after = extract_hand(hand, seats, actions, [('p', 'a', 'b')], {})[0]
    assert before['partner_strength'] != after['partner_strength']
    assert {c: before[c] for c in INPUTS} == {c: after[c] for c in INPUTS}


def test_previous_aggression_and_price_are_attributed_before_action():
    hand, seats, actions = fixture(); rows = extract_hand(hand, seats, actions, [('p', 'a', 'b')], {})
    assert rows[0]['partner_prior_aggression'] == 0 and rows[0]['facing_partner'] == 0
    assert rows[1]['partner_prior_aggression'] == 1 and rows[1]['facing_partner'] == 1
    assert rows[1]['price'] == 4/18 and rows[1]['action_class'] == 2


def test_pair_order_and_identifiers_do_not_change_predictors():
    hand, seats, actions = fixture(); a = extract_hand(hand, seats, actions, [('p', 'a', 'b')], {})
    b = extract_hand(hand, seats, actions, [('different', 'b', 'a')], {})
    assert [[r[c] for c in INPUTS] for r in a] == [[r[c] for c in INPUTS] for r in b]
    assert [r['direction'] for r in a] == [1-r['direction'] for r in b]
