"""Synthetic mechanical fixtures for contestable_state_features; no dataset access."""
import importlib.util
from pathlib import Path
import math

SOURCE = Path(__file__).resolve().parents[2] / 'src' / 'contestable_state_features.py'
spec = importlib.util.spec_from_file_location('contestable_state_features', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def seats(starts, totals, folded=()):
    players = ('a', 'x', 'c', 'b', 'd', 'e')
    return [{'player_id': player, 'seat_no': seat, 'starting_stack': starts[player],
             'total_contribution': totals[player], 'folded': player in folded}
            for seat, player in enumerate(players)]


def event(no, actor, action, amount, pot, stack, call):
    return {'action_no': no, 'player_id': actor, 'action': action, 'amount': amount,
            'pot_before': pot, 'stack_before': stack, 'to_call': call}


def hand(small=1, big=2):
    return {'hand_id': 'synthetic', 'button_seat': 2, 'small_blind': small,
            'big_blind': big, 'players_dealt': 6}


def scale_hand_case(case, factor):
    h, s, actions, pair = case
    scaled_h = {**h, 'small_blind': h['small_blind'] * factor, 'big_blind': h['big_blind'] * factor}
    money = ('starting_stack', 'total_contribution')
    scaled_s = [{**row, **{field: row[field] * factor for field in money}} for row in s]
    fields = ('amount', 'pot_before', 'stack_before', 'to_call')
    scaled_actions = [{**row, **{field: row[field] * factor for field in fields}} for row in actions]
    return scaled_h, scaled_s, scaled_actions, pair


def blind_fixture():
    start = {'a': 20, 'x': 20, 'c': 20, 'b': 20, 'd': 20, 'e': 20}
    total = {'a': 2, 'x': 0, 'c': 0, 'b': 2, 'd': 2, 'e': 0}
    actions = [event(0, 'a', 'call', 2, 3, 20, 2), event(1, 'x', 'fold', 0, 5, 20, 2),
               event(2, 'b', 'call', 1, 5, 19, 1), event(3, 'd', 'check', 0, 6, 18, 0)]
    return hand(), seats(start, total, ('x',)), actions, ('a', 'b')


def fold_allin_fixture():
    start = {'a': 20, 'x': 20, 'c': 20, 'b': 10, 'd': 20, 'e': 20}
    total = {'a': 10, 'x': 0, 'c': 0, 'b': 10, 'd': 2, 'e': 0}
    actions = [event(0, 'a', 'raise', 10, 3, 20, 2), event(1, 'b', 'all_in', 9, 13, 9, 9),
               event(2, 'd', 'fold', 0, 22, 18, 8), event(3, 'a', 'fold', 0, 22, 10, 0)]
    return hand(), seats(start, total, ('a', 'd')), actions, ('a', 'b')


def unequal_cap_fixture():
    start = {'a': 5, 'x': 20, 'c': 20, 'b': 20, 'd': 20, 'e': 20}
    total = {'a': 5, 'x': 0, 'c': 0, 'b': 11, 'd': 2, 'e': 0}
    actions = [event(0, 'b', 'raise', 10, 3, 19, 1), event(1, 'a', 'all_in', 5, 13, 5, 11),
               event(2, 'd', 'fold', 0, 18, 18, 9)]
    return hand(), seats(start, total, ('d',)), actions, ('a', 'b')


def blind_allin_fixture():
    start = {'a': 20, 'x': 20, 'c': 2, 'b': 1, 'd': 20, 'e': 20}
    total = {'a': 0, 'x': 0, 'c': 0, 'b': 1, 'd': 2, 'e': 0}
    actions = [event(0, 'a', 'fold', 0, 3, 20, 2)]
    return hand(), seats(start, total, ('a',)), actions, ('a', 'b')


def test_blinds_and_swap():
    h, s, actions, pair = blind_fixture()
    feature, parity = module.state_features(h, s, actions, pair)
    swapped, _ = module.state_features(h, s, actions, pair[::-1])
    assert feature == swapped
    assert parity['pot_before_checks'] == 4 and parity['final_contribution_checks'] == 6
    assert feature['cs_decisions'] == 2


def test_fold_and_allin_are_distinct():
    h, s, actions, pair = fold_allin_fixture()
    feature, _ = module.state_features(h, s, actions, pair)
    assert feature['cs_fold_partner_allin_count'] == 1
    assert feature['cs_decisions_partner_allin'] == 1
    assert feature['cs_decisions_both_can_wager'] == 2
    assert feature['cs_call_partner_active_count'] == 1
    assert feature['cs_aggr_partner_active_count'] == 1


def test_unequal_stack_cap_and_scale_invariance():
    case = unequal_cap_fixture()
    feature, _ = module.state_features(*case)
    assert math.isclose(feature['cs_actor_unreachable_frac_max'], 1 / 3)
    scaled, _ = module.state_features(*scale_hand_case(case, 7))
    for name in module.FEATURES:
        assert math.isclose(feature[name], scaled[name], abs_tol=1e-12), name
    assert feature == module.state_features(case[0], case[1], case[2], case[3][::-1])[0]


def test_posted_blind_allin_is_preaction_state():
    h, s, actions, pair = blind_allin_fixture()
    feature, _ = module.state_features(h, s, actions, pair)
    assert feature['cs_decisions_partner_allin'] == 1
    assert feature['cs_fold_partner_allin_count'] == 1
    assert feature['cs_decisions_both_can_wager'] == 0


if __name__ == '__main__':
    test_blinds_and_swap()
    test_fold_and_allin_are_distinct()
    test_unequal_stack_cap_and_scale_invariance()
    test_posted_blind_allin_is_preaction_state()
    print('contestable state fixtures passed')

