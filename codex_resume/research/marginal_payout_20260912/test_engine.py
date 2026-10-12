"""Synthetic mechanical tests for marginal_payout_engine; no real data access."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from marginal_payout_engine import HandPayouts, MechanicalPayoutError
from phevaluator import Card


def ids(*cards):
    return [Card.to_id(card) for card in cards]


HOLES = {
    'a': ids('As', 'Ad'), 'b': ids('Kc', 'Kd'), 'c': ids('3c', '4c'),
    'd': ids('5c', '6c'), 'e': ids('8d', '8h'), 'f': ids('Ts', 'Th'),
}
BOARD = ids('2c', '7d', '9h', 'Jc', 'Qs')


def close(left, right, tol=1e-10):
    assert math.isclose(left, right, rel_tol=0.0, abs_tol=tol), (left, right)


def conserved(result, contributions):
    close(math.fsum(result['payouts'].values()), math.fsum(contributions.values()))


def test_known_river_single_winner_and_unequal_stakes():
    engine = HandPayouts(HOLES, BOARD)
    contribution = {'a': 10, 'b': 50, 'c': 50, 'd': 100, 'e': 100, 'f': 10}
    result = engine.payouts(contribution, set(HOLES), 'river', 'single')
    assert result['runouts'] == 1
    close(result['payouts']['a'], 320.0)
    assert all(result['payouts'][p] == 0.0 for p in HOLES if p != 'a')
    conserved(result, contribution)


def test_folded_dead_money_remains_but_cannot_win():
    engine = HandPayouts(HOLES, BOARD)
    contribution = {p: 10 for p in HOLES}
    result = engine.payouts(contribution, set(HOLES) - {'a'}, 'river', 'single')
    assert result['payouts']['a'] == 0.0
    close(result['payouts']['b'], 60.0)
    conserved(result, contribution)


def test_layered_all_in_eligibility_excludes_short_stack_from_upper_layer():
    engine = HandPayouts(HOLES, BOARD)
    contribution = {'a': 20, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100}
    result = engine.payouts(contribution, {'a', 'b'}, 'river', 'layered')
    close(result['payouts']['a'], 120.0)
    close(result['payouts']['b'], 400.0)
    assert result['refunds']['a'] == result['refunds']['b'] == 0.0
    conserved(result, contribution)


def test_single_contributor_layer_is_reported_as_refund():
    engine = HandPayouts(HOLES, BOARD)
    contribution = {'a': 100, 'b': 20, 'c': 20, 'd': 20, 'e': 20, 'f': 20}
    result = engine.payouts(contribution, {'a', 'b'}, 'river', 'layered')
    close(result['refunds']['a'], 80.0)
    assert sum(layer['refund'] for layer in result['layers']) == 1
    conserved(result, contribution)


def test_three_way_tie_splits_equally():
    holes = {
        'a': ids('2c', '3d'), 'b': ids('4c', '5d'), 'c': ids('6c', '7d'),
        'd': ids('8c', '9d'), 'e': ids('Tc', '2d'), 'f': ids('3c', '4d'),
    }
    board = ids('As', 'Ks', 'Qs', 'Js', 'Ts')
    engine = HandPayouts(holes, board)
    contribution = {'a': 10, 'b': 10, 'c': 10, 'd': 0, 'e': 0, 'f': 0}
    result = engine.payouts(contribution, {'a', 'b', 'c'}, 'river', 'single')
    for player in 'abc':
        close(result['payouts'][player], 10.0)
    conserved(result, contribution)


def test_player_renaming_and_mapping_order_do_not_change_card_keyed_results():
    base = HandPayouts(HOLES, BOARD)
    contribution = {'a': 20, 'b': 50, 'c': 50, 'd': 50, 'e': 50, 'f': 50}
    base_result = base.payouts(contribution, {'a', 'b', 'c'}, 'river', 'layered')
    names = {'a': 'z', 'b': 'm', 'c': 'x', 'd': 'q', 'e': 'n', 'f': 'r'}
    renamed_holes = {names[player]: HOLES[player] for player in reversed(tuple(HOLES))}
    renamed = HandPayouts(renamed_holes, BOARD)
    renamed_result = renamed.payouts({names[p]: v for p, v in contribution.items()}, {names[p] for p in {'a', 'b', 'c'}}, 'river', 'layered')
    base_by_cards = {tuple(sorted(HOLES[p])): v for p, v in base_result['payouts'].items()}
    renamed_by_cards = {tuple(sorted(renamed_holes[p])): v for p, v in renamed_result['payouts'].items()}
    assert base_by_cards == renamed_by_cards


def test_money_scaling_and_equal_stakes_mode_equivalence():
    engine = HandPayouts(HOLES, BOARD)
    equal = {p: 17.5 for p in HOLES}
    single = engine.payouts(equal, set(HOLES), 'river', 'single')
    layered = engine.payouts(equal, set(HOLES), 'river', 'layered')
    assert single['payouts'] == layered['payouts']
    scaled = {p: value * 7 for p, value in equal.items()}
    result = engine.payouts(scaled, set(HOLES), 'river', 'layered')
    for player in HOLES:
        close(result['payouts'][player], layered['payouts'][player] * 7)
    conserved(result, scaled)


def test_future_board_suffix_is_ignored_at_flop():
    first = HandPayouts(HOLES, ids('2c', '7d', '9h', 'Jc', 'Qs'))
    second = HandPayouts(HOLES, ids('2c', '7d', '9h', '5h', '6h'))
    contribution = {p: 10 for p in HOLES}
    left = first.payouts(contribution, set(HOLES), 'flop', 'single')
    right = second.payouts(contribution, set(HOLES), 'flop', 'single')
    assert left['runouts'] == right['runouts'] == 666
    assert left['payouts'] == right['payouts']


def test_random_valid_states_conserve_in_both_modes():
    deck = list(range(52))
    rng = random.Random(917)
    for _ in range(24):
        rng.shuffle(deck)
        holes = {f'p{i}': [deck[2 * i], deck[2 * i + 1]] for i in range(6)}
        board = deck[12:17]
        contribution = {player: float(rng.randrange(0, 121)) for player in holes}
        engine = HandPayouts(holes, board)
        for mode in ('single', 'layered'):
            result = engine.payouts(contribution, set(holes), 'river', mode)
            conserved(result, contribution)


def test_invalid_empty_live_and_empty_eligible_layer_are_explicit():
    engine = HandPayouts(HOLES, BOARD)
    try:
        engine.payouts({p: 10 for p in HOLES}, set(), 'river', 'single')
    except MechanicalPayoutError:
        pass
    else:
        raise AssertionError('empty live set must fail')
    contribution = {'a': 10, 'b': 20, 'c': 20, 'd': 0, 'e': 0, 'f': 0}
    try:
        engine.payouts(contribution, {'a'}, 'river', 'layered')
    except MechanicalPayoutError:
        pass
    else:
        raise AssertionError('uncontested dead layer must fail')



def test_preflop_short_board_determinism_and_renaming_invariance():
    contribution = {'a': 20, 'b': 50, 'c': 50, 'd': 50, 'e': 50, 'f': 50}
    live = {'a', 'b', 'c'}
    empty_board = HandPayouts(HOLES, [])
    with_future_suffix = HandPayouts(HOLES, BOARD)
    first = empty_board.payouts(contribution, live, 'preflop', 'layered')
    repeat = empty_board.payouts(contribution, live, 'preflop', 'layered')
    future = with_future_suffix.payouts(contribution, live, 'preflop', 'layered')
    assert first['runouts'] == repeat['runouts'] == future['runouts'] == 4096
    assert first['payouts'] == repeat['payouts'] == future['payouts']
    names = {'a': 'z', 'b': 'm', 'c': 'x', 'd': 'q', 'e': 'n', 'f': 'r'}
    renamed_holes = {names[player]: HOLES[player] for player in reversed(tuple(HOLES))}
    renamed = HandPayouts(renamed_holes, [])
    renamed_result = renamed.payouts({names[p]: v for p, v in contribution.items()}, {names[p] for p in live}, 'preflop', 'layered')
    by_cards = {tuple(sorted(HOLES[p])): value for p, value in first['payouts'].items()}
    renamed_by_cards = {tuple(sorted(renamed_holes[p])): value for p, value in renamed_result['payouts'].items()}
    assert by_cards == renamed_by_cards


def test_partial_flop_board_is_valid_and_unavailable_turn_fails():
    engine = HandPayouts(HOLES, BOARD[:3])
    contribution = {p: 10 for p in HOLES}
    result = engine.payouts(contribution, set(HOLES), 'flop', 'single')
    assert result['runouts'] == 666
    try:
        engine.payouts(contribution, set(HOLES), 'turn', 'single')
    except MechanicalPayoutError:
        pass
    else:
        raise AssertionError('turn must fail when only a flop prefix is disclosed')
if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
    print(f'{len(tests)} synthetic payout-engine tests passed')

