import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from sequence_evidence_features import extract_pair_hand, NAMES


def action(n, player, kind, amount, call, pot, street='preflop'):
    return dict(action_no=n, player_id=player, action=kind, amount=amount, to_call=call, pot_before=pot, street=street)


def case():
    return {'A': (8., -8.), 'B': (18., 16.), 'C': (8., -8.)}, [
        action(1, 'A', 'raise', 8, 2, 3), action(2, 'C', 'call', 8, 8, 11),
        action(3, 'B', 'raise', 18, 8, 19), action(4, 'C', 'fold', 0, 10, 37),
        action(5, 'A', 'fold', 0, 10, 37)]


def test_coordinated_sequence_hand_calculation():
    seats, actions = case()
    result = dict(zip(NAMES, extract_pair_hand('A', 'B', 2, seats, actions)))
    assert result['outsider_call_to_pair_bb'] == 4
    assert result['outsider_fold_after_both_raise_n'] == 1
    assert result['outsider_fold_after_both_raise_invested_bb'] == 4
    assert result['pair_fold_after_both_raise_invested_bb'] == 4
    assert result['both_pair_raise_streets'] == 1
    assert result['pair_profit_after_both_raise_bb'] == 4
    assert result['pair_reraise_multiway_n'] == 1
    assert result['max_outsider_fold_odds_after_both_raise'] == 10/47


def test_pair_and_input_order_invariance():
    seats, actions = case()
    np.testing.assert_array_equal(extract_pair_hand('A', 'B', 2, seats, actions),
                                  extract_pair_hand('B', 'A', 2, seats, actions[::-1]))


def test_all_in_call_does_not_count_as_raise():
    seats = {'A': (10, 0), 'B': (10, 0), 'C': (0, 0)}
    actions = [action(1, 'A', 'raise', 10, 2, 3), action(2, 'B', 'all_in', 10, 10, 13), action(3, 'C', 'fold', 0, 10, 23)]
    result = dict(zip(NAMES, extract_pair_hand('A', 'B', 2, seats, actions)))
    assert result['both_pair_raise_streets'] == 0
    assert result['outsider_fold_after_both_raise_n'] == 0


def test_previous_street_aggression_is_not_carried_forward():
    seats = {'A': (4, 0), 'B': (4, 0), 'C': (4, 0)}
    actions = [action(1, 'A', 'raise', 4, 2, 3), action(2, 'B', 'call', 4, 4, 7),
               action(3, 'C', 'call', 4, 4, 11), action(4, 'C', 'check', 0, 0, 15, 'flop')]
    result = dict(zip(NAMES, extract_pair_hand('A', 'B', 2, seats, actions)))
    assert result['outsider_call_to_pair_bb'] == 2
    assert result['outsider_fold_to_pair_n'] == 0


def test_mutual_checks_require_both_pair_members_and_no_outsider():
    seats = {'A': (0, 0), 'B': (0, 0), 'C': (0, 0)}
    actions = [action(1, 'C', 'fold', 0, 2, 3), action(2, 'A', 'check', 0, 0, 3, 'flop'),
               action(3, 'B', 'check', 0, 0, 3, 'flop')]
    result = dict(zip(NAMES, extract_pair_hand('A', 'B', 2, seats, actions)))
    assert result['heads_up_mutual_check_streets'] == 1


def test_duplicate_actions_and_actions_after_fold_fail():
    seats, actions = case()
    with pytest.raises(ValueError, match='Duplicate'):
        extract_pair_hand('A', 'B', 2, seats, actions+[actions[0]])
    with pytest.raises(ValueError, match='after folding'):
        extract_pair_hand('A', 'B', 2, seats, actions+[action(6, 'A', 'check', 0, 0, 37)])
