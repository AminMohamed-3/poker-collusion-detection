from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from within_hand_prefix import MAX_PREFIX_EVENTS, TOKEN_DIM, extract_prefix  # noqa: E402


def seat_rows(prefix: str = "p"):
    return [
        {"player_id": prefix + "A", "seat_no": 0, "hole_card_1": "As", "terminal_net": 7},
        {"player_id": prefix + "B", "seat_no": 2, "hole_card_1": "Kd", "terminal_net": -3},
        {"player_id": prefix + "C", "seat_no": 4, "hole_card_1": "Qh", "terminal_net": -4},
    ]


def event(number, player, action, street="preflop", amount=0.0, to_call=0.0,
          pot_before=10.0, stack_before=90.0, players_active=3):
    return {
        "action_no": number,
        "player_id": player,
        "action": action,
        "street": street,
        "amount": amount,
        "to_call": to_call,
        "pot_before": pot_before,
        "stack_before": stack_before,
        "players_active": players_active,
        "winner": "irrelevant-terminal-field",
        "private_cards": "As Kd",
    }


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.hand = {
            "hand_id": "public-id-only",
            "big_blind": 2.0,
            "board_cards": "2c 2d Th Qs Ac",
            "outcome": "masked",
        }
        self.seats = seat_rows()
        self.actions = [
            event(1, "pA", "raise", amount=6, pot_before=3, stack_before=98),
            event(2, "pC", "call", amount=6, to_call=6, pot_before=9, stack_before=94),
            event(3, "pB", "all_in", amount=6, to_call=6, pot_before=15, stack_before=90),
            event(4, "pA", "check", street="flop", pot_before=21, stack_before=92),
            event(5, "pB", "raise", street="flop", amount=12, pot_before=21, stack_before=84),
        ]

    def extract(self, actions=None, hand=None, seats=None, focal="pA", partner="pB", target=4):
        return extract_prefix(
            self.hand if hand is None else hand,
            self.seats if seats is None else seats,
            self.actions if actions is None else actions,
            focal,
            partner,
            target,
        )

    def test_empty_and_short_prefix_are_right_padded_float32(self):
        empty, empty_length = self.extract(actions=[], target=0)
        self.assertEqual(empty.shape, (MAX_PREFIX_EVENTS, TOKEN_DIM))
        self.assertEqual(empty.dtype, np.float32)
        self.assertEqual(empty_length, 0)
        self.assertFalse(empty.any())

        short, short_length = self.extract(actions=self.actions[:2], target=3)
        self.assertEqual(short_length, 2)
        self.assertTrue(short[:2].any())
        self.assertFalse(short[2:].any())

    def test_latest_24_actions_and_action_number_order(self):
        actions = [
            event(number, "pC", "call", amount=float(number), to_call=float(number))
            for number in range(1, 27)
        ]
        output, length = self.extract(actions=list(reversed(actions)), target=27)
        self.assertEqual(length, 24)
        self.assertAlmostEqual(float(output[0, 9]), np.log1p(3.0 / 2.0) / 5.0)
        self.assertAlmostEqual(float(output[-1, 9]), np.log1p(26.0 / 2.0) / 5.0)

    def test_target_and_future_payload_mutations_cannot_affect_prefix(self):
        baseline, length = self.extract()
        changed = copy.deepcopy(self.actions)
        changed[3].update({"amount": -9, "action": "unrecognized", "street": "river"})
        changed[4].update({"pot_before": -5, "player_id": "missing", "action": "unrecognized"})
        future_board = dict(self.hand, board_cards="2c 2d Th XX YY")
        mutated, mutated_length = self.extract(actions=changed, hand=future_board)
        self.assertEqual(length, mutated_length)
        np.testing.assert_array_equal(baseline, mutated)

    def test_id_and_masked_private_terminal_mutations_do_not_change_encoding(self):
        base, length = self.extract()
        renamed_seats = seat_rows("x")
        renamed_actions = copy.deepcopy(self.actions)
        for row in renamed_actions:
            row["player_id"] = "x" + row["player_id"][1:]
            row["private_cards"] = "bad private data"
            row["winner"] = "different outcome"
        renamed_hand = dict(self.hand, hand_id="a-different-id", outcome="different")
        renamed, renamed_length = self.extract(
            actions=renamed_actions,
            hand=renamed_hand,
            seats=renamed_seats,
            focal="xA",
            partner="xB",
        )
        self.assertEqual(length, renamed_length)
        np.testing.assert_array_equal(base, renamed)

    def test_pair_role_swap_changes_directional_roles_and_relative_seats(self):
        base, length = self.extract()
        swapped, swapped_length = self.extract(focal="pB", partner="pA")
        self.assertEqual(length, swapped_length)
        self.assertEqual(tuple(base[0, 1:4]), (1.0, 0.0, 0.0))
        self.assertEqual(tuple(swapped[0, 1:4]), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(float(base[0, 14]), 0.0)
        self.assertAlmostEqual(float(swapped[0, 14]), 4.0 / 5.0)

    def test_common_money_and_blind_rescaling_is_invariant(self):
        base, length = self.extract()
        scaled_hand = dict(self.hand, big_blind=20.0)
        scaled_actions = copy.deepcopy(self.actions)
        for row in scaled_actions:
            for field in ("amount", "to_call", "pot_before", "stack_before"):
                row[field] *= 10.0
        scaled, scaled_length = self.extract(actions=scaled_actions, hand=scaled_hand)
        self.assertEqual(length, scaled_length)
        np.testing.assert_allclose(base, scaled, rtol=0.0, atol=1e-7)

    def test_abstract_token_order_follows_action_number(self):
        base, _ = self.extract()
        permuted, _ = self.extract(actions=[self.actions[2], self.actions[0], self.actions[1]])
        np.testing.assert_array_equal(base, permuted)
        reindexed = copy.deepcopy(self.actions[:3])
        reindexed[0]["action_no"], reindexed[1]["action_no"] = 2, 1
        changed, _ = self.extract(actions=reindexed)
        self.assertFalse(np.array_equal(base, changed))

    def test_invalid_prefix_order_unknown_cards_and_negative_money_rejected(self):
        bad_order = [
            event(1, "pA", "check", street="flop"),
            event(2, "pB", "check", street="preflop"),
        ]
        with self.assertRaisesRegex(ValueError, "earlier street"):
            self.extract(actions=bad_order, target=3)
        bad_board = [event(1, "pA", "check", street="flop")]
        with self.assertRaisesRegex(ValueError, "unknown board"):
            self.extract(actions=bad_board, hand=dict(self.hand, board_cards="2c 2d XX"), target=2)
        negative = [event(1, "pA", "call", amount=-1)]
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.extract(actions=negative, target=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
