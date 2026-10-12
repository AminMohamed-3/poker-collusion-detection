"""Synthetic mechanical tests for ordered_surprise_features; no project data."""
from __future__ import annotations

from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "src"))
from ordered_surprise_features import FEATURES32, aggregate_hand  # noqa: E402


def vector(feature_set: dict[str, float]) -> np.ndarray:
    return np.array([feature_set[name] for name in FEATURES32], dtype=np.float64)


class OrderedSurpriseFeaturesTest(unittest.TestCase):
    def test_one_edge_is_its_outer_product(self) -> None:
        previous = np.array([1.0, -2.0, 0.5, -1.5])
        current = np.array([-3.0, 2.0, 4.0, -0.25])
        result = aggregate_hand([0, 1], [0, 0], [4, 9], [previous, current])
        expected = np.outer(previous, current)
        expected_abs = np.abs(expected)
        np.testing.assert_allclose(vector(result["ordered"]), np.r_[expected.ravel(), expected_abs.ravel()])
        np.testing.assert_allclose(vector(result["marginal_control"]), vector(result["ordered"]))

    def test_zero_edges_for_same_actor_and_street_boundary(self) -> None:
        residuals = np.arange(12, dtype=float).reshape(3, 4) / 10
        same_actor = aggregate_hand([0, 0, 0], [0, 0, 0], [1, 2, 3], residuals)
        boundary = aggregate_hand([0, 1], [0, 1], [1, 2], residuals[:2])
        omitted_street = aggregate_hand([0, 1], [0, 2], [1, 2], residuals[:2])
        for result in (same_actor, boundary, omitted_street):
            np.testing.assert_array_equal(vector(result["ordered"]), np.zeros(32))
            np.testing.assert_array_equal(vector(result["marginal_control"]), np.zeros(32))

    def test_actor_flip_and_input_order_do_not_change_features(self) -> None:
        directions = np.array([0, 1, 0, 1])
        streets = np.zeros(4, dtype=int)
        actions = np.array([10, 20, 30, 40])
        residuals = np.array([[1, 2, 3, 4], [-2, 1, 0.5, 3], [4, -1, 2, 0], [3, 2, -4, 1]], dtype=float)
        base = aggregate_hand(directions, streets, actions, residuals)
        flipped = aggregate_hand(1 - directions, streets, actions, residuals)
        shuffled = aggregate_hand(directions[[2, 0, 3, 1]], streets[[2, 0, 3, 1]], actions[[2, 0, 3, 1]], residuals[[2, 0, 3, 1]])
        for arm in ("ordered", "marginal_control"):
            np.testing.assert_allclose(vector(base[arm]), vector(flipped[arm]))
            np.testing.assert_allclose(vector(base[arm]), vector(shuffled[arm]))

    def test_invalid_duplicate_action_or_street_return_fails(self) -> None:
        residuals = np.ones((3, 4))
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_hand([0, 1, 0], [0, 0, 0], [5, 5, 6], residuals)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            aggregate_hand([0, 1, 0], [0, 1, 0], [5, 6, 7], residuals)

    def test_enumerated_within_actor_permutations_equal_control(self) -> None:
        directions = np.array([0, 1, 0, 1])
        streets = np.zeros(4, dtype=int)
        actions = np.array([1, 2, 3, 4])
        residuals = np.array([[1, -2, 0.5, 3], [2, 1, -3, 4], [-4, 2, 1, -1], [3, -2, 4, 0.25]], dtype=float)
        expected = aggregate_hand(directions, streets, actions, residuals)["marginal_control"]
        ordered_draws = []
        for first in permutations((0, 2)):
            for second in permutations((1, 3)):
                draw = np.empty_like(residuals)
                draw[[0, 2]] = residuals[list(first)]
                draw[[1, 3]] = residuals[list(second)]
                ordered_draws.append(vector(aggregate_hand(directions, streets, actions, draw)["ordered"]))
        np.testing.assert_allclose(np.mean(ordered_draws, axis=0), vector(expected), rtol=0, atol=1e-12)

    def test_time_scrambling_preserves_control_but_changes_ordered(self) -> None:
        directions = np.array([0, 1, 0, 1])
        streets = np.zeros(4, dtype=int)
        actions = np.array([1, 2, 3, 4])
        residuals = np.array([[1, 0, 0, 0], [2, 0, 0, 0], [7, 0, 0, 0], [11, 0, 0, 0]], dtype=float)
        scrambled = residuals.copy()
        scrambled[[0, 2]] = scrambled[[2, 0]]
        original = aggregate_hand(directions, streets, actions, residuals)
        changed = aggregate_hand(directions, streets, actions, scrambled)
        np.testing.assert_allclose(vector(original["marginal_control"]), vector(changed["marginal_control"]))
        self.assertFalse(np.allclose(vector(original["ordered"]), vector(changed["ordered"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)

