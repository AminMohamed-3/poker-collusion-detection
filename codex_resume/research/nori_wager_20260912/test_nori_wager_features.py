from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nori_wager_feature_build import aggregate  # noqa: E402


class NoriWagerFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.keys = pl.DataFrame({"pair_id": ["p1", "p2", "p3"], "hand_id": ["h1", "h2", "h3"]})
        self.actions = pl.DataFrame({
            "pair_id": ["p1", "p1", "p2"], "hand_id": ["h1", "h1", "h2"],
            "action_no": [1, 2, 1], "direction": [0, 1, 0],
        })
        self.actual = np.array([2.0, 0.2, 1.5])
        self.upper = np.array([2.0, 0.8, 2.0])
        self.quantiles = np.array([[0.5, 1.0, 1.5], [1.5, 0.9, 2.0], [2.0, 0.9, 1.0]])
        self.strength = np.array([0.2, 0.8, 0.6])
        self.partner = np.array([1.0, 0.0, 1.0])
        self.active = np.array([3.0, 2.0, 4.0])

    def test_known_values_clipping_crossing_and_empty_hand(self):
        out = aggregate(self.keys, self.actions, self.actual, self.upper, self.quantiles,
                        self.strength, self.partner, self.active)
        self.assertEqual(out.shape, (3, 26))
        self.assertEqual(out.dtype, np.float32)
        # h1 direction 0: delta=1, standardized=1, below=0, above=.5,
        # outside=1, width=1, multiway lower=.4, partner upper=0.
        # h1 direction 1: sorted/clipped quantiles [.8,.8,.8], delta=-.6,
        # standardized=-5, below=.6, above=0, outside=1, width=0.
        np.testing.assert_allclose(out[0, :8], [0.2, -2.0, 0.3, 0.25, 1.0, 0.5, 0.2, 0.0], atol=1e-7)
        np.testing.assert_allclose(out[0, 8:16], [1.6, 6.0, 0.6, 0.5, 0.0, 1.0, 0.4, 0.0], atol=1e-7)
        np.testing.assert_allclose(out[2], np.zeros(26, dtype=np.float32), atol=0.0)
        # h2 has no direction-1 events, so empty-side zeros affect means/maxima.
        self.assertAlmostEqual(float(out[1, 24]), 1.0 / 24.0)
        self.assertAlmostEqual(float(out[1, 25]), 1.0 / 24.0)

    def test_pair_direction_swap_is_exact(self):
        baseline = aggregate(self.keys, self.actions, self.actual, self.upper, self.quantiles,
                             self.strength, self.partner, self.active)
        swapped = self.actions.with_columns((1 - pl.col("direction")).alias("direction"))
        actual = aggregate(self.keys, swapped, self.actual, self.upper, self.quantiles,
                           self.strength, self.partner, self.active)
        np.testing.assert_array_equal(baseline, actual)

    def test_invalid_target_bound_and_direction_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "actual and upper"):
            aggregate(self.keys, self.actions, np.array([2.1, .2, 1.5]), self.upper, self.quantiles,
                      self.strength, self.partner, self.active)
        bad_actions = self.actions.with_columns(pl.Series("direction", [0, 2, 0]))
        with self.assertRaisesRegex(ValueError, "direction"):
            aggregate(self.keys, bad_actions, self.actual, self.upper, self.quantiles,
                      self.strength, self.partner, self.active)


if __name__ == "__main__":
    unittest.main(verbosity=2)

