"""Synthetic mechanics for the pure outsider card-strength aggregation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "src"))
from outsider_card_features import FEATURES, aggregate


def check_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_single_event_control():
    result = aggregate(
        [2], [1], ["fold"], [[0.1, 0.2, 0.3, 0.8]], [0.25], [2.0]
    )
    assert tuple(result["linked"]) == FEATURES
    check_close(result["linked"]["od_one_fold_strength_sum"], 0.6)
    check_close(result["linked"]["od_one_fold_ahead_sum"], 0.1)
    check_close(result["card_control"]["od_one_fold_strength_sum"], 2.0 * np.mean([0.1, 0.2, 0.3, 0.8]))
    check_close(result["card_control"]["od_one_fold_ahead_sum"], 2.0 * np.mean([0, 0, 0.05, 0.55]))


def test_shared_permutation_mean_of_max():
    result = aggregate(
        [0, 0], [1, 1], ["call", "call"],
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], [0.0, 0.0], [1.0, 1.0],
    )
    check_close(result["linked"]["od_one_call_strength_max"], 0.4)
    check_close(result["card_control"]["od_one_call_strength_max"], 0.35)
    assert result["card_control"]["od_one_call_strength_max"] != 0.25


def test_global_reassignment_control_and_linked_change():
    base = aggregate(
        [0, 2, 1], [1, 2, 1], ["fold", "aggr", "call"],
        [[0.1, 0.2, 0.6, 0.8], [0.2, 0.7, 0.3, 0.4], [0.6, 0.1, 0.5, 0.2]],
        [0.15, 0.4, 0.3], [1.0, 2.0, 0.5],
    )
    permuted = aggregate(
        [0, 2, 1], [1, 2, 1], ["fold", "aggr", "call"],
        [[0.8, 0.6, 0.2, 0.1], [0.4, 0.3, 0.7, 0.2], [0.2, 0.5, 0.1, 0.6]],
        [0.15, 0.4, 0.3], [1.0, 2.0, 0.5],
    )
    assert any(base["linked"][name] != permuted["linked"][name] for name in FEATURES)
    for name in FEATURES:
        check_close(base["card_control"][name], permuted["card_control"][name])


def test_row_order_and_category_isolation():
    values = ([0, 1, 2], [1, 2, 1], ["fold", "call", "aggr"],
              [[0.9, 0.2, 0.1, 0.3], [0.4, 0.8, 0.2, 0.1], [0.3, 0.5, 0.7, 0.2]],
              [0.4, 0.7, 0.1], [1.0, 3.0, 2.0])
    first = aggregate(*values)
    order = [2, 0, 1]
    second = aggregate(*[np.asarray(v)[order] for v in values])
    for arm in ("linked", "card_control"):
        for name in FEATURES:
            check_close(first[arm][name], second[arm][name])
        assert first[arm]["od_one_fold_strength_sum"] > 0
        assert first[arm]["od_both_fold_strength_sum"] == 0
        assert first[arm]["od_both_aggr_strength_sum"] == 0


def test_empty_and_invalid_inputs():
    empty = aggregate([], [], [], np.empty((0, 4)), [], [])
    for arm in ("linked", "card_control"):
        assert tuple(empty[arm]) == FEATURES and all(value == 0 for value in empty[arm].values())
    invalid = [
        ([4], [1], ["fold"], [[.1, .2, .3, .4]], [.2], [1]),
        ([0], [3], ["fold"], [[.1, .2, .3, .4]], [.2], [1]),
        ([0], [1], ["x"], [[.1, .2, .3, .4]], [.2], [1]),
        ([0.5], [1], ["fold"], [[.1, .2, .3, .4]], [.2], [1]),
        ([0], [1], ["fold"], [[.1, .2, .3]], [.2], [1]),
        ([0], [1], ["fold"], [[1.1, .2, .3, .4]], [.2], [1]),
        ([0], [1], ["fold"], [[.1, .2, .3, .4]], [.2], [-1]),
    ]
    for args in invalid:
        try:
            aggregate(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")


def main():
    test_single_event_control()
    test_shared_permutation_mean_of_max()
    test_global_reassignment_control_and_linked_change()
    test_row_order_and_category_isolation()
    test_empty_and_invalid_inputs()
    print("outsider card feature synthetic tests passed")


if __name__ == "__main__":
    main()


