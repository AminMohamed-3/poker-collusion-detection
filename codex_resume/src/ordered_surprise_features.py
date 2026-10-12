"""Pure ordered partner-action surprise aggregates.

These are model-based observer descriptors.  They preserve chronological residual
products inside a street and provide an analytical within-actor permutation
control; they are not calibrated independence tests.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N_CLASSES = 4
FEATURES32 = tuple(
    [f"os_r{i}_r{j}_mean" for i in range(N_CLASSES) for j in range(N_CLASSES)]
    + [f"os_r{i}_r{j}_absmean" for i in range(N_CLASSES) for j in range(N_CLASSES)]
)


def _integer_vector(name: str, values: Sequence[object], minimum: int, maximum: int) -> np.ndarray:
    """Return a finite one-dimensional integer vector in the supplied bounds."""
    try:
        raw = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    rounded = np.rint(raw)
    if not np.array_equal(raw, rounded):
        raise ValueError(f"{name} must contain integers")
    result = rounded.astype(np.int64)
    if ((result < minimum) | (result > maximum)).any():
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _validated_inputs(
    directions: Sequence[object],
    streets: Sequence[object],
    action_numbers: Sequence[object],
    residuals: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions_array = _integer_vector("directions", directions, 0, 1)
    streets_array = _integer_vector("streets", streets, 0, 3)
    try:
        action_array = np.asarray(action_numbers, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("action_numbers must be numeric") from exc
    if action_array.ndim != 1 or not np.isfinite(action_array).all():
        raise ValueError("action_numbers must be a finite one-dimensional vector")
    rounded_actions = np.rint(action_array)
    if not np.array_equal(action_array, rounded_actions):
        raise ValueError("action_numbers must contain integers")
    action_array = rounded_actions.astype(np.int64)
    try:
        residual_array = np.asarray(residuals, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("residuals must be numeric") from exc
    count = directions_array.size
    if streets_array.size != count or action_array.size != count:
        raise ValueError("directions, streets, and action_numbers must have equal lengths")
    if residual_array.shape != (count, N_CLASSES) or not np.isfinite(residual_array).all():
        raise ValueError("residuals must be finite with shape (N, 4)")
    if np.unique(action_array).size != count:
        raise ValueError("action_numbers must be unique within a hand")
    order = np.argsort(action_array, kind="stable")
    directions_array = directions_array[order]
    streets_array = streets_array[order]
    action_array = action_array[order]
    residual_array = residual_array[order]
    # Pair-decision filtering can omit whole streets, but a real hand cannot
    # return to a betting street after a later street has started.
    if streets_array.size > 1 and np.any(np.diff(streets_array) < 0):
        raise ValueError("streets must be nondecreasing in action-number order")
    return directions_array, streets_array, action_array, residual_array


def _feature_dict(signed: np.ndarray, absolute: np.ndarray) -> dict[str, float]:
    """Emit the frozen row-major signed block followed by the absolute block."""
    values = np.concatenate((signed.reshape(-1), absolute.reshape(-1)))
    return {name: float(value) for name, value in zip(FEATURES32, values, strict=True)}


def aggregate_hand(
    directions: Sequence[object],
    streets: Sequence[object],
    action_numbers: Sequence[object],
    residuals: Sequence[Sequence[float]],
) -> dict[str, dict[str, float]]:
    """Aggregate one pair-hand's action residuals into frozen 32-field arms.

    Inputs contain only pair-decision rows.  Rows are sorted by their real
    ``action_no`` before adjacent, cross-partner edges are formed.  The ordered
    arm uses those chronological products.  The marginal arm is the exact
    independent-uniform within-actor/street permutation expectation over the
    same edge positions.
    """
    direction, street, _action_no, residual = _validated_inputs(
        directions, streets, action_numbers, residuals
    )
    edges = [
        index
        for index in range(direction.size - 1)
        if street[index] == street[index + 1] and direction[index] != direction[index + 1]
    ]
    ordered_signed = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    ordered_absolute = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    marginal_signed = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    marginal_absolute = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    if not edges:
        return {
            "ordered": _feature_dict(ordered_signed, ordered_absolute),
            "marginal_control": _feature_dict(marginal_signed, marginal_absolute),
        }

    # These are conditioned on exactly the actor and street of every original
    # edge, retaining each edge opportunity while discarding time association.
    means: dict[tuple[int, int], np.ndarray] = {}
    absolute_means: dict[tuple[int, int], np.ndarray] = {}
    for current_street in np.unique(street):
        for actor in (0, 1):
            values = residual[(street == current_street) & (direction == actor)]
            if values.size:
                means[(int(current_street), actor)] = values.mean(axis=0)
                absolute_means[(int(current_street), actor)] = np.abs(values).mean(axis=0)

    for index in edges:
        previous, current = residual[index], residual[index + 1]
        ordered_signed += np.outer(previous, current)
        ordered_absolute += np.abs(np.outer(previous, current))
        key_previous = (int(street[index]), int(direction[index]))
        key_current = (int(street[index + 1]), int(direction[index + 1]))
        marginal_signed += np.outer(means[key_previous], means[key_current])
        marginal_absolute += np.outer(absolute_means[key_previous], absolute_means[key_current])

    divisor = float(len(edges))
    return {
        "ordered": _feature_dict(ordered_signed / divisor, ordered_absolute / divisor),
        "marginal_control": _feature_dict(marginal_signed / divisor, marginal_absolute / divisor),
    }

