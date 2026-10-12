"""Pure outsider card-strength response summaries and a hand-level card control.

`linked` assigns each responding outsider its disclosed seat-card strength.  The
`card_control` keeps every outsider's complete card trajectory together and
averages the complete 24-field summary over all 24 seat-card permutations.  It
is an observer-side forensic control, not a normal-policy estimate.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np


CONTEXTS = ("one", "both")
ACTIONS = ("fold", "call", "aggr")
VALUES = ("strength", "ahead")
STATS = ("sum", "max")
FEATURES = tuple(
    f"od_{context}_{action}_{value}_{stat}"
    for context in CONTEXTS
    for action in ACTIONS
    for value in VALUES
    for stat in STATS
)
_PERMUTATIONS = tuple(permutations(range(4)))


def _integer_vector(values, name, allowed):
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must be a one-dimensional numeric vector")
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError(f"{name} must contain finite integers")
    result = raw.astype(np.int64)
    if not np.isin(result, tuple(allowed)).all():
        raise ValueError(f"{name} has an invalid value")
    return result


def _validate(actor_index, pressure, action, strengths, aggressor_strength, weights):
    actor = _integer_vector(actor_index, "actor_index", range(4))
    n = len(actor)
    context = _integer_vector(pressure, "pressure", (1, 2))
    acts = np.asarray(action, dtype=object)
    matrix = np.asarray(strengths, dtype=float)
    aggressor = np.asarray(aggressor_strength, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if len(context) != n or acts.ndim != 1 or len(acts) != n:
        raise ValueError("actor_index, pressure, and action must have equal length")
    if matrix.shape != (n, 4):
        raise ValueError("strengths must have shape (N, 4)")
    if aggressor.shape != (n,) or weight.shape != (n,):
        raise ValueError("aggressor_strength and weights must have shape (N,)")
    if not np.isfinite(matrix).all() or not np.isfinite(aggressor).all() or not np.isfinite(weight).all():
        raise ValueError("strengths, aggressor_strength, and weights must be finite")
    if (matrix < 0).any() or (matrix > 1).any() or (aggressor < 0).any() or (aggressor > 1).any():
        raise ValueError("strengths and aggressor_strength must lie in [0, 1]")
    if (weight < 0).any():
        raise ValueError("weights must be nonnegative")
    if not set(acts.tolist()).issubset(ACTIONS):
        raise ValueError("action has an invalid value")
    return actor, context, acts, matrix, aggressor, weight


def _summary(actor, context, acts, matrix, aggressor, weight):
    values = {name: 0.0 for name in FEATURES}
    for i in range(len(actor)):
        ctx = CONTEXTS[context[i] - 1]
        act = acts[i]
        strength = matrix[i, actor[i]] * weight[i]
        ahead = max(0.0, matrix[i, actor[i]] - aggressor[i]) * weight[i]
        for value_name, value in (("strength", strength), ("ahead", ahead)):
            sum_name = f"od_{ctx}_{act}_{value_name}_sum"
            max_name = f"od_{ctx}_{act}_{value_name}_max"
            values[sum_name] += float(value)
            values[max_name] = max(values[max_name], float(value))
    return values


def aggregate(actor_index, pressure, action, strengths, aggressor_strength, weights):
    """Return linked and shared-card-permutation control summaries.

    Rows are response events from one hand.  Column ``j`` of ``strengths`` is
    outsider seat/card trajectory ``j`` for every row; the control applies one
    common column permutation to the entire matrix for each of the 24 cases.
    """
    actor, context, acts, matrix, aggressor, weight = _validate(
        actor_index, pressure, action, strengths, aggressor_strength, weights
    )
    linked = _summary(actor, context, acts, matrix, aggressor, weight)
    control = {name: 0.0 for name in FEATURES}
    for permutation in _PERMUTATIONS:
        current = _summary(actor, context, acts, matrix[:, permutation], aggressor, weight)
        for name in FEATURES:
            control[name] += current[name] / len(_PERMUTATIONS)
    if not all(np.isfinite(value) and value >= 0 for value in (*linked.values(), *control.values())):
        raise AssertionError("invalid aggregate")
    return {"linked": linked, "card_control": control}
