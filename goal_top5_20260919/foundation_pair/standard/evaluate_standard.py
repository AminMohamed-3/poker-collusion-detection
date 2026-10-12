"""Frozen CPU-only evaluator for the standard TabPFN-3.5 architecture check.

Run only after known-fold, prefix, and fresh-process replay artifacts are terminal.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

OUT = Path(__file__).resolve().parent
SCREEN = OUT.parent / "screen"
FAMILIES = (1, 2, 3)
SEED = 35935
N_BOOT = 1000


def ap(y, score):
    return float(average_precision_score(np.asarray(y, dtype=bool), score))


def rank01(x):
    return rankdata(x, method="average") / len(x)


def metrics(y, risk, route):
    pair_ap = ap(y > 0, risk)
    behavior = float(np.mean([
        ap(y == family, np.where(route == family, risk, 0.0))
        for family in FAMILIES
    ]))
    return {
        "n": int(len(y)),
        "positives": int((y > 0).sum()),
        "pair_ap": pair_ap,
        "behavior_map_predicted": behavior,
        "pair_plus_behavior": .7 * pair_ap + .1 * behavior,
    }


def delta(candidate, baseline):
    return {key: float(candidate[key] - baseline[key]) for key in
            ("pair_ap", "behavior_map_predicted", "pair_plus_behavior")}


def validate_probabilities(p):
    assert p.ndim == 2 and p.shape[1] == 4
    assert np.isfinite(p).all() and np.all(p >= 0) and np.all(p <= 1)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)


def known_fold(fold, fast_archive, tree_archive):
    evaluator = np.load(SCREEN / f"evaluator_fold{fold}.npz", allow_pickle=False)
    context = np.load(SCREEN / f"context_fold{fold}.npz", allow_pickle=False)
    standard = np.load(OUT / f"predictions_fold{fold}.npz", allow_pickle=False)
    mask = tree_archive["fold"] == fold
    keys = evaluator["pair_id"]
    assert np.array_equal(keys, context["pair_id"])
    assert np.array_equal(keys, standard["pair_id"])
    assert np.array_equal(keys, tree_archive["pair_id"][mask])
    p_standard = standard["probabilities"]
    p_fast = fast_archive[f"known_fold{fold}"]
    p_lgb = tree_archive["lgb128"][mask]
    for p in (p_standard, p_fast, p_lgb):
        assert len(p) == len(keys)
        validate_probabilities(p)
    return {
        "pair_id": keys,
        "table_id": evaluator["table_id"],
        "y": evaluator["y"],
        "standard": p_standard,
        "fast": p_fast,
        "lgb": p_lgb,
    }


def prefix_fold(fold, fast_archive, tree_stress):
    evaluator = np.load(SCREEN / f"evaluator_fold{fold}_prefix60.npz", allow_pickle=False)
    context = np.load(SCREEN / f"context_fold{fold}_prefix60.npz", allow_pickle=False)
    standard = np.load(OUT / f"predictions_fold{fold}_prefix60.npz", allow_pickle=False)
    keys = evaluator["pair_id"]
    assert np.array_equal(keys, context["pair_id"])
    assert np.array_equal(keys, standard["pair_id"])
    p_standard = standard["probabilities"]
    p_fast = fast_archive[f"prefix60_fold{fold}"]
    p_lgb = tree_stress[f"prefix60_fold{fold}"]
    for p in (p_standard, p_fast, p_lgb):
        assert len(p) == len(keys)
        validate_probabilities(p)
    return {
        "pair_id": keys,
        "table_id": evaluator["table_id"],
        "y": evaluator["y"],
        "standard": p_standard,
        "fast": p_fast,
        "lgb": p_lgb,
    }


def score_frame(frame):
    y = frame["y"]
    risks = {name: 1.0 - frame[name][:, 0] for name in ("standard", "fast", "lgb")}
    routes = {name: 1 + np.argmax(frame[name][:, 1:], axis=1)
              for name in ("standard", "fast", "lgb")}
    primary_risk = .8 * rank01(risks["fast"]) + .2 * rank01(risks["standard"])
    lgb_analogue_risk = .8 * rank01(risks["lgb"]) + .2 * rank01(risks["standard"])
    scored = {
        "Fast": metrics(y, risks["fast"], routes["fast"]),
        "Standard_own_route": metrics(y, risks["standard"], routes["standard"]),
        "LGB128": metrics(y, risks["lgb"], routes["lgb"]),
        "primary_80Fast_20Standard_Fast_route": metrics(y, primary_risk, routes["fast"]),
        "diagnostic_80LGB_20Standard_LGB_route": metrics(y, lgb_analogue_risk, routes["lgb"]),
    }
    scored["primary_delta_vs_Fast"] = delta(
        scored["primary_80Fast_20Standard_Fast_route"], scored["Fast"])
    scored["lgb_analogue_delta_vs_LGB128"] = delta(
        scored["diagnostic_80LGB_20Standard_LGB_route"], scored["LGB128"])
    return scored, risks, routes, primary_risk


def concatenate(frames):
    return {key: np.concatenate([frame[key] for frame in frames])
            for key in frames[0]}


def bootstrap(frame, primary_risk, fast_risk, fast_route):
    pools = np.unique(frame["table_id"])
    by_pool = {pool: np.flatnonzero(frame["table_id"] == pool) for pool in pools}
    rng = np.random.default_rng(SEED)
    values = np.empty((N_BOOT, 3), dtype=np.float64)
    for iteration in range(N_BOOT):
        ix = np.concatenate([by_pool[p] for p in rng.choice(pools, len(pools), replace=True)])
        candidate = metrics(frame["y"][ix], primary_risk[ix], fast_route[ix])
        baseline = metrics(frame["y"][ix], fast_risk[ix], fast_route[ix])
        values[iteration] = [
            candidate["pair_ap"] - baseline["pair_ap"],
            candidate["behavior_map_predicted"] - baseline["behavior_map_predicted"],
            candidate["pair_plus_behavior"] - baseline["pair_plus_behavior"],
        ]
    names = ("pair_ap", "behavior_map_predicted", "pair_plus_behavior")
    return {
        "unit": "table_id",
        "tables": int(len(pools)),
        "resamples": N_BOOT,
        "seed": SEED,
        "recompute_ap_each_resample": True,
        "primary_minus_Fast": {
            name: {
                "p_gt_0": float(np.mean(values[:, j] > 0)),
                "ci95": [float(v) for v in np.quantile(values[:, j], [.025, .975])],
            } for j, name in enumerate(names)
        },
    }


def main():
    replay = json.loads((OUT / "replay_check.json").read_text())
    assert replay["exact"], "Fresh-process fold0 replay must be exact before evaluation"
    fast = np.load(SCREEN / "pf35_predictions.npz", allow_pickle=False)
    tree = np.load(SCREEN / "tree_oof_predictions.npz", allow_pickle=False)
    tree_stress = np.load(SCREEN / "tree_stress_predictions.npz", allow_pickle=False)

    known = [known_fold(fold, fast, tree) for fold in (0, 1)]
    prefix = [prefix_fold(fold, fast, tree_stress) for fold in (0, 1)]
    fold_results = {}
    for fold, frame in enumerate(known):
        fold_results[str(fold)], _, _, _ = score_frame(frame)
    prefix_fold_results = {}
    for fold, frame in enumerate(prefix):
        prefix_fold_results[str(fold)], _, _, _ = score_frame(frame)

    pooled = concatenate(known)
    pooled_results, pooled_risks, pooled_routes, pooled_primary = score_frame(pooled)
    pooled_prefix = concatenate(prefix)
    pooled_prefix_results, _, _, _ = score_frame(pooled_prefix)
    boot = bootstrap(pooled, pooled_primary, pooled_risks["fast"], pooled_routes["fast"])
    advance = bool(
        pooled_results["primary_delta_vs_Fast"]["pair_plus_behavior"] > 0
        and all(fold_results[str(f)]["primary_delta_vs_Fast"]["pair_plus_behavior"] > 0
                for f in (0, 1))
    )
    result = {
        "scope": "reused-label architecture screen; existing Fast candidate unaffected by failure",
        "fixed_primary": "0.8*global_percentile_rank(Fast risk) + 0.2*global_percentile_rank(Standard risk), Fast route retained",
        "diagnostic_lgb_analogue": "0.8*global_percentile_rank(LGB128 risk) + 0.2*global_percentile_rank(Standard risk), LGB128 route retained",
        "pooled_known": pooled_results,
        "known_folds": fold_results,
        "pooled_prefix60": pooled_prefix_results,
        "prefix60_folds": prefix_fold_results,
        "bootstrap": boot,
        "advance_rule": "pooled and both fold primary pair_plus_behavior deltas versus Fast are strictly positive",
        "advance_to_lofo": advance,
        "existing_fast_candidate_affected": False,
    }
    (OUT / "known_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "advance_to_lofo": advance,
        "pooled_primary_delta": pooled_results["primary_delta_vs_Fast"],
        "fold_primary_pair_plus_behavior_deltas": [
            fold_results[str(f)]["primary_delta_vs_Fast"]["pair_plus_behavior"] for f in (0, 1)
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
