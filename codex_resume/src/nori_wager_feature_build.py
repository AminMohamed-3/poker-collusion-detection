"""Frozen symmetric conditional-wager surprise summaries; no model fitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from within_hand_replay_capped import sha, write

HERE = Path(__file__).resolve().parents[1]
FDIR = HERE / "research/nori_wager_20260912"
ARMS = ("tree_quantiles", "nori")
COLS = [f"nw_summary_{index}" for index in range(26)]
AK = ["pair_id", "hand_id", "action_no", "direction"]
REQUIRED_QUERY_FIELDS = ("own_strength", "facing_partner", "players_active")


def pair_summary(means: np.ndarray, maxima: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if means.shape != maxima.shape or means.ndim != 3 or means.shape[1:] != (2, 8):
        raise ValueError("means and maxima must have shape [hands, 2, 8]")
    if counts.shape != means.shape[:2] or np.any(counts < 0):
        raise ValueError("counts must have shape [hands, 2] and be nonnegative")
    return np.column_stack((
        (means[:, 0] + means[:, 1]) / 2,
        np.abs(means[:, 0] - means[:, 1]),
        np.maximum(maxima[:, 0], maxima[:, 1]),
        counts.sum(axis=1) / 24,
        np.abs(counts[:, 0] - counts[:, 1]) / 24,
    ))


def aggregate(
    keys: pl.DataFrame,
    actions: pl.DataFrame,
    actual: np.ndarray,
    upper: np.ndarray,
    quantiles: np.ndarray,
    own_strength: np.ndarray,
    facing_partner: np.ndarray,
    players_active: np.ndarray,
) -> np.ndarray:
    """Aggregate one estimator's raw q10/q50/q90 score predictions into 26 fields."""
    if keys.columns != ["pair_id", "hand_id"] or keys.select(["pair_id", "hand_id"]).n_unique() != keys.height:
        raise ValueError("keys must be unique pair_id/hand_id rows")
    if any(name not in actions.columns for name in AK):
        raise ValueError("actions must include pair_id, hand_id, action_no, direction")
    if actions.select(AK).n_unique() != actions.height:
        raise ValueError("actions must have unique directional action keys")
    rows = actions.height
    arrays = {"actual": actual, "upper": upper, "own_strength": own_strength,
              "facing_partner": facing_partner, "players_active": players_active}
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.shape != (rows,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite numeric vector aligned to actions")
    if not isinstance(quantiles, np.ndarray) or quantiles.shape != (rows, 3) or not np.isfinite(quantiles).all():
        raise ValueError("quantiles must be a finite [actions, 3] array")
    if np.any(actual < 0) or np.any(upper < 0) or np.any(actual > upper + 1e-5):
        raise ValueError("actual and upper must be nonnegative with actual <= upper")
    direction = actions["direction"].to_numpy()
    if direction.dtype.kind not in "iu" or np.any((direction < 0) | (direction > 1)):
        raise ValueError("direction must contain only 0 or 1")
    mapping = {key: index for index, key in enumerate(keys.iter_rows())}
    try:
        hand_index = np.asarray([mapping[key] for key in actions.select(["pair_id", "hand_id"]).iter_rows()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("action key is absent from target keys") from exc

    ordered = np.sort(quantiles.astype(np.float64, copy=False), axis=1)
    q10, q50, q90 = np.clip(ordered, 0.0, upper[:, None]).T
    lower = np.clip(actual - q50, -3.0, 3.0)
    standardized = np.clip((actual - q50) / np.maximum(q90 - q10, 0.05), -5.0, 5.0)
    below = np.clip(np.maximum(q10 - actual, 0.0), 0.0, 3.0)
    above = np.clip(np.maximum(actual - q90, 0.0), 0.0, 3.0)
    outside = ((actual < q10) | (actual > q90)).astype(np.float64)
    width = q90 - q10
    strength = np.clip(own_strength, 0.0, 1.0)
    multiway_lower = above * (1.0 - strength) * (players_active >= 3)
    partner_upper = below * strength * (facing_partner > 0)
    values = np.column_stack((lower, standardized, below, above, outside, width, multiway_lower, partner_upper))

    sums = np.zeros((keys.height, 2, 8), dtype=np.float64)
    maxima = np.full_like(sums, -np.inf)
    counts = np.zeros((keys.height, 2), dtype=np.int64)
    np.add.at(sums, (hand_index, direction), values)
    np.add.at(counts, (hand_index, direction), 1)
    np.maximum.at(maxima, (hand_index, direction), values)
    means = sums / np.maximum(counts[:, :, None], 1)
    maxima[counts == 0] = 0.0
    out = pair_summary(means, maxima, counts)
    if out.shape != (keys.height, 26) or not np.isfinite(out).all():
        raise ValueError("invalid aggregate output")
    if not np.array_equal(out, pair_summary(means[:, ::-1], maxima[:, ::-1], counts[:, ::-1])):
        raise ValueError("pair summary is not direction-symmetric")
    return out.astype(np.float32)


def _checked_sources(corpus: dict, training: dict) -> list[Path]:
    sources: list[Path] = []
    for path, digest in corpus["source_hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"corpus source hash mismatch: {path}")
        sources.append(Path(path))
    for name, digest in corpus["artifacts"].items():
        path = FDIR / name
        if sha(path) != digest:
            raise ValueError(f"corpus artifact hash mismatch: {name}")
        sources.append(path)

    training_freeze = FDIR / "training-freeze.json"
    if sha(training_freeze) != training.get("training_freeze_sha256"):
        raise ValueError("training-freeze hash mismatch")
    frozen_training = json.loads(training_freeze.read_text(encoding="utf-8"))
    for path, digest in frozen_training["source_hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"frozen training source hash mismatch: {path}")
        sources.append(Path(path))
    sources.append(training_freeze)

    receipts = training.get("receipts")
    if training.get("status") != "complete" or not isinstance(receipts, list) or len(receipts) != 10:
        raise ValueError("training receipt must contain ten complete estimator references")
    for receipt in receipts:
        arm, fold = receipt.get("arm"), receipt.get("fold")
        if arm not in ARMS or not isinstance(fold, int) or not receipt.get("reload_exact"):
            raise ValueError("invalid estimator receipt")
        score_path = FDIR / f"{arm}-{fold}-quantiles.npy"
        held_path = FDIR / f"{arm}-{fold}-held-quantiles.npy"
        if sha(score_path) != receipt.get("quantiles_sha256") or sha(held_path) != receipt.get("held_quantiles_sha256"):
            raise ValueError(f"quantile hash mismatch: {arm} fold {fold}")
        sources.extend((score_path, held_path))
        if arm == "tree_quantiles":
            models = receipt.get("model_hashes")
            if not isinstance(models, dict) or len(models) != 3:
                raise ValueError(f"missing tree model hashes for fold {fold}")
            for path, digest in models.items():
                if sha(path) != digest:
                    raise ValueError(f"tree model hash mismatch: {path}")
                sources.append(Path(path))
    if {(item["arm"], item["fold"]) for item in receipts} != {(arm, fold) for arm in ARMS for fold in range(5)}:
        raise ValueError("missing or duplicate estimator receipt")
    return sources
def main() -> None:
    assert not (FDIR / "features-receipt.json").exists(), "write-once feature receipt"
    protocol = HERE / "research/nori_wager_protocol_20260912.md"
    corpus_path = FDIR / "corpus-freeze.json"
    training_path = FDIR / "training-receipt.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    inputs = corpus.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 38 or inputs[-1] != "is_raise":
        raise ValueError("corpus must freeze exactly 38 inputs ending in is_raise")
    if any(name not in inputs[:37] for name in REQUIRED_QUERY_FIELDS):
        raise ValueError("missing fixed query fields needed for Nori summaries")
    receipt_sources = _checked_sources(corpus, training)
    keys = pl.read_parquet(FDIR / "target-keys.parquet")
    actions = pl.read_parquet(FDIR / "score-keys.parquet")
    if not keys.equals(keys.sort(["pair_id", "hand_id"])) or not actions.equals(actions.sort(AK)):
        raise ValueError("target/action keys must be in frozen sorted order")
    bundle = np.load(FDIR / "score-corpus.npz", allow_pickle=False)
    try:
        x, actual, upper = bundle["x"], bundle["y"], bundle["upper"]
    finally:
        bundle.close()
    if x.shape != (actions.height, 38):
        raise ValueError("score corpus x must align to eligible score actions and 38 inputs")
    indices = {name: inputs.index(name) for name in REQUIRED_QUERY_FIELDS}
    summary_inputs = [x[:, indices[name]].astype(np.float64, copy=False) for name in REQUIRED_QUERY_FIELDS]
    sources = [Path(__file__), protocol, corpus_path, training_path, FDIR / "target-keys.parquet",
               FDIR / "score-keys.parquet", FDIR / "score-corpus.npz", *receipt_sources]
    artifacts: dict[str, str] = {}
    for arm in ARMS:
        for fold in range(5):
            path = FDIR / f"{arm}-{fold}-quantiles.npy"
            out = aggregate(keys, actions, actual, upper, np.load(path, allow_pickle=False), *summary_inputs)
            feature_path = FDIR / f"{arm}-{fold}-features.parquet"
            keys.with_columns(*[pl.Series(name, out[:, index]) for index, name in enumerate(COLS)]).write_parquet(feature_path)
            artifacts[feature_path.name] = sha(feature_path)
    write(FDIR / "features-receipt.json", {
        "status": "complete",
        "source_hashes": {str(path): sha(path) for path in sources},
        "artifacts": artifacts,
        "columns": COLS,
        "rows": keys.height,
        "fields": 26,
        "zero_event_hands": int(keys.height - len(set(actions.select(["pair_id", "hand_id"]).iter_rows()))),
        "pair_swap_exact": True,
        "no_evidence_truth_read": True,
    })
    print("10 Nori/tree feature tables complete; pair symmetry exact")


if __name__ == "__main__":
    main()

