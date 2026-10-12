"""Frozen full-hybrid Nori wager screen; execute only after all receipts are complete."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from audit_common import *  # supplies the frozen current-cache helpers and np/pl

OUT = HERE / "experiments/nori_wager_20260912"
FDIR = HERE / "research/nori_wager_20260912"
ARMS = ("incumbent", "tree_quantiles", "nori")


def _index_file(design: dict, kind: str) -> Path:
    """Return the conventionally named, hash-bound frozen index artifact."""
    fold = design["fold"]
    path = FDIR / f"{kind}-{fold}-indices.npy"
    if sha(path) != design[f"{kind}_indices_sha256"]:
        raise ValueError(f"frozen {kind} index hash mismatch for fold {fold}")
    return path


def _validate_reference_design(corpus: dict) -> tuple[list[dict], Path]:
    inputs = corpus.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 38 or inputs[-1] != "is_raise":
        raise ValueError("Nori corpus must contain the frozen 38 input columns")
    designs = corpus.get("design")
    if not isinstance(designs, list) or {item.get("fold") for item in designs} != set(range(5)):
        raise ValueError("corpus design must contain folds 0..4")
    normal_path = FDIR / "normal-keys.parquet"
    normal_keys = pl.read_parquet(normal_path).select("table_id")
    pools = normal_keys["table_id"].to_numpy()
    outer = set(corpus["outer_pools"])
    exclusions = corpus["encoder_exclusions"]
    for item in designs:
        fold = item["fold"]
        train, held = set(item["train_pools"]), set(item["held_pools"])
        if train & held or train & outer or train & set(exclusions[str(fold)]):
            raise ValueError(f"invalid Nori reference-pool isolation for fold {fold}")
        train_indices = np.load(_index_file(item, "train"), allow_pickle=False)
        held_indices = np.load(_index_file(item, "held"), allow_pickle=False)
        for name, indices, expected in (("train", train_indices, 2048), ("held", held_indices, 1024)):
            if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != expected:
                raise ValueError(f"invalid {name} index shape for fold {fold}")
            if np.any(indices < 0) or np.any(indices >= len(pools)) or np.unique(indices).size != len(indices):
                raise ValueError(f"invalid {name} index values for fold {fold}")
        if np.intersect1d(train_indices, held_indices).size:
            raise ValueError(f"train/held index overlap for fold {fold}")
        if set(pools[train_indices]) != train or set(pools[held_indices]) != held:
            raise ValueError(f"frozen index pools disagree with design for fold {fold}")
    return designs, normal_path


def _training_artifacts(training: dict) -> tuple[Path, list[Path]]:
    freeze_path = FDIR / "training-freeze.json"
    if sha(freeze_path) != training.get("training_freeze_sha256"):
        raise ValueError("training-freeze hash mismatch")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    for path, digest in frozen["source_hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"frozen training source changed: {path}")
    artifacts: list[Path] = [freeze_path]
    receipts = training.get("receipts", [])
    expected = {(arm, fold) for arm in ARMS[1:] for fold in range(5)}
    if {(row.get("arm"), row.get("fold")) for row in receipts} != expected:
        raise ValueError("training receipt must contain ten arm/fold records")
    for receipt in receipts:
        arm, fold = receipt["arm"], receipt["fold"]
        score_path = FDIR / f"{arm}-{fold}-quantiles.npy"
        held_path = FDIR / f"{arm}-{fold}-held-quantiles.npy"
        if not receipt.get("reload_exact") or sha(score_path) != receipt.get("quantiles_sha256") or sha(held_path) != receipt.get("held_quantiles_sha256"):
            raise ValueError(f"quantile receipt mismatch: {arm} fold {fold}")
        artifacts.extend((score_path, held_path))
        if arm == "tree_quantiles":
            hashes = receipt.get("model_hashes")
            if not isinstance(hashes, dict) or len(hashes) != 3:
                raise ValueError(f"tree receipt lacks three model hashes for fold {fold}")
            for path, digest in hashes.items():
                if sha(path) != digest:
                    raise ValueError(f"tree model changed: {path}")
                artifacts.append(Path(path))
    return freeze_path, artifacts
def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "freeze.json").exists(), "write-once audit"
    corpus_path = FDIR / "corpus-freeze.json"
    training_path = FDIR / "training-receipt.json"
    feature_path = FDIR / "features-receipt.json"
    protocol = HERE / "research/nori_wager_protocol_20260912.md"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    if training.get("status") != "complete" or features.get("status") != "complete":
        raise ValueError("Nori training and feature receipts must be complete")
    if features.get("fields") != 26 or len(features.get("columns", [])) != 26 or not features.get("pair_swap_exact"):
        raise ValueError("invalid symmetric Nori feature receipt")
    reference_design, normal_keys_path = _validate_reference_design(corpus)
    for path, digest in corpus["source_hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"corpus source changed: {path}")
    for name, digest in corpus["artifacts"].items():
        if sha(FDIR / name) != digest:
            raise ValueError(f"corpus artifact changed: {name}")
    for path, digest in features["source_hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"feature input changed: {path}")
    training_freeze_path, training_artifacts = _training_artifacts(training)
    receipt_by_arm_fold = {(row["arm"], row["fold"]): row for row in training.get("receipts", [])}
    if set(receipt_by_arm_fold) != {(arm, fold) for arm in ARMS[1:] for fold in range(5)}:
        raise ValueError("training receipt must contain exactly ten estimator references")
    for (arm, fold), receipt in receipt_by_arm_fold.items():
        quantile_path = FDIR / f"{arm}-{fold}-quantiles.npy"
        if not receipt.get("reload_exact") or sha(quantile_path) != receipt.get("quantiles_sha256"):
            raise ValueError(f"invalid frozen quantiles for {arm} fold {fold}")
    for name, digest in features["artifacts"].items():
        if sha(FDIR / name) != digest:
            raise ValueError(f"feature artifact changed: {name}")

    best = CV / "outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv"
    best_sha = "6f064147875b0c448355666eb6448a2f5105f19cd89bcaf4a5308d995b7f678b"
    assert sha(best) == best_sha
    protected = check_protected()
    sources = [
        Path(__file__), HERE / "src/audit_common.py", protocol, corpus_path, training_path, feature_path,
        normal_keys_path, training_freeze_path, *training_artifacts, *(_index_file(item, kind) for item in reference_design for kind in ("train", "held")),
        *(FDIR / name for name in features["artifacts"]), PREV / "scores.parquet",
        PREV / "predictions-frozen.json", best,
    ]
    write(OUT / "freeze.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {str(path): sha(path) for path in sources},
        "arms": ARMS,
        "eligible_arms": ["nori"],
        "additional_gate": "nori-minus-tree_quantiles paired pool-bootstrap lower bound > 0",
        "parameters": dict(CLF, random_state=11),
        "fits": 45,
        "appended_features": features["columns"],
        "protected": protected,
        "original_outer_pools_excluded": True,
        "limits": [
            "Fixed current upstream cache is not fully nested.",
            "Reused selection audit, not fresh confirmation.",
            "Same fold-specific wager reference for downstream train and held rows.",
            "Only Nori is eligible; no control substitution or rescue.",
        ],
    })

    data, shared, rcols, saved = load_current()
    keys = pl.read_parquet(FDIR / "target-keys.parquet").sort("pair_id", "hand_id")
    expected = data.filter(pl.col("oof_rank") <= 20).select("pair_id", "hand_id")
    assert keys.equals(expected) and keys.height == 3240
    fcols = features["columns"]
    extra: dict[tuple[str, int], np.ndarray] = {}
    for arm in ARMS[1:]:
        for fold in range(5):
            feature_table = pl.read_parquet(FDIR / f"{arm}-{fold}-features.parquet").sort("pair_id", "hand_id")
            assert feature_table.select("pair_id", "hand_id").equals(keys)
            joined = data.select("pair_id", "hand_id").join(
                feature_table, on=["pair_id", "hand_id"], how="left", validate="1:1"
            ).sort("pair_id", "hand_id")
            assert joined.select("pair_id", "hand_id").equals(data.select("pair_id", "hand_id"))
            extra[(arm, fold)] = joined.select(fcols).to_numpy()

    predictions = {arm: data["oof"].to_numpy().copy() - 1e6 for arm in ARMS}
    fills = {arm: np.zeros(data.height, dtype=np.int8) for arm in ARMS}
    controls: list[dict] = []
    outer = set(corpus["outer_pools"])
    for family in FAMILIES:
        index = np.flatnonzero(((data["behavior_family"] == family) & (data["oof_rank"] <= 20)).to_numpy())
        query = data[index]
        columns = shared + (rcols if family != "coordinated_isolation" else [])
        x = query.select(columns).to_numpy().astype(float)
        y = query["relevance"].to_numpy().astype(int)
        groups = query["table_id"].to_numpy()
        for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(x, y, groups)):
            detector = next(item for item in corpus["detector_design"] if item["family"] == family and item["fold"] == fold)
            reference = next(item for item in reference_design if item["fold"] == fold)
            train_pools, held_pools = set(reference["train_pools"]), set(reference["held_pools"])
            assert set(groups[train]) == set(detector["training_pools"])
            assert set(groups[test]) == set(detector["held_pools"])
            assert not train_pools & set(groups[test])
            assert not train_pools & held_pools and not train_pools & outer
            assert not train_pools & set(corpus["encoder_exclusions"][str(fold)])
            for arm in ARMS:
                features_x = x if arm == "incumbent" else np.column_stack((x, extra[(arm, fold)][index]))
                assert np.isfinite(features_x).all()
                model = lgb.LGBMClassifier(**CLF, random_state=11).fit(features_x[train], y[train])
                probability = model.predict_proba(features_x[test])[:, 1]
                model_path = OUT / f"{arm}-{family}-{fold}.txt"
                model.booster_.save_model(str(model_path))
                replay = lgb.Booster(model_file=str(model_path)).predict(features_x[test], num_threads=2)
                assert np.array_equal(probability, replay)
                control = {"arm": arm, "family": family, "fold": fold,
                           "reload_exact": True, "model_sha256": sha(model_path)}
                if arm == "incumbent":
                    assert np.array_equal(probability + 10, saved["incumbent"].to_numpy()[index[test]])
                    control["reference_exact"] = True
                predictions[arm][index[test]] = probability + 10
                fills[arm][index[test]] += 1
                controls.append(control)
        print("Completed", family, round(time.perf_counter() - started, 1), flush=True)

    top = (data["oof_rank"] <= 20).to_numpy()
    assert all(np.all(value[top] == 1) and np.all(value[~top] == 0) for value in fills.values())
    assert all(np.array_equal(value[~top], data["oof"].to_numpy()[~top] - 1e6) for value in predictions.values())
    predictions["stage1_only"] = data["oof"].to_numpy()
    scores_path = OUT / "scores.parquet"
    data.select("pair_id", "hand_id", "table_id").with_columns(
        *(pl.Series(name, values) for name, values in predictions.items())
    ).write_parquet(scores_path)
    write(OUT / "predictions-frozen.json", {
        "freeze_sha256": sha(OUT / "freeze.json"),
        "scores_sha256": sha(scores_path),
        "controls": controls,
        "top20_all_once": True,
        "tail_exact": True,
        "outer_truth_read": False,
    })

    result = evaluate(data, predictions)
    contrast = evaluate(data, {"incumbent": predictions["tree_quantiles"], "nori": predictions["nori"]})
    result["contrasts"] = {"nori_minus_tree_quantiles": contrast["comparisons"]["nori"]}
    passed = result["comparisons"]["nori"]["gate_pass"] and contrast["comparisons"]["nori"]["ci95"][0] > 0
    result["passed"] = ["nori"] if passed else []
    result["selected_for_confirmation"] = "nori" if passed else None
    result["negative_control_rejected"] = not result["comparisons"]["stage1_only"]["gate_pass"]
    assert result["negative_control_rejected"]
    result["protected_unchanged"] = check_protected() == protected and sha(best) == best_sha
    assert result["protected_unchanged"]
    for path, digest in json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))["source_hashes"].items():
        assert sha(path) == digest
    result["seconds"] = time.perf_counter() - started
    result["no_outer_evidence_truth_read"] = True
    write(OUT / "results.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()



