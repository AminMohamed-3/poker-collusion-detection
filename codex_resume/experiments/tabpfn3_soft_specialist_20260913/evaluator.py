"""Write-once soft-play score assembly, then a separate held-out evaluation.

`assemble` reads frozen predictions and score inputs, never held outcomes.
`evaluate` is the only mode that reads development evidence and labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

STUDY = Path(__file__).resolve().parent
CODEX = STUDY.parents[1]
ROOT = CODEX.parent
SOURCE = CODEX / "experiments/tabpfn3_confirmation_20260912"
QUALIFIED = CODEX / "experiments/tabpfn3_confirmation_vm_20260912"
RETURNED = STUDY / "returned"
RUN = RETURNED / "run"
STEMS = tuple(f"fold{fold}-soft_play" for fold in range(3))
T_NAME = "tabpfn3_default"
FT_NAME = "sp_specialist"
NEG_NAME = "sp_stage1_negative"

sys.path.insert(0, str(CODEX / "src"))
sys.path.insert(0, str(ROOT / "Claude Version/src"))
from evidence_aug.frame import FAMILIES  # noqa: E402
from evidence_aug.paths import ORIGINAL, check_protected  # noqa: E402
from evidence_aug.step5_evaluate import half, per_pair_map, pool_bootstrap  # noqa: E402


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path: Path, value: dict) -> None:
    assert not path.exists(), f"write-once output exists: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verified_inputs() -> tuple[dict, dict, dict]:
    plan = read(STUDY / "plan.json")
    context = read(SOURCE / "context-receipt.json")
    assert sha(SOURCE / "context-receipt.json") == plan["source_context_receipt_sha256"]
    assert context["protected_unchanged"] and context["controls_exact"] == 9
    assert check_protected() == plan["protected"]
    for name, digest in plan["input_hashes"].items():
        assert sha(Path(name)) == digest, name
    qualified = read(QUALIFIED / "predictions-frozen.json")
    assert qualified["all_predictions_before_held_metrics"]
    assert qualified["context_receipt_sha256"] == sha(QUALIFIED / "context-receipt.json")
    assert sha(QUALIFIED / "context-receipt.json") == plan["source_context_receipt_sha256"]
    assert len(qualified["scores"]) == 9
    for name, digest in qualified["scores"].items():
        assert sha(QUALIFIED / "scores" / name) == digest, name
    manifests = {m["stem"]: m for m in context["contexts"] if m["family"] == "soft_play"}
    assert set(manifests) == set(STEMS)
    for stem, manifest in manifests.items():
        assert manifest["fold"] == int(stem[4])
        path = SOURCE / manifest["context_file"].replace("\\", "/")
        assert sha(path) == manifest["context_sha256"]
        assert manifest["width"] == 453
    return plan, manifests, qualified


def verified_run(plan: dict, manifests: dict) -> dict:
    study_freeze = RETURNED / "study-freeze.json"
    run_freeze = RUN / "run-freeze.json"
    complete = RUN / "complete.json"
    integrity_path = RETURNED / "remote-final-integrity.json"
    frozen = read(study_freeze)
    runner = read(run_freeze)
    done = read(complete)
    integrity = read(integrity_path)
    frozen_hash = sha(study_freeze)
    assert frozen["local_plan_sha256"] == sha(STUDY / "plan.json")
    assert runner["study_freeze_sha256"] == done["study_freeze_sha256"] == frozen_hash
    assert integrity["study_freeze_sha256"] == frozen_hash
    assert integrity["checked_contexts"] == 3 and integrity["all_live_artifact_hashes_match"] is True
    assert set(integrity["contexts"]) == set(STEMS)
    assert runner["epoch_count"] == 10
    assert set(done["context_receipts"]) == set(STEMS)
    assert sha(RUN / "baseline.json") == done["baseline_sha256"]
    baseline = read(RUN / "baseline.json")
    assert baseline["probabilities_exact"] and baseline["ranking_exact"]
    assert set(baseline["contexts"]) == set(STEMS)
    for item in baseline["contexts"].values():
        assert item["probabilities_exact"] and item["ranking_exact"]
        assert item["peak_reserved_bytes"] <= 13.5 * 1024**3
    for stem in STEMS:
        out = RUN / "contexts" / stem
        rec_path = out / "receipt.json"
        rec = read(rec_path)
        digests = integrity["contexts"][stem]
        assert set(digests) == {"receipt_sha256", "weights_sha256", "fit_state_sha256", "probabilities_sha256"}
        assert sha(rec_path) == done["context_receipts"][stem] == digests["receipt_sha256"]
        assert rec["status"] == "passed" and rec["epochs_required"] == 10
        assert len(rec["losses"]) == rec["gradient_trace"]["optimizer_step_calls"] == 10
        assert rec["training_autocast_dtype"] == "bfloat16" and rec["gradient_scaling_enabled"] is False
        assert rec["validation_disabled"] and rec["automatic_selection_disabled"]
        assert rec["saved_state_equals_live"] and rec["inference_config_matches_original"]
        assert rec["original_checkpoint_unchanged"]
        assert rec["train_shape"] == manifests[stem]["train_shape"]
        assert rec["query_shape"] == manifests[stem]["query_shape"]
        assert sha(out / "probabilities.npy") == rec["probabilities_sha256"] == digests["probabilities_sha256"]
        assert rec["weights_sha256"] == digests["weights_sha256"]
        assert rec["fit_state_sha256"] == digests["fit_state_sha256"]
        optional_artifacts = {
            "tabpfn-v3-classifier-finetuned-epoch10.ckpt": "weights_sha256",
            "matched_context.tabpfn_fit": "fit_state_sha256",
        }
        for filename, field in optional_artifacts.items():
            local = out / filename
            if local.exists():
                assert sha(local) == rec[field]
        if "files_sha256" in frozen:
            assert rec["source_sha256"] == frozen["files_sha256"]["tabpfn3_finetune_vm.py"]
    assert sha(RUN / "synthetic/replay.json") == done["synthetic_replay_sha256"]
    synthetic_replay = read(RUN / "synthetic/replay.json")
    real_stem = frozen["replay_stem"]
    assert real_stem in STEMS
    assert sha(RUN / "contexts" / real_stem / "replay.json") == done["real_replay_sha256"]
    real_replay = read(RUN / "contexts" / real_stem / "replay.json")
    for replay in (synthetic_replay, real_replay):
        assert replay["saved_fit_probabilities_exact"] and replay["refitted_checkpoint_probabilities_exact"]
        assert replay["saved_fit_rank_exact"] and replay["refitted_checkpoint_rank_exact"]
    return {
        "study_freeze_sha256": frozen_hash,
        "run_freeze_sha256": sha(run_freeze),
        "complete_sha256": sha(complete),
        "baseline_sha256": done["baseline_sha256"],
        "remote_integrity_sha256": sha(integrity_path),
        "synthetic_replay_sha256": done["synthetic_replay_sha256"],
        "real_replay_sha256": done["real_replay_sha256"],
    }

def score_path(fold: int, arm: str) -> Path:
    return STUDY / "scores" / f"fold{fold}-{arm}.parquet"


def qualified_path(fold: int, arm: str) -> Path:
    return QUALIFIED / "scores" / f"fold{fold}-{arm}.parquet"


def frame_for(fold: int, arm: str) -> pl.DataFrame:
    path = score_path(fold, arm) if arm in (FT_NAME, NEG_NAME) else qualified_path(fold, arm)
    frame = pl.read_parquet(path)
    assert frame.columns == ["pair_id", "hand_id", "table_id", "score"]
    assert frame["score"].is_finite().all()
    return frame


def assemble() -> None:
    assert not (STUDY / "predictions-frozen.json").exists()
    assert not (STUDY / "results.json").exists()
    assert not (STUDY / "scores").exists()
    plan, manifests, qualified = verified_inputs()
    run = verified_run(plan, manifests)
    (STUDY / "scores").mkdir(parents=True, exist_ok=False)
    scores = {}
    coverage = {}
    for fold, stem in enumerate(STEMS):
        manifest = manifests[stem]
        held = pl.read_parquet(SOURCE / f"fold{fold}-held.parquet")
        t = frame_for(fold, T_NAME)
        assert t.select("pair_id", "hand_id", "table_id").equals(held.select("pair_id", "hand_id", "table_id"))
        with np.load(SOURCE / manifest["context_file"].replace("\\", "/"), allow_pickle=False) as context:
            assert set(context.files) == {"xtrain", "ytrain", "xtest", "held_indices", "held_pair_ids", "groups_train", "groups_held"}
            indices = context["held_indices"]
            pair_ids = context["held_pair_ids"]
            assert indices.dtype == np.int64 and len(indices) == manifest["query_shape"][0]
            assert np.array_equal(np.unique(indices), indices)
            assert indices.min() >= 0 and indices.max() < held.height
            assert hashlib.sha256(indices.tobytes()).hexdigest() == manifest["held_indices_sha256"]
            assert np.array_equal(pair_ids, held["pair_id"].to_numpy()[indices])
            assert not set(context["groups_train"]) & set(context["groups_held"])
        top = ((held["predicted_behavior"] == "soft_play") & (held["oof_rank"] <= 20)).to_numpy()
        assert np.array_equal(indices, np.flatnonzero(top))
        route = (held["predicted_behavior"] == "soft_play").to_numpy()
        assert np.all(top <= route)
        out = RUN / "contexts" / stem
        proba = np.load(out / "probabilities.npy", allow_pickle=False)
        assert proba.shape == (len(indices),)
        assert np.isfinite(proba).all() and np.all((proba >= 0) & (proba <= 1))
        baseline = t["score"].to_numpy()
        candidate = baseline.copy()
        candidate[indices] = proba + 10
        negative = baseline.copy()
        negative[route] = held["oof"].to_numpy()[route]
        assert np.array_equal(candidate[~top], baseline[~top])
        assert np.array_equal(negative[~route], baseline[~route])
        assert np.array_equal(candidate[indices], proba + 10)
        assert np.isfinite(candidate).all() and np.isfinite(negative).all()
        for arm, values in ((FT_NAME, candidate), (NEG_NAME, negative)):
            path = score_path(fold, arm)
            assert not path.exists()
            t.with_columns(pl.Series("score", values)).write_parquet(path)
            scores[path.name] = sha(path)
        coverage[stem] = {"top_rows": int(top.sum()), "routed_rows": int(route.sum()), "context_sha256": manifest["context_sha256"], "receipt_sha256": sha(out / "receipt.json")}
    assert len(scores) == 6
    assert check_protected() == plan["protected"]
    write_once(STUDY / "predictions-frozen.json", {
        "plan_sha256": sha(STUDY / "plan.json"),
        "source_context_receipt_sha256": plan["source_context_receipt_sha256"],
        "qualified_predictions_frozen_sha256": sha(QUALIFIED / "predictions-frozen.json"),
        "study_freeze_sha256": run["study_freeze_sha256"],
        "run_freeze_sha256": run["run_freeze_sha256"],
        "complete_sha256": run["complete_sha256"],
        "baseline_sha256": run["baseline_sha256"],
        "remote_integrity_sha256": run["remote_integrity_sha256"],
        "synthetic_replay_sha256": run["synthetic_replay_sha256"],
        "real_replay_sha256": run["real_replay_sha256"],
        "scores": scores,
        "coverage": coverage,
        "all_predictions_before_held_metrics": True,
        "protected_unchanged": True,
    })


def frozen_for_evaluation() -> tuple[dict, dict]:
    plan, manifests, qualified = verified_inputs()
    run = verified_run(plan, manifests)
    frozen = read(STUDY / "predictions-frozen.json")
    assert frozen["all_predictions_before_held_metrics"] and frozen["protected_unchanged"]
    assert frozen["plan_sha256"] == sha(STUDY / "plan.json")
    assert frozen["source_context_receipt_sha256"] == plan["source_context_receipt_sha256"]
    assert frozen["qualified_predictions_frozen_sha256"] == sha(QUALIFIED / "predictions-frozen.json")
    assert frozen["study_freeze_sha256"] == run["study_freeze_sha256"]
    assert frozen["run_freeze_sha256"] == run["run_freeze_sha256"]
    assert frozen["complete_sha256"] == run["complete_sha256"]
    assert frozen["baseline_sha256"] == run["baseline_sha256"]
    assert frozen["remote_integrity_sha256"] == run["remote_integrity_sha256"]
    assert frozen["synthetic_replay_sha256"] == run["synthetic_replay_sha256"]
    assert frozen["real_replay_sha256"] == run["real_replay_sha256"]
    assert set(frozen["scores"]) == {score_path(fold, arm).name for fold in range(3) for arm in (FT_NAME, NEG_NAME)}
    for name, digest in frozen["scores"].items():
        assert sha(STUDY / "scores" / name) == digest
    return plan, frozen


def _mean(values: pl.Series) -> float:
    assert len(values) > 0
    return float(values.mean())


def _unique(appearances: pl.DataFrame) -> pl.DataFrame:
    return appearances.group_by("pair_id", "table_id", "family").agg(
        pl.col("delta").mean(), pl.col("base_map5").mean(), pl.col("map5").mean()
    )


def _summary(appearances: pl.DataFrame) -> dict:
    unique = _unique(appearances)
    delta = dict(unique.select("pair_id", "delta").iter_rows())
    tables = dict(unique.select("pair_id", "table_id").iter_rows())
    assert len(delta) == unique.height and len(tables) == unique.height
    ci = pool_bootstrap(delta, tables)
    folds = {str(f): _mean(appearances.filter(pl.col("fold") == f)["delta"]) for f in range(3)}
    halves = {str(h): _mean(unique.filter(pl.col("table_id").map_elements(half, return_dtype=pl.Int64) == h)["delta"]) for h in (0, 1)}
    return {"unique_pairs": unique.height, "unique_pools": unique["table_id"].n_unique(), "appearances": appearances.height,
            "mean_delta": _mean(unique["delta"]), "ci95": ci, "by_fold": folds, "halves": halves,
            "base_map5": _mean(unique["base_map5"]), "map5": _mean(unique["map5"]),
            "appearance_weighted_delta": _mean(appearances["delta"])}


def _incumbent_summary(appearances: pl.DataFrame) -> dict:
    summary = _summary(appearances)
    unique = _unique(appearances)
    summary["by_family"] = {f: _mean(unique.filter(pl.col("family") == f)["delta"]) for f in FAMILIES}
    summary["by_fold_family"] = {str(fold): {f: _mean(appearances.filter((pl.col("fold") == fold) & (pl.col("family") == f))["delta"]) for f in FAMILIES} for fold in range(3)}
    return summary


def evaluate() -> None:
    assert not (STUDY / "evaluation-invocation.json").exists()
    assert not (STUDY / "results.json").exists()
    assert not (STUDY / "pair_metrics.parquet").exists()
    plan, frozen = frozen_for_evaluation()
    write_once(STUDY / "evaluation-invocation.json", {
        "invoked_at": datetime.now(timezone.utc).isoformat(),
        "evaluator_sha256": sha(Path(__file__)),
        "plan_sha256": sha(STUDY / "plan.json"),
        "predictions_frozen_sha256": sha(STUDY / "predictions-frozen.json"),
        "all_scores_frozen_before_held_labels": True,
    })
    # Held outcomes enter only after all candidate and negative-control scores are frozen.
    evidence = pl.read_csv(ORIGINAL / "data/development_evidence.csv").select("pair_id", "hand_id")
    truths = {key[0]: set(group["hand_id"]) for key, group in evidence.group_by("pair_id")}
    labels = dict(pl.read_csv(ORIGINAL / "data/development_labels.csv").select("pair_id", "behavior_family").iter_rows())
    rows = []
    for fold in range(3):
        t = frame_for(fold, T_NAME)
        inc = frame_for(fold, "incumbent")
        ft = frame_for(fold, FT_NAME)
        neg = frame_for(fold, NEG_NAME)
        held = pl.read_parquet(SOURCE / f"fold{fold}-held.parquet")
        keys = t.select("pair_id", "hand_id", "table_id")
        assert keys.equals(inc.select("pair_id", "hand_id", "table_id"))
        assert keys.equals(ft.select("pair_id", "hand_id", "table_id"))
        assert keys.equals(neg.select("pair_id", "hand_id", "table_id"))
        assert keys.equals(held.select("pair_id", "hand_id", "table_id"))
        route = (held["predicted_behavior"] == "soft_play").to_numpy()
        assert np.array_equal(ft["score"].to_numpy()[~route], t["score"].to_numpy()[~route])
        assert np.array_equal(neg["score"].to_numpy()[~route], t["score"].to_numpy()[~route])
        role_rows = held.select("pair_id", "predicted_behavior").unique()
        assert role_rows["pair_id"].n_unique() == role_rows.height
        roles = dict(role_rows.iter_rows())
        tables = dict(t.select("pair_id", "table_id").unique().iter_rows())
        maps = {arm: per_pair_map(frame, truths) for arm, frame in ((T_NAME, t), ("incumbent", inc), (FT_NAME, ft), (NEG_NAME, neg))}
        assert set(maps[T_NAME]) == set(maps["incumbent"]) == set(maps[FT_NAME]) == set(maps[NEG_NAME]) == set(tables)
        for arm in (T_NAME, FT_NAME, NEG_NAME):
            base = maps["incumbent"] if arm == T_NAME else maps[T_NAME]
            for pair_id, value in maps[arm].items():
                rows.append({"fold": fold, "arm": arm, "pair_id": pair_id, "table_id": tables[pair_id],
                             "family": labels[pair_id], "predicted": roles[pair_id],
                             "base_map5": base[pair_id], "map5": value, "delta": value - base[pair_id]})
        # Candidate versus incumbent is needed for the original qualification gate.
        for pair_id, value in maps[FT_NAME].items():
            rows.append({"fold": fold, "arm": "specialist_vs_incumbent", "pair_id": pair_id,
                         "table_id": tables[pair_id], "family": labels[pair_id], "predicted": roles[pair_id],
                         "base_map5": maps["incumbent"][pair_id], "map5": value,
                         "delta": value - maps["incumbent"][pair_id]})
    frame = pl.DataFrame(rows)
    t_vs_inc = _incumbent_summary(frame.filter(pl.col("arm") == T_NAME))
    ft_vs_t = _summary(frame.filter(pl.col("arm") == FT_NAME))
    sp = _summary(frame.filter((pl.col("arm") == FT_NAME) & (pl.col("predicted") == "soft_play")))
    neg_sp = _summary(frame.filter((pl.col("arm") == NEG_NAME) & (pl.col("predicted") == "soft_play")))
    ft_vs_inc = _incumbent_summary(frame.filter(pl.col("arm") == "specialist_vs_incumbent"))
    assert t_vs_inc["unique_pairs"] == ft_vs_t["unique_pairs"] == ft_vs_inc["unique_pairs"] == 210
    assert t_vs_inc["unique_pools"] == ft_vs_t["unique_pools"] == ft_vs_inc["unique_pools"] == 141
    assert t_vs_inc["appearances"] == ft_vs_t["appearances"] == ft_vs_inc["appearances"] == 284
    previous = read(QUALIFIED / "results.json")["comparisons"][T_NAME]
    assert abs(t_vs_inc["base_map5"] - previous["unique_pair_base_map5"]) <= 1e-12
    assert abs(t_vs_inc["map5"] - previous["unique_pair_map5"]) <= 1e-12
    sp_checks = {"positive_unique_mean": sp["mean_delta"] > 0,
                 "positive_pool_ci_lower": sp["ci95"][0] > 0,
                 "all_folds_nonnegative": min(sp["by_fold"].values()) >= 0,
                 "both_halves_nonnegative": min(sp["halves"].values()) >= 0}
    negative_checks = {"positive_unique_mean": neg_sp["mean_delta"] > 0,
                       "positive_pool_ci_lower": neg_sp["ci95"][0] > 0,
                       "all_folds_nonnegative": min(neg_sp["by_fold"].values()) >= 0,
                       "both_halves_nonnegative": min(neg_sp["halves"].values()) >= 0}
    old_checks = {"mean_delta_min": ft_vs_inc["mean_delta"] >= .01,
                  "ci_lower_positive": ft_vs_inc["ci95"][0] > 0,
                  "all_fold_nonnegative": min(ft_vs_inc["by_fold"].values()) >= 0,
                  "family_floor": min(ft_vs_inc["by_family"].values()) >= -.01,
                  "every_fold_family_floor": min(value for fold in ft_vs_inc["by_fold_family"].values() for value in fold.values()) >= -.01,
                  "halves_nonnegative": min(ft_vs_inc["halves"].values()) >= 0}
    checks = {"sp_gate": all(sp_checks.values()), "overall_vs_T_nonnegative": ft_vs_t["mean_delta"] >= 0,
              "incumbent_gate_retained": all(old_checks.values()), "negative_control_rejected": not all(negative_checks.values()),
              "protected_unchanged": check_protected() == plan["protected"]}
    result = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "reused_outer_confirmation": True,
              "selected_after_prior_sp_slice": True, "predictions_frozen_sha256": sha(STUDY / "predictions-frozen.json"),
              "qualified_T_reproduced": True, "sp_vs_T": sp, "sp_checks": sp_checks,
              "negative_sp_vs_T": neg_sp, "negative_checks": negative_checks,
              "overall_vs_T": ft_vs_t, "T_vs_incumbent": t_vs_inc,
              "specialist_vs_incumbent": ft_vs_inc, "old_incumbent_checks": old_checks,
              "checks": checks, "passed": all(checks.values())}
    frame.write_parquet(STUDY / "pair_metrics.parquet")
    write_once(STUDY / "results.json", result)
    print(json.dumps({"passed": result["passed"], "sp_mean_delta": sp["mean_delta"], "sp_ci95": sp["ci95"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("assemble", "evaluate"))
    args = parser.parse_args()
    {"assemble": assemble, "evaluate": evaluate}[args.mode]()
