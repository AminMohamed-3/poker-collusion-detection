"""Hash the bounded, selected artifacts needed to preserve the current candidate."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "Claude Version"
OUT = Path(__file__).resolve().parent

FILES = [
    ROOT / "goal_top5_20260919/runtime.py",
    ROOT / "goal_top5_20260919/compose_ready.py",
    ROOT / "goal_top5_20260919/audit_ready.py",
    ROOT / "goal_top5_20260919/outputs/composition_protocol.json",
    ROOT / "goal_top5_20260919/outputs/composition_receipt.json",
    ROOT / "goal_top5_20260919/outputs/kaggle_result.json",
    ROOT / "goal_top5_20260919/outputs/submission_sp277_ci276_on_step272.csv",
    ROOT / "goal_top5_20260919/reproduction/replay_step272_key.py",
    ROOT / "goal_top5_20260919/reproduction/replay_step272.py",
    ROOT / "goal_top5_20260919/reproduction/replay_current_composition.py",
    ROOT / "goal_top5_20260919/reproduction/step277_A_portable.py",
    ROOT / "goal_top5_20260919/reproduction/recovered_step272/block_key_M3xKM_16.parquet",
    ROOT / "goal_top5_20260919/reproduction/recovered_step272/V_arch3.parquet",
    ROOT / "goal_top5_20260919/reproduction/recovered_chronology_manifest.json",
    ROOT / "goal_top5_20260919/reproduction/inc_p_PRODbare.npy",
    ROOT / "goal_top5_20260919/reproduction/submission_step272_replayed.csv",
    ROOT / "goal_top5_20260919/reproduction/submission_current_replayed.csv",
    ROOT / "goal_top5_20260919/reproduction/step272_replay_receipt.json",
    ROOT / "goal_top5_20260919/reproduction/current_composition_replay_receipt.json",
    ROOT / "src/validation.py",
    ROOT / "requirements.lock.txt",
    ROOT / "requirements-v2.lock.txt",
    CV / "src/evidence_aug/paths.py",
    CV / "src/evidence_aug/step104_C_typed_decoder_build.py",
    CV / "src/evidence_aug/step108_C_stacked_decoder_production.py",
    CV / "src/evidence_aug/step128_C_v1_view_averaged_production.py",
    CV / "src/evidence_aug/step165_C_e2e_audit.py",
    CV / "src/evidence_aug/step167_C_lp_production.py",
    CV / "src/evidence_aug/step257_mech_eval.py",
    CV / "src/evidence_aug/step276_G_ci_gamma_build.py",
    CV / "src/evidence_aug/step277_A_sp_rule_features.py",
    CV / "src/evidence_aug/step277_B_sp_rule_build.py",
    CV / "outputs_step169_20260915/submission_step169.csv",
    CV / "outputs_step259b_20260918/submission_step259b_geofusion_dt_sp_on_step251.csv",
    CV / "outputs_step259b_20260918/scope_scores_geo.parquet",
    CV / "outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv",
    CV / "outputs_step276_20260919/submission_step276_cigamma2_on_step272.csv",
    CV / "outputs_step276_20260919/receipt_step276_gamma2.json",
    CV / "outputs_step277_20260919/submission_step277_spRule_on_step272.csv",
    CV / "outputs_step277_20260919/receipt_step277_w0.3.json",
    CV / "experiments/step272_mlp2_block_20260919/registration272.json",
    CV / "experiments/step272_mlp2_block_20260919/receipt_step272_build.json",
    CV / "experiments/step254_mechanical_tensor_20260917/step254_features.parquet",
    CV / "experiments/step277_generator_rule_20260919/sp_eval_features.parquet",
    CV / "experiments/step277_generator_rule_20260919/sp_rule_eval.parquet",
    ROOT / "reports/replay-v13/selected-pairs.parquet",
]
FILES += sorted((CV / "experiments/step277_generator_rule_20260919/sp_rule_models").glob("*.txt"))

TEMP_BASE = Path(
    r"C:\Users\USER\AppData\Local\Temp\claude\C--Data-Projects-Money-Making-work-poker"
    r"\f2bb4f17-323f-4806-98b3-984023c70212\scratchpad\step257"
)
LARGE_DIRS = {
    "step251_work": CV / "outputs_step251_20260917/work",
    "chrono_development_external_temp": TEMP_BASE / "chrono_development",
    "chrono_evaluation_external_temp": TEMP_BASE / "chrono_evaluation",
    "arch2_evaluation_views": CV / "experiments/arch2_views_20260915",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def file_entry(path: Path) -> dict:
    if not path.is_file():
        return {"path": rel(path), "exists": False}
    return {"path": rel(path), "exists": True, "bytes": path.stat().st_size, "sha256": sha256(path)}


def dir_entry(path: Path) -> dict:
    if not path.is_dir():
        return {"path": rel(path), "exists": False}
    members = [p for p in path.rglob("*") if p.is_file()]
    return {
        "path": rel(path),
        "exists": True,
        "file_count": len(members),
        "bytes": sum(p.stat().st_size for p in members),
        "note": "bounded inventory only; individual members intentionally not hashed in this audit",
    }


packages = {}
for package in ("numpy", "polars", "lightgbm", "scikit-learn", "joblib", "pyarrow", "pandas"):
    try:
        packages[package] = version(package)
    except PackageNotFoundError:
        packages[package] = None

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "packages": packages,
    "files": [file_entry(p) for p in FILES],
    "large_dependency_directories": {k: dir_entry(v) for k, v in LARGE_DIRS.items()},
    "source_integrity": {
        "own_python_files_le_10_bytes": [
            rel(p) for p in (CV / "src").rglob("*.py") if p.stat().st_size <= 10
        ],
        "pair_side_builder_source_present": True,
        "pair_side_builder_note": "Recovered exact B3 fitted-key and B4 CSV builders from the original temp session; portable copies replayed the key and d5891a68 CSV exactly.",
    },
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"manifest": str(OUT / "manifest.json"), "files": len(manifest["files"]),
                  "missing": [x["path"] for x in manifest["files"] if not x["exists"]]}, indent=2))
