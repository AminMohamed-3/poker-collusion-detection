"""Inventory and byte-copy immutable parent T contexts for fixed-epoch fine-tuning.

No feature construction, label access, fit, inference, selection, or metrics.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
CV = ROOT / "Claude Version"
PARENT = HERE / "experiments" / "tabpfn3_relative_20260912"
OUT = HERE / "experiments" / "tabpfn3_finetune_20260913"
DEFAULT_PROTOCOL = HERE / "research/tabpfn_finetune_quality_20260913/PROTOCOL.md"


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    path = Path(path)
    assert not path.exists(), f"write-once: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def protocol_path(value):
    path = Path(value) if value else DEFAULT_PROTOCOL
    if not path.is_absolute(): path = (HERE / path).resolve()
    assert path.exists(), f"frozen protocol is required: {path}"
    return path


def protected():
    import sys
    sys.path.insert(0, str(CV / "src"))
    from evidence_aug.paths import check_protected
    result = check_protected()
    for path in (CV / "outputs_stage2_r382_20260911" / "submission_M3_stage2_r382DTSP.csv",
                 HERE / "outputs_contestable_exploratory_20260912" / "submission_contestable_exploratory.csv"):
        result[str(path)] = sha(path)
    return result


def required_parent():
    paths = {"preflight": PARENT / "preflight.json", "normal": PARENT / "normal-reference-receipt.json",
             "contexts": PARENT / "context-receipt.json", "frame": PARENT / "audit-feature-frame.parquet"}
    assert all(path.exists() for path in paths.values()), paths
    return paths


def t_contexts(receipt):
    records = [x for x in receipt["contexts"] if x["arm"] == "T"]
    assert len(records) == 45 and not any(x["arm"] == "RZ" for x in records)
    indexed = {(x["salt"], x["fold"], x["family"]): x for x in records}
    families = {"directed_transfer", "soft_play", "coordinated_isolation"}
    assert set(indexed) == {(salt, fold, family) for salt in range(3) for fold in range(5) for family in families}
    return [indexed[key] for key in sorted(indexed)]


def validate_parent():
    """Hash-check the full parent provenance and return the sealed T manifests."""
    paths = required_parent()
    preflight, normal, receipt = (read(paths[key]) for key in ("preflight", "normal", "contexts"))
    assert sha(paths["preflight"]) == receipt["preflight_sha256"] == normal["preflight_sha256"]
    assert sha(paths["normal"]) == receipt["normal_reference_receipt_sha256"]
    assert sha(paths["frame"]) == receipt["artifact_hashes"]["audit-feature-frame.parquet"]
    assert len(preflight["audit_pools"]) == 104 and len(preflight["reference_pools"]) == 65
    assert not set(preflight["audit_pools"]) & set(preflight["reference_pools"])
    assert not (set(preflight["audit_pools"]) | set(preflight["reference_pools"])) & set(preflight["excluded_old_outer_pools"])
    assert preflight["reference_audit_seated_player_overlap"] == 0
    assert preflight["development_players_crossing_pools"] == 0
    assert normal["training_pools"] == preflight["reference_pools"]
    assert set(normal["scored_pools"]) == set(preflight["audit_pools"])
    assert protected() == preflight["protected"]
    for path, digest in {**preflight["inputs"], **preflight["sources"]}.items(): assert sha(path) == digest, path
    for relpath, digest in receipt["artifact_hashes"].items(): assert sha(PARENT / relpath) == digest, relpath

    partition = {x["salt_index"]: x for x in preflight["partitions"]}
    assert set(partition) == {0, 1, 2}
    assert len(receipt["nested"]) == 15
    assert len({x["frame"] for x in receipt["nested"]}) == 15
    for name, digest in normal["outputs"].items(): assert sha(PARENT/name) == digest
    for frame_record in receipt["nested"]:
        stem = frame_record["frame"]
        salt, fold = (int(value[1:]) for value in stem.split("-"))
        spec = next(x for x in partition[salt]["folds"] if x["fold"] == fold)
        mapping = partition[salt]["mapping"]
        outer = {pool for pool, value in mapping.items() if value == fold}
        outer_train = set(mapping) - outer
        assert set(frame_record["outer_train_pools"]) == outer_train
        assert set(frame_record["outer_held_pools"]) == outer
        assert {x["inner"] for x in frame_record["inner"]} == set(range(5))
        for inner in frame_record["inner"]:
            score_pools = {pool for pool, value in spec["inner_mapping"].items() if value == inner["inner"]}
            assert set(inner["score_pools"]) == score_pools
            assert set(inner["train_pools"]) == outer_train - score_pools
            assert not set(inner["train_pools"]) & set(inner["score_pools"])
            model = PARENT / "models" / f"{stem}-inner{inner['inner']}.txt"
            assert sha(model) == inner["model_sha256"], model

    inventory = []
    for manifest in t_contexts(receipt):
        source = PARENT / manifest["context_file"]
        assert source.exists() and sha(source) == manifest["context_sha256"]
        with np.load(source, allow_pickle=False) as arrays:
            expected = {"xtrain", "ytrain", "xtest", "held_indices", "held_pair_ids", "groups_train", "groups_held"}
            assert expected == set(arrays.files), (source, arrays.files)
            assert "ytest" not in arrays.files and "held_labels" not in arrays.files
            assert arrays["xtest"].dtype == arrays["xtrain"].dtype == np.float64
            assert np.isfinite(arrays["xtest"]).all() and np.isfinite(arrays["xtrain"]).all()
            assert set(arrays["ytrain"]) == {0,1}
            mapping = partition[manifest["salt"]]["mapping"]
            outer = {pool for pool, value in mapping.items() if value == manifest["fold"]}
            assert set(arrays["groups_train"]) <= (set(mapping)-outer)
            assert set(arrays["groups_held"]) <= outer
            assert not set(arrays["groups_train"]) & set(arrays["groups_held"])
            assert list(arrays["xtrain"].shape) == manifest["train_shape"]
            assert list(arrays["xtest"].shape) == manifest["query_shape"]
            assert len(arrays["held_indices"]) == arrays["xtest"].shape[0]
        inventory.append({"stem":manifest["stem"],"frame":manifest["frame"],"salt": manifest["salt"], "fold": manifest["fold"], "family": manifest["family"], "arm": "T",
                          "columns": manifest["columns"], "source_file": manifest["context_file"], "source_sha256": manifest["context_sha256"],
                          "train_shape": manifest["train_shape"], "query_shape": manifest["query_shape"], "class_counts": manifest["class_counts"],
                          "frames_reference_parent_unchanged": "frames/" + manifest["frame"] + ".parquet"})
    return paths, inventory


def preflight(protocol=None):
    """Validate the immutable source chain and inventory 45 T contexts; no fit or labels."""
    assert not (OUT / "preflight.json").exists(), "new output must be empty"
    frozen = protocol_path(protocol)
    paths, inventory = validate_parent()
    result = {"utc": datetime.now(timezone.utc).isoformat(), "protocol": str(frozen), "protocol_sha256": sha(frozen),
              "parent": {key: {"path": str(path), "sha256": sha(path)} for key, path in paths.items()},
              "preparation_source_sha256":sha(Path(__file__)),"contexts": inventory, "context_count": 45, "arms": ["T"],
              "fixed_training": {"epochs": 10, "early_stopping": False, "scheduler": False, "validation_split_ratio": 0,
                                 "validation_arrays": None, "all_parent_training_rows_retained": True},
              "frames_reference_parent_unchanged": True, "no_new_models": True, "no_outer_query_labels_parsed": True,
              "no_metrics_or_inference": True, "no_outer_query_labels": True}
    write_once(OUT / "preflight.json", result)
    return result


def check(protocol=None):
    result = read(OUT / "preflight.json")
    assert result["preparation_source_sha256"] == sha(Path(__file__))
    frozen = protocol_path(protocol)
    assert str(frozen) == result["protocol"] and sha(frozen) == result["protocol_sha256"]
    paths, inventory = validate_parent()
    assert result["parent"] == {key: {"path": str(path), "sha256": sha(path)} for key, path in paths.items()}
    assert result["contexts"] == inventory
    return result


def prepare(protocol=None):
    """Byte-copy parent T NPZs only. No arrays are opened after preflight or altered."""
    pre = check(protocol)
    assert not (OUT / "context-receipt.json").exists(), "write-once output already exists"
    target = OUT / "contexts"
    target.mkdir(exist_ok=False)
    copied = []
    for spec in pre["contexts"]:
        source = PARENT / spec["source_file"]
        destination = target / source.name
        shutil.copyfile(source, destination)
        assert sha(destination) == spec["source_sha256"]
        copied.append({**spec, "context_file": "contexts/" + destination.name, "context_sha256": sha(destination), "byte_exact_parent_npz": True})
    assert len(copied) == 45
    check(protocol)
    write_once(OUT / "context-receipt.json", {"preflight_sha256": sha(OUT / "preflight.json"), "contexts": copied,
               "copied_context_count": 45, "byte_exact_parent_npzs": True, "frames_reference_parent_unchanged": True,
               "no_new_models": True, "no_outer_query_labels_parsed": True, "no_metrics_or_inference": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "prepare"))
    parser.add_argument("--protocol", default=None, help="frozen protocol path; default is the experiment PROTOCOL.md")
    args = parser.parse_args()
    globals()[args.mode](args.protocol)
