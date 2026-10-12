"""Release-gated two-worker table shard writer. Does not fit or score models."""
import concurrent.futures as cf, hashlib, json, os, shutil, sys, time
from pathlib import Path
os.environ["POLARS_MAX_THREADS"] = "1"
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / ".venv/Lib/site-packages")); sys.path.insert(0, str(HERE))
from stream_contexts import INPUTS, build_style_counts, extract_table, sha, table_ids

def worker(table_id):
    import polars as pl
    out = HERE / "table_shards" / f"{table_id}.parquet"; receipt = out.with_suffix(".json")
    assert not out.exists() and not receipt.exists()
    assert shutil.disk_usage(HERE).free >= 2 * 1024**3, "worker sees less than 2 GiB free before extraction"
    start = time.perf_counter(); frame = extract_table(table_id, HERE / "evaluation_style_counts.parquet")
    assert shutil.disk_usage(HERE).free >= 2 * 1024**3, "worker sees less than 2 GiB free before write"
    frame.write_parquet(out)
    report = {"table_id": table_id, "rows": frame.height, "pairs": frame["pair_id"].n_unique(),
              "directions": frame.select("pair_id", "direction").unique().height,
              "sha256": sha(out), "seconds": time.perf_counter() - start,
              "feature_columns": INPUTS, "label_columns": []}
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return report

def main():
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    release = json.loads((HERE / "EXTRACTION_RELEASE.json").read_text()) if (HERE / "EXTRACTION_RELEASE.json").exists() else {}
    assert release.get("manifest_sha256") == sha(HERE / "MANIFEST.json"), "root extraction release absent"
    for rel, expected in manifest["inputs"].items(): assert sha(ROOT / rel) == expected, rel
    for rel, expected in manifest["code"].items(): assert sha(HERE / rel) == expected, rel
    assert not (HERE / "table_shards").exists() and not (HERE / "evaluation_style_counts.parquet").exists()
    assert not (HERE / "pair_coverage.parquet").exists() and not (HERE / "COVERAGE.json").exists()
    assert shutil.disk_usage(HERE).free >= 2 * 1024**3, "less than 2 GiB free before extraction"
    (HERE / "table_shards").mkdir(); start = time.perf_counter()
    style = build_style_counts(HERE / "evaluation_style_counts.parquet")
    reports = []; tables = iter(table_ids()); pool = cf.ProcessPoolExecutor(max_workers=2); pending = {}
    try:
        for _ in range(2):
            table = next(tables, None)
            if table is not None: pending[pool.submit(worker, table)] = table
        while pending:
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for future in done:
                table = pending.pop(future); report = future.result(); reports.append(report)
                artifact_bytes = sum(p.stat().st_size for p in (HERE / "table_shards").glob("*.parquet"))
                assert artifact_bytes <= 2 * 1024**3, "2 GiB artifact cap exceeded"
                assert shutil.disk_usage(HERE).free >= 2 * 1024**3, "2 GiB free-space reserve crossed"
                print(json.dumps({**report, "artifact_bytes": artifact_bytes}), flush=True)
                next_table = next(tables, None)
                if next_table is not None: pending[pool.submit(worker, next_table)] = next_table
    except BaseException:
        for future in pending: future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    import polars as pl
    directions = (pl.scan_parquet(HERE / "table_shards/*.parquet").group_by("pair_id", "direction")
                  .agg(pl.len().alias("contexts")).collect())
    assert set(directions["direction"].unique().to_list()) <= {0, 1}
    wide = (directions.with_columns((pl.lit("direction_") + pl.col("direction").cast(pl.String)).alias("d"))
            .pivot(on="d", index="pair_id", values="contexts").fill_null(0))
    keys = pl.read_parquet(HERE / "candidate_pair_keys.parquet").select("pair_id")
    assert keys.height == 112540 and keys["pair_id"].n_unique() == 112540
    assert directions.join(keys, on="pair_id", how="anti").height == 0, "generated pair outside candidate keys"
    pair_coverage = keys.join(wide, on="pair_id", how="left").fill_null(0)
    assert pair_coverage.height == 112540 and pair_coverage["pair_id"].n_unique() == 112540
    for name in ("direction_0", "direction_1"):
        if name not in pair_coverage.columns: pair_coverage = pair_coverage.with_columns(pl.lit(0).alias(name))
    pair_coverage.write_parquet(HERE / "pair_coverage.parquet")
    missing = pair_coverage.filter((pl.col("direction_0") == 0) | (pl.col("direction_1") == 0)).height
    coverage = {"tables": len(reports), "rows": sum(r["rows"] for r in reports),
                "style_counts": style, "seconds": time.perf_counter() - start,
                "candidate_pairs": pair_coverage.height, "exact_candidate_key_coverage": True,
                "extra_generated_pairs": 0, "directions_valid_subset_0_1": True,
                "pairs_missing_either_direction": missing,
                "pair_coverage_sha256": sha(HERE / "pair_coverage.parquet"),
                "malformed_cards": 0, "unknown_actions": 0, "nonfinite_features": 0,
                "strict_failure_semantics": "any such row aborts extraction rather than being imputed or silently dropped",
                "neutral_fallback": "future score 0.0 if either direction has zero valid contexts; availability is receipt-only",
                "policy_or_TN_convention_defined": False}
    (HERE / "COVERAGE.json").write_text(json.dumps(coverage, indent=2) + "\n")
if __name__ == "__main__": main()
