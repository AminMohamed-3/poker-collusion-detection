# P4 Step A parity only: fold-0 screen rebuild vs pair-predictions.csv (full/conditional).
# CPU-only, 2 threads, utf-8. Reads read-only inputs; writes only next to this script.
# Adapts Claude Version/src/pu_pairs/build_allpairs_fold.py logic to fold-0 screen assets:
#   policy reports/conditional-action-experiment/policy-screen/held-models.joblib
#   caf    .../policy-screen/hand-features.parquet (ca_* columns)
#   bundle reports/conditional-action-experiment/screen-results/conditional-pair.joblib
#   saved  .../screen-results/pair-predictions.csv (history=full,candidate=conditional)
#   held   reports/tabicl-screen/held-pairs.csv (457 labelled pairs, 100 pools)
# Does NOT build prefix frames. If parity fails, stops and records why.
import os
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import sys, json, gc, hashlib, argparse
from pathlib import Path

ROOT = Path(r"C:\Data\Projects\Money Making\work\poker")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Claude Version" / "src"))
sys.path.insert(0, str(ROOT / "src"))

POLICY = ROOT / "reports/conditional-action-experiment/policy-screen/held-models.joblib"
HF = ROOT / "reports/conditional-action-experiment/policy-screen/hand-features.parquet"
BUNDLE = ROOT / "reports/conditional-action-experiment/screen-results/conditional-pair.joblib"
SAVED = ROOT / "reports/conditional-action-experiment/screen-results/pair-predictions.csv"
HELD = ROOT / "reports/tabicl-screen/held-pairs.csv"
STYLE = ROOT / "reports/conditional-action-experiment/public-action-style-counts.parquet"
PLAN = ROOT / "reports/conditional-action-experiment/screen-results/plan.json"

def sha(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools-per-batch", type=int, default=2)
    ap.add_argument("--fast-only", action="store_true")
    ap.add_argument("--rebuild-only", action="store_true")
    args = ap.parse_args()
    res = {"step": "P4A_parity_fold0", "assets": {}, "missing": []}
    for name, p in [("policy", POLICY), ("hand_features", HF), ("pair_bundle", BUNDLE),
                    ("pair_predictions", SAVED), ("held_pairs", HELD),
                    ("style_counts", STYLE), ("plan", PLAN)]:
        ok = p.exists()
        res["assets"][name] = {"path": str(p.relative_to(ROOT)), "exists": ok,
                               "sha256": sha(p)[:16] if ok and p.suffix in (".joblib", ".csv", ".json") else None}
        if not ok:
            res["missing"].append(str(p.relative_to(ROOT)))
    # hand-features/hand parquet are large: record only existence + row sample later
    if res["missing"]:
        res["verdict"] = "FAIL_MISSING_ASSETS"
        (OUT / "p4a_parity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1))
        return
    import polars as pl, numpy as np, joblib
    held = pl.read_csv(HELD)
    saved_all = pl.read_csv(SAVED)
    saved = saved_all.filter((pl.col("history") == "full") & (pl.col("candidate") == "conditional")).sort("pair_id")
    bundle = joblib.load(BUNDLE)
    feats = bundle["features"]
    policy = joblib.load(POLICY)
    caf = [c for c in pl.read_parquet(HF, n_rows=1).columns if c.startswith("ca_")]
    pools = sorted(held["table_id"].unique().to_list())
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    res.update({"n_held": held.height, "n_saved_cond_full": saved.height,
                "n_pools": len(pools), "pools_match_plan": set(pools) == set(plan["held_pools"]),
                "n_features": len(feats), "n_caf": len(caf),
                "policy_inputs": len(policy["inputs"])})
    if held.height != 457 or saved.height != 457:
        res["verdict"] = "FAIL_COUNT_MISMATCH"
        res["note"] = "expected 457 held and 457 saved full/conditional rows"
        (OUT / "p4a_parity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1))
        return
    # Fast cached parity: same construction as evaluate_conditional_action_screen.py (no seat/action rebuild)
    if not args.rebuild_only:
        try:
            from robust_pair_experiment import aggregate, matrix
            from tail_pair_experiment import tails
            from evaluate_conditional_action_screen import extra_aggregate
            import glob as _glob
            raw = pl.read_parquet(sorted(_glob.glob(str(ROOT / "outputs_v2/hands_development/batch_*.parquet"))))
            names = [c for c in raw.columns if c not in ("pair_id", "hand_id", "table_id")]
            events = [c for c in names if c.startswith("v2_")]
            ca = pl.read_parquet(HF)
            ca_feats = [c for c in ca.columns if c.startswith("ca_")]
            assert ca_feats == caf, "caf mismatch"
            combined = raw.join(ca.drop("table_id"), on=["pair_id", "hand_id"], how="left",
                                validate="1:1", maintain_order="left").with_columns(pl.col(ca_feats).fill_null(0))
            labels = pl.read_csv(ROOT / "data/development_labels.csv")
            base_full = pl.read_parquet(ROOT / "outputs_v2/pairs_development.parquet").join(
                tails(raw, events), on=["pair_id", "table_id"]).join(labels, on="pair_id").sort("pair_id")
            full = base_full.join(extra_aggregate(combined, ca_feats), on=["pair_id", "table_id"]).sort("pair_id")
            ext = [c + s for s in ("_mean", "_std", "_max", "_q95") for c in ca_feats]
            # bundle features = base(765) + ext(132); verify subset present
            miss = [f for f in feats if f not in full.columns]
            hf_frame = full.filter(pl.col("table_id").is_in(pools)).sort("pair_id")
            risk = None
            if not miss:
                risk = 1 - bundle["model"].predict_proba(matrix(hf_frame, feats))[:, 0]
                j = hf_frame.select("pair_id").with_columns(pl.Series("rebuild_risk", risk)).join(
                    saved.select("pair_id", pl.col("risk").alias("saved_risk")), on="pair_id", how="inner")
                d = (j["rebuild_risk"] - j["saved_risk"]).abs()
                res["fast_cached"] = {"n": j.height, "max_abs_delta": float(d.max()),
                                      "mean_abs_delta": float(d.mean()), "missing_features": []}
            else:
                res["fast_cached"] = {"n": 0, "missing_features": miss[:10]}
            del raw, ca, combined, base_full, full, hf_frame
            gc.collect()
        except Exception as e:
            res["fast_cached"] = {"error": f"{type(e).__name__}: {e}"}
    # Rebuild parity: build_allpairs_fold functions on held subset only (batched, full-length)
    if not args.fast_only:
        try:
            from pu_pairs.build_allpairs_fold import base_features, v2_features, conditional_features
            from robust_pair_experiment import aggregate, matrix
            from tail_pair_experiment import tails
            from evaluate_conditional_action_screen import extra_aggregate
            from conditional_action_features import card_features
            import numpy as _np
            counts = {r[0]: _np.array(r[1:], dtype=_np.int64) for r in pl.read_parquet(STYLE).iter_rows()}
            dev_labels = pl.read_csv(ROOT / "data/development_labels.csv").select("pair_id", "player_1", "player_2")
            hp = held.join(dev_labels, on="pair_id", how="left", validate="1:1")
            assert hp.filter(pl.col("player_1").is_null()).height == 0, "held pair missing players"
            hands_all = pl.read_parquet(ROOT / "data/hands.parquet").filter(
                (pl.col("phase") == "development") & pl.col("table_id").is_in(pools))
            risks = []
            for s in range(0, len(pools), args.pools_per_batch):
                these = pools[s:s + args.pools_per_batch]
                h = hands_all.filter(pl.col("table_id").is_in(these))
                p = hp.filter(pl.col("table_id").is_in(these)).select("pair_id", "player_1", "player_2", "table_id")
                raw = base_features(h, p)
                p2 = p.join(raw.select("pair_id").unique(), on="pair_id", how="semi")
                raw = v2_features(raw, p2)
                names = [c for c in raw.columns if c not in ("pair_id", "hand_id", "table_id")]
                events = [c for c in names if c.startswith("v2_")]
                keys = raw.select("pair_id", "hand_id", "table_id").join(
                    p2.select("pair_id", "player_1", "player_2"), on="pair_id", validate="m:1")
                cas = []
                for pool in these:
                    ca, _, _ = conditional_features(
                        keys.filter(pl.col("table_id") == pool),
                        h.select("hand_id", "table_id", "board_cards", "button_seat", "big_blind"),
                        policy, counts, caf)
                    cas.append(ca)
                    gc.collect()
                ca = pl.concat(cas)
                pf = aggregate(raw, names).join(tails(raw, events), on=["pair_id", "table_id"], validate="1:1").join(
                    extra_aggregate(ca, caf), on=["pair_id", "table_id"], validate="1:1").sort("pair_id")
                assert pf.height == p2.height, f"pairframe {pf.height} != pairs {p2.height}"
                r = 1 - bundle["model"].predict_proba(matrix(pf, feats))[:, 0]
                risks.append(pf.select("pair_id").with_columns(pl.Series("rebuild_risk", r)))
                card_features.cache_clear()
                del raw, ca, cas, pf, keys, h, p, p2
                gc.collect()
            rb = pl.concat(risks).sort("pair_id").join(
                saved.select("pair_id", pl.col("risk").alias("saved_risk")), on="pair_id", how="inner")
            d = (rb["rebuild_risk"] - rb["saved_risk"]).abs()
            res["rebuild"] = {"n": rb.height, "max_abs_delta": float(d.max()),
                              "mean_abs_delta": float(d.mean()),
                              "n_ge_1e12": int((d >= 1e-12).sum()), "n_ge_1e9": int((d >= 1e-9).sum())}
        except Exception as e:
            import traceback
            res["rebuild"] = {"error": f"{type(e).__name__}: {e}",
                              "trace_tail": traceback.format_exc()[-1500:]}
    # Verdict: primary = rebuild max|delta| == 0
    rb = res.get("rebuild", {})
    fc = res.get("fast_cached", {})
    if "max_abs_delta" in rb:
        res["parity_max_abs_delta"] = rb["max_abs_delta"]
        res["parity_n"] = rb["n"]
        res["verdict"] = "PASS" if rb["max_abs_delta"] == 0.0 and rb["n"] == 457 else "FAIL_REBUILD_MISMATCH"
        if res["verdict"].startswith("FAIL"):
            res["why"] = "rebuild risks differ from saved full/conditional; STOP, do not build prefix frames"
    elif "max_abs_delta" in fc and args.fast_only:
        res["parity_max_abs_delta"] = fc["max_abs_delta"]
        res["parity_n"] = fc["n"]
        res["verdict"] = "PASS_FAST_ONLY" if fc["max_abs_delta"] == 0.0 else "FAIL_FAST_MISMATCH"
    else:
        res["verdict"] = "FAIL_NO_PARITY"
        res["why"] = str(rb.get("error", fc.get("error", "unknown")))[:500]
    (OUT / "p4a_parity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
