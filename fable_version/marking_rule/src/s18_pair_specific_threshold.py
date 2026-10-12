"""S18 - could the marking be deterministic with a PAIR-SPECIFIC parameter? E.g. each pair marks every fold-ahead
whose folder equity (or transfer, or model score) exceeds its own threshold. Test: within each pair, among the
fold-ahead hands (listed first-block vs confirmed non-event), does a perfect separating threshold exist on the
candidate scalar? Compare the share of separable pairs with the share expected when labels are permuted within
the pair (same counts). If the generator used a pair threshold, separability would be ~1.0 on the right scalar.
"""
import polars as pl, numpy as np, json, os
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
lab = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet")).select("pair_id", "hand_id", "y2", "d_fold_ahead_any", "d_eq_at_fold_max", "d_call_vs_partner_any")
oof = pl.read_parquet(os.path.join(OUT, "s03_oof_mech_v2.parquet"))
scol = ["oof_mech"]; oof = oof.select("pair_id", "hand_id", "oof_mech")
d = base.join(lab, on=["pair_id", "hand_id"]).join(oof, on=["pair_id", "hand_id"], how="left")
rng = np.random.default_rng(0)
def separable(v, y):
    v = np.asarray(v, float); y = np.asarray(y, int)
    return bool(v[y == 1].min() > v[y == 0].max())
def run(fam, subset_expr, scalars, label):
    sub = d.filter((pl.col("family") == fam) & subset_expr & pl.col("y2").is_not_null())
    res = {}
    for sc in scalars:
        n = 0; sep = 0; null = []
        for pid, g in sub.group_by("pair_id"):
            y = g["y2"].to_numpy(); v = g[sc].to_numpy()
            if (y == 1).sum() == 0 or (y == 0).sum() == 0 or np.isnan(v).any():
                continue
            n += 1; sep += separable(v, y)
            null.append(np.mean([separable(v, rng.permutation(y)) for _ in range(50)]))
        res[sc] = {"pairs": n, "separable_share": sep / max(1, n), "null_share": float(np.mean(null)) if null else None}
        print(f"  {label:22s} {sc:26s} pairs {n:3d}  separable {sep / max(1, n):.3f}  permutation null {np.mean(null) if null else float('nan'):.3f}")
    return res
out = {}
for fam in ("directed_transfer", "soft_play"):
    print(f"\n===== {fam}")
    out[fam] = {}
    out[fam]["foldahead_b0"] = run(fam, (pl.col("d_fold_ahead_any") == 1) & ((pl.col("block") == 0) | (pl.col("y2") == 0)), ["d_eq_at_fold_max", "t5_transfer_bb", "t5_pot_bb", "t2_strgap_flop"] + scol[:1], "fold-ahead, first block")
    out[fam]["callpartner_b1"] = run(fam, (pl.col("d_call_vs_partner_any") == 1) & ((pl.col("block") == 1) | (pl.col("y2") == 0)), ["t5_transfer_bb", "t5_pot_bb", "t5_net_bb_mx"] + scol[:1], "call-partner, second block")
    out[fam]["all_listed"] = run(fam, pl.lit(True), scol[:1], "all hands, model score")
print("\ncoordinated_isolation")
out["coordinated_isolation"] = {"all_listed": run("coordinated_isolation", pl.lit(True), scol[:1], "all hands, model score")}
json.dump(out, open(os.path.join(OUT, "s18_pair_specific_threshold.json"), "w"), indent=1)
