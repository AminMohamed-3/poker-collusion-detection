"""S08 - what does the generator mean by 'ahead' in the fold-type event? Necessity (share of fold-type listed
hands satisfying) and false-positive rate on confirmed non-events (y2) under alternative definitions:
  eq>t     : folder's enumerated equity against the partner's real cards at the fold street > t
  strgap   : folder's strength percentile (vs random) at the fold street > partner's
  best     : folder held the best hand among all live players at the fold
  pf169    : preflop only: folder's 169-class percentile > partner's
Also the distribution of the folder's equity on fold-to-partner hands: listed (fold-type) vs non-events.
"""
import polars as pl, numpy as np, json, os
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
lab = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet"))
df = base.join(lab.select(["pair_id", "hand_id", "y2", "y2x", "B1"] + [c for c in lab.columns if c.startswith("d_")]), on=["pair_id", "hand_id"])
# folder = the member who folded facing the partner as aggressor (A first)
fs = {"A": pl.col("t5_fold_street_A"), "B": pl.col("t5_fold_street_B")}
def pick(street_col, names):  # names: list per street index 0..3
    return pl.when(street_col == 0).then(pl.col(names[0])).when(street_col == 1).then(pl.col(names[1])).when(street_col == 2).then(pl.col(names[2])).when(street_col == 3).then(pl.col(names[3])).otherwise(None)
df = df.with_columns(
    folder=pl.when(pl.col("t1_ap_fold_aggressor_A") > 0).then(pl.lit("A")).when(pl.col("t1_ap_fold_aggressor_B") > 0).then(pl.lit("B")).otherwise(None))
df = df.with_columns(
    f_street=pl.when(pl.col("folder") == "A").then(fs["A"]).when(pl.col("folder") == "B").then(fs["B"]).otherwise(None))
strgapA = pick(pl.col("f_street"), ["t2_strgap_pre", "t2_strgap_flop", "t2_strgap_turn", "t2_strgap_river"])
eqA = pick(pl.col("f_street"), ["t2_eq_pre_A", "t2_eq_flop_A", "t2_eq_turn_A", "t2_eq_river_A"])
df = df.with_columns(
    f_strgap=pl.when(pl.col("folder") == "A").then(strgapA).when(pl.col("folder") == "B").then(-strgapA).otherwise(None),
    f_eq=pl.when(pl.col("folder") == "A").then(eqA).when(pl.col("folder") == "B").then(1 - eqA).otherwise(None),
    f_best=pl.when(pl.col("folder").is_not_null()).then(pl.col("t2_best_at_fold_sm") > 0).otherwise(None),
    f_pf_gap=pl.when(pl.col("folder") == "A").then(pl.col("t2_hole_pf169_A") - pl.col("t2_hole_pf169_B")).when(pl.col("folder") == "B").then(pl.col("t2_hole_pf169_B") - pl.col("t2_hole_pf169_A")).otherwise(None),
    f_rank=pl.col("t2_rank_at_fold_mn"),
)
FT = df.filter(pl.col("folder").is_not_null())  # fold-to-partner hands
out = {}
for fam in ("directed_transfer", "soft_play"):
    d = FT.filter(pl.col("family") == fam)
    # fold-type listed = listed hands with a fold to partner (block 0 determined, or ambiguous)
    pos = d.filter((pl.col("listed") == 1))
    pos0 = d.filter((pl.col("listed") == 1) & (pl.col("block") == 0))
    neg = d.filter(pl.col("y2") == 0)
    res = {"n_fold_to_partner_listed": pos.shape[0], "n_b0": pos0.shape[0], "n_nonevent_fold_to_partner": neg.shape[0],
           "nonevent_total": int((df.filter(pl.col("family") == fam)["y2"] == 0).sum())}
    defs = {}
    for name, expr in {"eq>.5": pl.col("f_eq") > 0.5, "eq>.45": pl.col("f_eq") > 0.45, "eq>.4": pl.col("f_eq") > 0.4, "eq>.35": pl.col("f_eq") > 0.35, "eq>.6": pl.col("f_eq") > 0.6,
                       "strgap>0": pl.col("f_strgap") > 0, "strgap>-.05": pl.col("f_strgap") > -0.05, "best_live": pl.col("f_best"), "rank<=2": pl.col("f_rank") <= 2,
                       "eq>.5|best": (pl.col("f_eq") > 0.5) | pl.col("f_best"), "eq>.4|best": (pl.col("f_eq") > 0.4) | pl.col("f_best"),
                       "eq>.5|strgap>0": (pl.col("f_eq") > 0.5) | (pl.col("f_strgap") > 0), "eq>.35|strgap>0": (pl.col("f_eq") > 0.35) | (pl.col("f_strgap") > 0)}.items():
        defs[name] = {"nec_b0": float(pos0.select(expr.cast(pl.Float64).mean()).item()), "nec_all_fold_listed": float(pos.select(expr.cast(pl.Float64).mean()).item()),
                      "fpr_within_fold_nonevents": float(neg.select(expr.cast(pl.Float64).mean()).item()),
                      "fpr_all_nonevents": float(neg.select(expr.cast(pl.Float64).sum()).item() / res["nonevent_total"])}
    res["defs"] = defs
    # equity histograms
    bins = [-0.001, .2, .3, .35, .4, .45, .5, .55, .6, .7, .8, .9, 1.001]
    res["eq_hist_b0"] = np.histogram(pos0["f_eq"].fill_null(-1).to_numpy(), bins=bins)[0].tolist()
    res["eq_hist_nonevent"] = np.histogram(neg["f_eq"].fill_null(-1).to_numpy(), bins=bins)[0].tolist()
    res["eq_bins"] = bins
    # per street
    res["b0_fold_street"] = pos0.group_by("f_street").len().sort("f_street").to_dicts()
    res["nonevent_fold_street"] = neg.group_by("f_street").len().sort("f_street").to_dicts()
    # the below-.5 block-0 hands: strgap and best flags
    lo = pos0.filter(pl.col("f_eq") <= 0.5)
    res["b0_eq_le_.5"] = {"n": lo.shape[0], "strgap>0": float((lo["f_strgap"] > 0).mean()) if lo.shape[0] else None, "best": float(lo["f_best"].mean()) if lo.shape[0] else None,
                          "streets": lo.group_by("f_street").len().sort("f_street").to_dicts(), "eq": lo["f_eq"].round(3).to_list(), "strgap": lo["f_strgap"].round(3).to_list()}
    out[fam] = res
    print(f"\n===== {fam}"); print(json.dumps(res, indent=1, default=str))
json.dump(out, open(os.path.join(OUT, "s08_ahead_definitions.json"), "w"), indent=1, default=str)
