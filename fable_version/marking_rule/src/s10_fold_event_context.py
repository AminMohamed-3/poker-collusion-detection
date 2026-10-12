"""S10 - sufficiency of the fold-while-ahead event: among hands where a member folded facing the partner as
aggressor while ahead by strength (strgap > 0), what separates the LISTED first-block hands from the CONFIRMED
non-events? Tabulate by street, outsider state at the fold, pot size, folder contribution, whether the folder
called earlier, the partner's bet size, both-VPIP; then a shallow cross-validated tree on those context atoms.
"""
import polars as pl, numpy as np, json, os
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold, cross_val_predict
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
lab = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet"))
raw = pl.read_parquet(os.path.join(OUT, "raw_atoms.parquet"))
df = base.join(lab.select(["pair_id", "hand_id", "y2", "y2x", "B1"] + [c for c in lab.columns if c.startswith("d_")]), on=["pair_id", "hand_id"]).join(raw, on=["pair_id", "hand_id"])
def pick(street_col, names):
    return pl.when(street_col == 0).then(pl.col(names[0])).when(street_col == 1).then(pl.col(names[1])).when(street_col == 2).then(pl.col(names[2])).when(street_col == 3).then(pl.col(names[3])).otherwise(None)
df = df.with_columns(folder=pl.when(pl.col("t1_ap_fold_aggressor_A") > 0).then(pl.lit("A")).when(pl.col("t1_ap_fold_aggressor_B") > 0).then(pl.lit("B")).otherwise(None))
df = df.with_columns(f_street=pl.when(pl.col("folder") == "A").then(pl.col("t5_fold_street_A")).when(pl.col("folder") == "B").then(pl.col("t5_fold_street_B")).otherwise(None))
sg = pick(pl.col("f_street"), ["t2_strgap_pre", "t2_strgap_flop", "t2_strgap_turn", "t2_strgap_river"])
eqA = pick(pl.col("f_street"), ["t2_eq_pre_A", "t2_eq_flop_A", "t2_eq_turn_A", "t2_eq_river_A"])
live = pick(pl.col("f_street"), ["t3_live_flop", "t3_live_flop", "t3_live_turn", "t3_live_river"])
strA = pick(pl.col("f_street"), ["t2_hole_pf169_A", "t2_str_flop_A", "t2_str_turn_A", "t2_str_river_A"])
strB = pick(pl.col("f_street"), ["t2_hole_pf169_B", "t2_str_flop_B", "t2_str_turn_B", "t2_str_river_B"])
isA = pl.col("folder") == "A"
df = df.with_columns(
    f_strgap=pl.when(isA).then(sg).otherwise(-sg), f_eq=pl.when(isA).then(eqA).otherwise(1 - eqA),
    f_str=pl.when(isA).then(strA).otherwise(strB), p_str=pl.when(isA).then(strB).otherwise(strA),
    f_contrib=pl.when(isA).then(pl.col("t5_contrib_bb_A")).otherwise(pl.col("t5_contrib_bb_B")),
    p_contrib=pl.when(isA).then(pl.col("t5_contrib_bb_B")).otherwise(pl.col("t5_contrib_bb_A")),
    f_called_partner=pl.when(isA).then(pl.col("t1_ap_call_aggressor_A")).otherwise(pl.col("t1_ap_call_aggressor_B")),
    p_agg_bb=pl.when(isA).then(pl.col("t1_bbs_aggressor_B")).otherwise(pl.col("t1_bbs_aggressor_A")),
    p_pre_raise=pl.when(isA).then(pl.col("t1_a_pre_raise_mx")).otherwise(pl.col("t1_a_pre_raise_mx")),
    f_pos=pl.when(isA).then(pl.col("t3_pos_A")).otherwise(pl.col("t3_pos_B")),
    f_blind=pl.when(isA).then(pl.col("t3_A_blind")).otherwise(pl.col("t3_B_blind")),
    out_noout=pl.col("t1_ao_fold_noout_sm"), out_outnoagg=pl.col("t1_ao_fold_outnoagg_sm"), out_outagg=pl.col("t1_ao_fold_outagg_sm"),
    n_live_at_street=live, outs_folded_to_mem=pl.col("r_n_out_fold_vs_mem"), outs_vpip=pl.col("r_n_out_vpip"),
    f_is_loser=isA.cast(pl.Int8),
)
FA = df.filter(pl.col("folder").is_not_null() & (pl.col("f_strgap") > 0))
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(24); pl.Config.set_tbl_width_chars(250)
ctx = ["f_street", "f_is_loser", "f_blind", "f_pos", "f_eq", "f_str", "p_str", "f_strgap", "t5_pot_bb", "f_contrib", "p_contrib", "f_called_partner", "p_agg_bb", "n_live_at_street", "out_noout", "out_outnoagg", "out_outagg",
       "outs_folded_to_mem", "outs_vpip", "d_both_vpip", "t5_vpip_sm", "r_pair_alone_end", "r_mem_fold_after_outs_out", "t3_bothlive_flop", "t2_n_streets", "t5_transfer_bb", "r_first_aggr_mem"]
out = {}
for fam in ("directed_transfer", "soft_play"):
    d = FA.filter(pl.col("family") == fam)
    pos = d.filter((pl.col("listed") == 1) & (pl.col("block") == 0)); posall = d.filter(pl.col("listed") == 1); neg = d.filter(pl.col("y2") == 0)
    print(f"\n===== {fam}: fold-to-partner while ahead (strgap>0): listed b0 {pos.shape[0]}, listed any {posall.shape[0]}, confirmed non-events {neg.shape[0]}, unknown {d.filter(pl.col('y2').is_null() & (pl.col('listed')==0)).shape[0]}")
    t = pl.concat([pos.select(ctx).with_columns(g=pl.lit("L_b0")), neg.select(ctx).with_columns(g=pl.lit("nonevent"))])
    m = t.group_by("g").agg([pl.len().alias("n")] + [pl.col(c).cast(pl.Float64).mean().round(3).alias(c) for c in ctx]).sort("g")
    print(m.transpose(include_header=True, header_name="ctx", column_names=m["g"].to_list()))
    # by street x loser
    print("counts by fold street (listed b0 / nonevent):")
    print(pos.group_by("f_street").len().sort("f_street").join(neg.group_by("f_street").len().sort("f_street"), on="f_street", how="full", suffix="_neg"))
    print("by street x folder is loser (A):")
    print(pos.group_by("f_street", "f_is_loser").len().sort("f_street", "f_is_loser").join(neg.group_by("f_street", "f_is_loser").len(), on=["f_street", "f_is_loser"], how="full", suffix="_neg").sort("f_street", "f_is_loser"))
    # shallow tree, cross-validated by table
    X = t.select(ctx).to_numpy().astype(float); y = (t["g"] == "L_b0").to_numpy().astype(int)
    groups = pl.concat([pos.select("table_id"), neg.select("table_id")])["table_id"].to_numpy()
    for depth in (2, 3, 4):
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=15, random_state=0)
        p = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
        pred = p > 0.5
        rec = float(pred[y == 1].mean()); fpr = float(pred[y == 0].mean()); prec = float(y[pred].mean()) if pred.sum() else 0
        print(f"tree depth {depth}: OOF recall {rec:.3f} fpr-within-foldahead {fpr:.3f} precision {prec:.3f}")
        out[f"{fam}_tree_d{depth}"] = {"rec": rec, "fpr": fpr, "prec": prec}
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=15, random_state=0).fit(X, y)
    print(export_text(clf, feature_names=ctx, max_depth=3))
    out[fam + "_means"] = m.to_dicts()
json.dump(out, open(os.path.join(OUT, "s10_fold_event_context.json"), "w"), indent=1, default=str)
