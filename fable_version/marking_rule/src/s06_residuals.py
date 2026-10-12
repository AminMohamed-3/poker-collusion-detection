"""S06 - residual reads: listed hands that miss the obvious family predicate, and the anatomy of each block.
Listed hands only (no region assumption needed). Prints compact tables per family/block.
"""
import polars as pl, numpy as np, json, os
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
lab = pl.read_parquet(os.path.join(OUT, "labels_derived.parquet"))
raw = pl.read_parquet(os.path.join(OUT, "raw_atoms.parquet"))
df = base.join(lab.select(["pair_id", "hand_id", "y"] + [c for c in lab.columns if c.startswith("d_")]), on=["pair_id", "hand_id"]).join(raw, on=["pair_id", "hand_id"])
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(40); pl.Config.set_fmt_str_lengths(20); pl.Config.set_tbl_width_chars(260)
# equity of the folder vs partner at the fold street, per member, and rank among live at the fold
df = df.with_columns(
    folder=pl.when(pl.col("t1_ap_fold_aggressor_A") > 0).then(pl.lit("A")).when(pl.col("t1_ap_fold_aggressor_B") > 0).then(pl.lit("B")).otherwise(pl.lit("-")),
    eq_folder=pl.when(pl.col("t1_ap_fold_aggressor_A") > 0).then(pl.col("d_eq_at_fold_A")).when(pl.col("t1_ap_fold_aggressor_B") > 0).then(pl.col("d_eq_at_fold_B")).otherwise(-1.0),
)
L = df.filter(pl.col("listed") == 1)
out = {}
def show(name, d, cols, n=40):
    print(f"\n----- {name}: n={d.shape[0]}")
    print(d.select(cols).head(n))

# ---------- DT block 0 misses of fold-ahead
cols_fold = ["pair_id", "evidence_rank", "block", "folder", "eq_folder", "t5_fold_street_A", "t5_fold_street_B", "t5_fold_to_partner_sm", "t5_fold_to_outsider_sm", "t2_best_at_fold_sm", "t2_rank_at_fold_mn",
             "t5_vpip_sm", "t3_bothlive_flop", "t5_net_bb_A", "t5_net_bb_B", "t5_transfer_bb_A2B", "t5_pot_bb", "t5_both_showdown", "t1_ap_call_aggressor_sm", "r_n_out_fold_vs_mem", "r_pair_alone_end"]
for fam in ("directed_transfer", "soft_play"):
    b0 = L.filter((pl.col("family") == fam) & (pl.col("block") == 0))
    miss = b0.filter(pl.col("d_fold_ahead_any") == 0)
    show(f"{fam} block0 NOT fold-ahead", miss, cols_fold)
    out[f"{fam}_b0_miss_foldahead"] = {"n_b0": b0.shape[0], "n_miss": miss.shape[0],
                                       "miss_fold_to_partner": int((miss["t5_fold_to_partner_sm"] > 0).sum()),
                                       "miss_eq_hist": np.histogram(miss["eq_folder"].to_numpy(), bins=[-1.5, -0.5, 0.3, 0.4, 0.5, 0.6, 1.01])[0].tolist()}
    # distribution of folder equity among b0 fold-to-partner hands
    e = b0.filter(pl.col("eq_folder") >= 0)["eq_folder"].to_numpy()
    out[f"{fam}_b0_eq_quantiles"] = {q: float(np.quantile(e, q)) for q in (0, .02, .05, .1, .25, .5)}
    print(f"{fam} b0 folder equity quantiles:", out[f"{fam}_b0_eq_quantiles"])
    # b1 anatomy
    b1 = L.filter((pl.col("family") == fam) & (pl.col("block") == 1))
    show(f"{fam} block1 sample", b1, ["pair_id", "evidence_rank", "folder", "eq_folder", "t1_ap_call_aggressor_A", "t1_ap_call_aggressor_B", "t2_rank_at_call_mn", "t2_best_at_call_sm", "t5_both_showdown", "t5_players_at_showdown",
                                      "t5_net_bb_A", "t5_net_bb_B", "t5_transfer_bb_A2B", "t5_pot_bb", "t2_eq_river_A", "t2_eq_flop_A", "t5_fold_street_A", "t5_fold_street_B", "r_pair_alone_end", "t1_bbs_aggressor_B", "t1_bbs_aggressor_A"], n=30)
    out[f"{fam}_b1_facts"] = {"n": b1.shape[0], "call_vs_partner": float((b1["t1_ap_call_aggressor_sm"] > 0).mean()), "A_calls": float((b1["t1_ap_call_aggressor_A"] > 0).mean()), "B_calls": float((b1["t1_ap_call_aggressor_B"] > 0).mean()),
                              "both_showdown": float(b1["t5_both_showdown"].mean()), "A_loses": float((b1["t5_net_bb_A"] < 0).mean()), "transfer_A2B": float((b1["t5_transfer_bb_A2B"] > 0).mean()),
                              "pot_bb_q": {q: float(b1["t5_pot_bb"].quantile(q)) for q in (.05, .25, .5, .75)},
                              "A_fold_street": b1.group_by("t5_fold_street_A").len().sort("t5_fold_street_A").to_dicts(), "B_fold_street": b1.group_by("t5_fold_street_B").len().sort("t5_fold_street_B").to_dicts()}
    print(fam, "b1 facts:", json.dumps(out[f"{fam}_b1_facts"], default=str))

# ---------- CI anatomy
ci = L.filter(pl.col("family") == "coordinated_isolation")
show("CI listed sample", ci, ["pair_id", "evidence_rank", "block", "r_first_aggr_mem", "r_n_out_fold_vs_mem", "r_n_out_vpip", "r_n_out_call_mem_agg", "r_pair_reraise", "r_mem_raise_over_out", "r_out_raise_over_mem", "t1_a_pre_raise_A", "t1_a_pre_raise_B",
                              "t1_wgss_pre_A", "t1_wgss_pre_B", "t2_hole_pf169_A", "t2_hole_pf169_B", "folder", "eq_folder", "t5_fold_street_A", "t5_fold_street_B", "t5_net_bb_A", "t5_net_bb_B", "t5_pot_bb", "t5_outsider_won", "r_pair_alone_end"], n=40)
out["ci_facts"] = {"n": ci.shape[0], "pre_raise_A": float((ci["t1_a_pre_raise_A"] > 0).mean()), "pre_raise_B": float((ci["t1_a_pre_raise_B"] > 0).mean()), "both_pre_raise": float(((ci["t1_a_pre_raise_A"] > 0) & (ci["t1_a_pre_raise_B"] > 0)).mean()),
                   "first_aggr_mem": float(ci["r_first_aggr_mem"].mean()), "out_fold_vs_mem_ge1": float((ci["r_n_out_fold_vs_mem"] >= 1).mean()), "out_fold_vs_mem_ge2": float((ci["r_n_out_fold_vs_mem"] >= 2).mean()),
                   "pair_reraise": float((ci["r_pair_reraise"] >= 1).mean()), "mem_raise_over_out": float((ci["r_mem_raise_over_out"] >= 1).mean()), "n_out_vpip_dist": ci.group_by("r_n_out_vpip").len().sort("r_n_out_vpip").to_dicts(),
                   "fold_to_partner": float((ci["t5_fold_to_partner_sm"] > 0).mean()), "outsider_won": float(ci["t5_outsider_won"].mean()),
                   "wgss_pre_sm_q": {q: float(ci["t1_wgss_pre_sm"].quantile(q)) for q in (.05, .25, .5)},
                   "pf169_raiser_min": None}
print("CI facts:", json.dumps(out["ci_facts"], default=str))
# CI: the weakest preflop raiser's 169-class percentile
ci2 = ci.with_columns(pf_raiser=pl.when((pl.col("t1_a_pre_raise_A") > 0) & (pl.col("t1_a_pre_raise_B") > 0)).then(pl.min_horizontal("t2_hole_pf169_A", "t2_hole_pf169_B"))
                      .when(pl.col("t1_a_pre_raise_A") > 0).then(pl.col("t2_hole_pf169_A")).when(pl.col("t1_a_pre_raise_B") > 0).then(pl.col("t2_hole_pf169_B")).otherwise(None))
print("CI weakest preflop raiser pf169 quantiles:", {q: float(ci2["pf_raiser"].drop_nulls().quantile(q)) for q in (.05, .25, .5, .75, .95)})
ne = df.filter((pl.col("family") == "coordinated_isolation") & (pl.col("y") == 0) & (pl.col("d_pair_pre_raise") == 1))
ne2 = ne.with_columns(pf_raiser=pl.when((pl.col("t1_a_pre_raise_A") > 0) & (pl.col("t1_a_pre_raise_B") > 0)).then(pl.min_horizontal("t2_hole_pf169_A", "t2_hole_pf169_B"))
                      .when(pl.col("t1_a_pre_raise_A") > 0).then(pl.col("t2_hole_pf169_A")).otherwise(pl.col("t2_hole_pf169_B")))
print("CI NON-EVENT pair-preflop-raise hands: n", ne.shape[0], "raiser pf169 quantiles:", {q: float(ne2["pf_raiser"].quantile(q)) for q in (.05, .25, .5, .75, .95)})
json.dump(out, open(os.path.join(OUT, "s06_residuals.json"), "w"), indent=1, default=str)
