"""S13 - two region-free checks.
(1) The 32 UNCAPPED pairs (3 or 4 listed): the marked set is complete, so every unlisted hand is a non-event with no
    assumption. Count unlisted fold-to-partner-while-ahead hands per pair at several equity levels, and unlisted
    call-vs-partner / both-showdown hands (second-block type). Also: are the listed positions uniform over the history?
(2) High-equity (>.8) fold-to-partner hands that are confirmed non-events (y2==0): printed with context.
"""
import polars as pl, numpy as np, json, os
from scipy import stats
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
lab = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet"))
raw = pl.read_parquet(os.path.join(OUT, "raw_atoms.parquet"))
df = base.join(lab.select(["pair_id", "hand_id", "y2", "n_listed", "B1"] + [c for c in lab.columns if c.startswith("d_")]), on=["pair_id", "hand_id"]).join(raw, on=["pair_id", "hand_id"])
def pick(street_col, names):
    return pl.when(street_col == 0).then(pl.col(names[0])).when(street_col == 1).then(pl.col(names[1])).when(street_col == 2).then(pl.col(names[2])).when(street_col == 3).then(pl.col(names[3])).otherwise(None)
df = df.with_columns(folder=pl.when(pl.col("t1_ap_fold_aggressor_A") > 0).then(pl.lit("A")).when(pl.col("t1_ap_fold_aggressor_B") > 0).then(pl.lit("B")).otherwise(None))
df = df.with_columns(f_street=pl.when(pl.col("folder") == "A").then(pl.col("t5_fold_street_A")).when(pl.col("folder") == "B").then(pl.col("t5_fold_street_B")).otherwise(None))
sg = pick(pl.col("f_street"), ["t2_strgap_pre", "t2_strgap_flop", "t2_strgap_turn", "t2_strgap_river"])
eqA = pick(pl.col("f_street"), ["t2_eq_pre_A", "t2_eq_flop_A", "t2_eq_turn_A", "t2_eq_river_A"])
isA = pl.col("folder") == "A"
df = df.with_columns(f_strgap=pl.when(isA).then(sg).otherwise(-sg), f_eq=pl.when(isA).then(eqA).otherwise(1 - eqA),
                     f_is_loser=isA.cast(pl.Int8), f_called=pl.when(isA).then(pl.col("t1_ap_call_aggressor_A")).otherwise(pl.col("t1_ap_call_aggressor_B")),
                     f_contrib=pl.when(isA).then(pl.col("t5_contrib_bb_A")).otherwise(pl.col("t5_contrib_bb_B")))
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(16); pl.Config.set_tbl_width_chars(250)
out = {}
# ---------- (1) uncapped pairs
U = df.filter(pl.col("n_listed") < 5)
rows = []
for pid, g in U.group_by("pair_id", maintain_order=True):
    pid = pid[0] if isinstance(pid, tuple) else pid
    unl = g.filter(pl.col("listed") == 0); lst = g.filter(pl.col("listed") == 1)
    fa = unl.filter(pl.col("folder").is_not_null())
    rows.append({"pair_id": pid, "family": g["family"][0], "n_hands": g.shape[0], "n_listed": lst.shape[0],
                 "L_fold_to_partner": int((lst["folder"].is_not_null()).sum()), "L_fold_ahead_eq.5": int(((lst["folder"].is_not_null()) & (lst["f_eq"] > .5)).sum()),
                 "L_call_partner": int((lst["t1_ap_call_aggressor_sm"] > 0).sum()), "L_both_sd": int(lst["t5_both_showdown"].sum()),
                 "U_fold_to_partner": fa.shape[0], "U_fold_strgap>0": int((fa["f_strgap"] > 0).sum()), "U_fold_eq>.5": int((fa["f_eq"] > .5).sum()), "U_fold_eq>.7": int((fa["f_eq"] > .7).sum()), "U_fold_eq>.9": int((fa["f_eq"] > .9).sum()),
                 "U_fold_eq>.5_postflop": int(((fa["f_eq"] > .5) & (fa["f_street"] >= 1)).sum()), "U_fold_eq>.5_loser_called": int(((fa["f_eq"] > .5) & (fa["f_is_loser"] == 1) & (fa["f_called"] > 0)).sum()),
                 "U_call_partner": int((unl["t1_ap_call_aggressor_sm"] > 0).sum()), "U_call_partner_A": int((unl["t1_ap_call_aggressor_A"] > 0).sum()), "U_both_sd": int(unl["t5_both_showdown"].sum()),
                 "U_both_sd_A_loses": int(((unl["t5_both_showdown"] == 1) & (unl["t5_net_bb_A"] < 0)).sum()), "U_transfer_A2B>5bb": int((unl["t5_transfer_bb_A2B"] > 5).sum()),
                 "U_pair_pre_raise_outfold2": int(((unl["d_pair_pre_raise"] == 1) & (unl["r_n_out_fold_vs_mem"] >= 2)).sum()),
                 "listed_frac": lst["frac"].round(2).to_list()})
R = pl.DataFrame(rows).sort(["family", "pair_id"])
print("===== UNCAPPED pairs (complete marked sets): unlisted hands satisfying the event predicates")
print(R.drop("listed_frac"))
agg = R.group_by("family").agg(pl.len().alias("pairs"), pl.col("n_listed").sum(), pl.col("L_fold_ahead_eq.5").sum(), pl.col("U_fold_to_partner").sum(), pl.col("U_fold_strgap>0").sum(), pl.col("U_fold_eq>.5").sum(), pl.col("U_fold_eq>.7").sum(), pl.col("U_fold_eq>.9").sum(),
                                pl.col("U_fold_eq>.5_postflop").sum(), pl.col("U_fold_eq>.5_loser_called").sum(), pl.col("U_call_partner").sum(), pl.col("U_both_sd").sum(), pl.col("U_both_sd_A_loses").sum(), pl.col("U_transfer_A2B>5bb").sum())
print(agg)
out["uncapped"] = R.to_dicts(); out["uncapped_agg"] = agg.to_dicts()
# uniformity of listed positions in uncapped pairs
fr = np.array([f for r in rows for f in r["listed_frac"]])
ks = stats.kstest(fr, "uniform")
out["uncapped_listed_frac_ks"] = {"n": int(len(fr)), "stat": float(ks.statistic), "p": float(ks.pvalue), "mean": float(fr.mean()), "q": [float(np.quantile(fr, q)) for q in (.1, .25, .5, .75, .9)]}
print("uncapped listed positions (fraction of history) uniformity KS:", out["uncapped_listed_frac_ks"])
# ---------- (2) high-equity fold-to-partner confirmed non-events
H = df.filter((pl.col("y2") == 0) & pl.col("folder").is_not_null() & (pl.col("f_eq") > 0.8) & pl.col("family").is_in(["directed_transfer", "soft_play"]))
cols = ["pair_id", "family", "n_listed", "idx", "n_hands", "folder", "f_street", "f_eq", "f_strgap", "f_called", "f_contrib", "t5_pot_bb", "t5_transfer_bb", "t3_live_flop", "r_n_out_vpip", "t1_ao_fold_outagg_sm", "t1_ao_fold_outnoagg_sm", "t1_fl_allin_1_sm", "t1_fl_overbet_1_sm", "t5_fold_to_outsider_sm", "r_out_raise_over_mem", "t1_bbs_aggressor_A", "t1_bbs_aggressor_B"]
print(f"\n===== confirmed NON-EVENT fold-to-partner hands with folder equity > .8: n={H.shape[0]}")
print(H.select(cols).sort("family", "f_eq", descending=[False, True]).head(60))
out["high_eq_nonevents"] = {"n": H.shape[0], "by_family": H.group_by("family").len().to_dicts(), "by_street": H.group_by("f_street").len().sort("f_street").to_dicts(),
                            "loser_is_folder": float(H["f_is_loser"].mean()), "called_before": float((H["f_called"] > 0).mean()), "in_uncapped_pairs": int((H["n_listed"] < 5).sum()),
                            "outsider_raised_over_member": float((H["r_out_raise_over_mem"] > 0).mean()), "allin_faced": float((H["t1_fl_allin_1_sm"] > 0).mean())}
# compare with listed b0 high-eq folds
Lh = df.filter((pl.col("listed") == 1) & (pl.col("block") == 0) & pl.col("folder").is_not_null() & (pl.col("f_eq") > 0.8) & pl.col("family").is_in(["directed_transfer", "soft_play"]))
out["high_eq_listed_b0"] = {"n": Lh.shape[0], "loser_is_folder": float(Lh["f_is_loser"].mean()), "called_before": float((Lh["f_called"] > 0).mean()),
                            "outsider_raised_over_member": float((Lh["r_out_raise_over_mem"] > 0).mean()), "allin_faced": float((Lh["t1_fl_allin_1_sm"] > 0).mean()),
                            "by_street": Lh.group_by("f_street").len().sort("f_street").to_dicts()}
print(json.dumps({"high_eq_nonevents": out["high_eq_nonevents"], "high_eq_listed_b0": out["high_eq_listed_b0"]}, indent=1, default=str))
json.dump(out, open(os.path.join(OUT, "s13_uncapped_highfold.json"), "w"), indent=1, default=str)
