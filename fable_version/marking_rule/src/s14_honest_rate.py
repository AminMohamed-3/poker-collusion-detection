"""S14 - are the unmarked fold-while-ahead hands NATURAL? Compare the per-hand rate of event-shaped hands in
  (a) honest pairs (confirmed_non_target, 1,488 pairs, all their shared hands),
  (b) positive pairs' CONFIRMED non-event region (y2 == 0), split by how the region was established,
  (c) positive pairs' unknown region (after the list, censored),
  (d) listed hands.
If (b) ~ (a), the unmarked fold-aheads are what honest players do and the marking is a random subset of the
opportunities; if (b) >> (a), the confirmed region holds marked-but-unlisted hands (the listing model is incomplete)
or collusion changes play outside the marked hands. Uses the STEP254 columns for the negative pairs (read-only).
"""
import polars as pl, numpy as np, json, os
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
S254 = os.path.join(ROOT, "Claude Version", "experiments", "step254_mechanical_tensor_20260917")
cols = ["pair_id", "hand_id", "t1_ap_fold_aggressor_A", "t1_ap_fold_aggressor_B", "t5_fold_street_A", "t5_fold_street_B", "t2_eq_pre_A", "t2_eq_flop_A", "t2_eq_turn_A", "t2_eq_river_A",
        "t2_strgap_pre", "t2_strgap_flop", "t2_strgap_turn", "t2_strgap_river", "t1_ap_call_aggressor_sm", "t1_ap_call_aggressor_A", "t5_both_showdown", "t5_net_bb_A", "t5_transfer_bb", "t5_transfer_bb_A2B",
        "t1_a_pre_raise_sm", "t1_wgss_pre_sm", "t3_bothlive_flop", "t5_vpip_sm", "t5_pot_bb"]
feat = pl.read_parquet(os.path.join(S254, "step254_features.parquet"), columns=cols)
oof = pl.read_parquet(os.path.join(S254, "hand_oof_step254.parquet"), columns=["pair_id", "hand_id", "label", "family", "listed"])
hands = pl.read_parquet(os.path.join(ROOT, "data", "hands.parquet"), columns=["hand_id", "phase"])
f = feat.join(oof, on=["pair_id", "hand_id"]).join(hands, on="hand_id")
assert (f["phase"] == "development").all()
lab = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet")).select("pair_id", "hand_id", "y2", "n_listed", "n_desc", "k0_hat", "t_us", "t_last", "listed")
f = f.join(lab.drop("listed"), on=["pair_id", "hand_id"], how="left")
def pick(street_col, names):
    return pl.when(street_col == 0).then(pl.col(names[0])).when(street_col == 1).then(pl.col(names[1])).when(street_col == 2).then(pl.col(names[2])).when(street_col == 3).then(pl.col(names[3])).otherwise(None)
f = f.with_columns(folder=pl.when(pl.col("t1_ap_fold_aggressor_A") > 0).then(pl.lit("A")).when(pl.col("t1_ap_fold_aggressor_B") > 0).then(pl.lit("B")).otherwise(None))
f = f.with_columns(f_street=pl.when(pl.col("folder") == "A").then(pl.col("t5_fold_street_A")).when(pl.col("folder") == "B").then(pl.col("t5_fold_street_B")).otherwise(None))
sg = pick(pl.col("f_street"), ["t2_strgap_pre", "t2_strgap_flop", "t2_strgap_turn", "t2_strgap_river"])
eqA = pick(pl.col("f_street"), ["t2_eq_pre_A", "t2_eq_flop_A", "t2_eq_turn_A", "t2_eq_river_A"])
isA = pl.col("folder") == "A"
f = f.with_columns(f_strgap=pl.when(isA).then(sg).otherwise(-sg), f_eq=pl.when(isA).then(eqA).otherwise(1 - eqA))
f = f.with_columns(
    e_fold_to_partner=pl.col("folder").is_not_null(),
    e_fold_eq5=pl.col("folder").is_not_null() & (pl.col("f_eq") > .5),
    e_fold_eq7=pl.col("folder").is_not_null() & (pl.col("f_eq") > .7),
    e_fold_eq8_post=pl.col("folder").is_not_null() & (pl.col("f_eq") > .8) & (pl.col("f_street") >= 1),
    e_fold_strgap=pl.col("folder").is_not_null() & (pl.col("f_strgap") > 0),
    e_call_partner=pl.col("t1_ap_call_aggressor_sm") > 0,
    e_call_partner_A_loses=(pl.col("t1_ap_call_aggressor_A") > 0) & (pl.col("t5_net_bb_A") < 0),
    e_both_sd=pl.col("t5_both_showdown") == 1,
    e_transfer_A2B_20=pl.col("t5_transfer_bb_A2B") > 20,
    e_pair_pre_raise_weak=(pl.col("t1_a_pre_raise_sm") > 0) & (pl.col("t1_wgss_pre_sm") > 0.5),
    e_bothlive_flop=pl.col("t3_bothlive_flop") == 1,
    e_both_vpip=pl.col("t5_vpip_sm") == 2,
)
ev = [c for c in f.columns if c.startswith("e_")]
capped = pl.col("n_listed") == 5
f = f.with_columns(region=pl.when(pl.col("label") == 0).then(pl.lit("honest"))
                   .when(pl.col("listed") == 1).then(pl.lit("listed"))
                   .when(pl.col("y2") == 0).then(pl.when(~capped).then(pl.lit("pos_uncapped_unlisted")).when(pl.col("n_desc") == 1).then(pl.lit("pos_1desc_before_t5")).otherwise(pl.lit("pos_0desc_before_t5")))
                   .when(pl.col("y2").is_null() & (pl.col("listed") == 0)).then(pl.lit("pos_unknown_after")).otherwise(pl.lit("other")))
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(26); pl.Config.set_tbl_width_chars(260)
out = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    d = f.filter((pl.col("family") == fam) | (pl.col("label") == 0))
    t = d.group_by("region").agg([pl.len().alias("n")] + [(pl.col(c).cast(pl.Float64).mean() * 100).round(3).alias(c[2:] + "_%") for c in ev]).sort("region")
    print(f"\n===== {fam} (rates in % of hands; honest = all 1,488 confirmed non-target pairs)")
    print(t.transpose(include_header=True, header_name="event_shape", column_names=t["region"].to_list()))
    out[fam] = t.to_dicts()
# per-pair excess in the uncapped positive pairs vs honest pairs, for the fold-eq>.5 shape: distribution
hp = f.filter(pl.col("label") == 0).group_by("pair_id").agg(pl.len().alias("n"), pl.col("e_fold_eq5").sum().alias("k"))
print("honest pairs: fold-to-partner eq>.5 per 100 hands, quantiles:", {q: float((100 * hp["k"] / hp["n"]).quantile(q)) for q in (.1, .5, .9, .99)})
up = f.filter((pl.col("label") == 1) & (pl.col("n_listed") < 5) & (pl.col("listed") == 0)).group_by("pair_id").agg(pl.len().alias("n"), pl.col("e_fold_eq5").sum().alias("k"), pl.col("family").first())
print("uncapped positive pairs, UNLISTED hands: fold-to-partner eq>.5 per 100 hands:", (100 * up["k"] / up["n"]).round(2).to_list())
json.dump(out, open(os.path.join(OUT, "s14_honest_rate.json"), "w"), indent=1, default=str)
