"""S04 - what do the listed hands look like structurally, against the confirmed non-events?
Shares per family and per target (b0 / b1 / ambiguous / non-event) of crisp structural facts. The purpose is to
find listed hands with NO observable member interaction (the annotator marked a hand in which nothing between
the two members happened) -- those bound any rule's recall from above.
"""
import polars as pl, numpy as np, json, os
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
lab = pl.read_parquet(os.path.join(OUT, "labels_derived.parquet"))
raw = pl.read_parquet(os.path.join(OUT, "raw_atoms.parquet"))
df = base.join(lab.select(["pair_id", "hand_id", "y"] + [c for c in lab.columns if c.startswith("d_")]), on=["pair_id", "hand_id"]).join(raw, on=["pair_id", "hand_id"])
df = df.with_columns(
    f_both_vpip=pl.col("d_both_vpip") == 1,
    f_any_vpip=pl.col("t5_vpip_sm") >= 1,
    f_bothlive_flop=pl.col("t3_bothlive_flop") == 1,
    f_bothlive_turn=pl.col("t3_bothlive_turn") == 1,
    f_fold_to_partner=pl.col("d_fold_to_partner_any") == 1,
    f_fold_ahead=pl.col("d_fold_ahead_any") == 1,
    f_call_vs_partner=pl.col("d_call_vs_partner_any") == 1,
    f_raise_over_partner=pl.col("d_raise_over_partner_any") == 1,
    f_both_showdown=pl.col("t5_both_showdown") == 1,
    f_transfer=pl.col("t5_transfer_bb") > 0,
    f_transfer_A2B=pl.col("t5_transfer_bb_A2B") > 0,
    f_transfer_B2A=pl.col("t5_transfer_bb_A2B") < 0,
    f_A_wins=pl.col("t5_net_bb_A") > 0,
    f_B_wins=pl.col("t5_net_bb_B") > 0,
    f_outsider_won=pl.col("t5_outsider_won") == 1,
    f_pair_alone_end=pl.col("r_pair_alone_end") == 1,
    f_out_fold_vs_mem=pl.col("r_n_out_fold_vs_mem") >= 1,
    f_out_fold_vs_mem2=pl.col("r_n_out_fold_vs_mem") >= 2,
    f_first_aggr_mem=pl.col("r_first_aggr_mem") == 1,
    f_pair_reraise=pl.col("r_pair_reraise") >= 1,
    f_mem_fold_after_outs_out=pl.col("r_mem_fold_after_outs_out") >= 1,
    f_pair_pre_raise=pl.col("d_pair_pre_raise") == 1,
    f_both_pre_nofold=pl.col("d_both_pre_nofold") == 1,
    f_net_opposite=pl.col("r_net_sign_opposite") == 1,
    f_pot_ge20bb=pl.col("t5_pot_bb") >= 20,
    f_pot_ge50bb=pl.col("t5_pot_bb") >= 50,
    f_partner_interaction=(pl.col("t1_ap_fold_aggressor_sm") + pl.col("t1_ap_call_aggressor_sm") + pl.col("t1_ap_raise_aggressor_sm")
                           + pl.col("t1_ap_check_passive_sm") + pl.col("t1_ap_bet_passive_sm") + pl.col("t1_ap_raise_passive_sm")
                           + pl.col("t1_ap_call_passive_sm") + pl.col("t1_ap_fold_passive_sm")) >= 1,
)
facts = [c for c in df.columns if c.startswith("f_")]
df = df.with_columns(grp=pl.when(pl.col("listed") == 1).then(pl.when(pl.col("block") == 0).then(pl.lit("L_b0")).when(pl.col("block") == 1).then(pl.lit("L_b1")).otherwise(pl.lit("L_amb")))
                     .when(pl.col("y") == 0).then(pl.lit("nonevent")).otherwise(pl.lit("unknown")))
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30)
out = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    d = df.filter(pl.col("family") == fam)
    t = d.group_by("grp").agg([pl.len().alias("n")] + [pl.col(f).cast(pl.Float64).mean().round(3).alias(f[2:]) for f in facts]).sort("grp")
    print(f"\n===== {fam}")
    tt = t.transpose(include_header=True, header_name="fact", column_names=t["grp"].to_list())
    print(tt)
    out[fam] = tt.to_dicts()
    # listed hands with no member-member interaction at all
    L = d.filter(pl.col("listed") == 1)
    noint = L.filter(~pl.col("f_partner_interaction") & ~pl.col("f_both_showdown"))
    out[fam + "_listed_no_interaction"] = {"n": noint.shape[0], "share": round(noint.shape[0] / L.shape[0], 4),
                                           "by_block": noint.group_by("block").len().sort("block").to_dicts()}
    print("listed with no member-member interaction and no both-showdown:", out[fam + "_listed_no_interaction"])
    print(noint.select("pair_id", "hand_id", "evidence_rank", "block", "t5_vpip_sm", "t5_fold_street_A", "t5_fold_street_B", "t3_bothlive_flop", "t5_net_bb_A", "t5_net_bb_B", "t5_pot_bb", "r_n_out_fold_vs_mem").head(12))
json.dump(out, open(os.path.join(OUT, "s04_interaction_facts.json"), "w"), indent=1, default=str)
