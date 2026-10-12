"""S09 - the lists whose FIRST block holds showdown-type hands (reversed type order), printed hand by hand,
and the sub-type anatomy of listed hands: for every listed hand the crisp type, fold street, folder equity /
strength gap, pot, both-showdown, net chips, position in history. Purpose: is the block order a fixed
type order with exceptions, or a finer sub-type order?
"""
import polars as pl, numpy as np, json, os
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
lab = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet"))
df = base.join(lab.select(["pair_id", "hand_id", "B1"] + [c for c in lab.columns if c.startswith("d_")]), on=["pair_id", "hand_id"])
def pick(street_col, names):
    return pl.when(street_col == 0).then(pl.col(names[0])).when(street_col == 1).then(pl.col(names[1])).when(street_col == 2).then(pl.col(names[2])).when(street_col == 3).then(pl.col(names[3])).otherwise(None)
df = df.with_columns(folder=pl.when(pl.col("t1_ap_fold_aggressor_A") > 0).then(pl.lit("A")).when(pl.col("t1_ap_fold_aggressor_B") > 0).then(pl.lit("B")).otherwise(pl.lit("-")))
df = df.with_columns(f_street=pl.when(pl.col("folder") == "A").then(pl.col("t5_fold_street_A")).when(pl.col("folder") == "B").then(pl.col("t5_fold_street_B")).otherwise(-1))
sg = pick(pl.col("f_street"), ["t2_strgap_pre", "t2_strgap_flop", "t2_strgap_turn", "t2_strgap_river"])
eqA = pick(pl.col("f_street"), ["t2_eq_pre_A", "t2_eq_flop_A", "t2_eq_turn_A", "t2_eq_river_A"])
df = df.with_columns(f_strgap=pl.when(pl.col("folder") == "A").then(sg).when(pl.col("folder") == "B").then(-sg).otherwise(None),
                     f_eq=pl.when(pl.col("folder") == "A").then(eqA).when(pl.col("folder") == "B").then(1 - eqA).otherwise(None),
                     typ=pl.when(pl.col("t5_fold_to_partner_sm") > 0).then(pl.lit("F")).when(pl.col("t5_both_showdown") == 1).then(pl.lit("S")).when(pl.col("t1_ap_call_aggressor_sm") > 0).then(pl.lit("C")).otherwise(pl.lit("O")))
L = df.filter(pl.col("listed") == 1).sort(["pair_id", "evidence_rank"])
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(16); pl.Config.set_tbl_width_chars(250)
cols = ["pair_id", "evidence_rank", "block", "idx", "typ", "folder", "f_street", "f_eq", "f_strgap", "t5_pot_bb", "t5_both_showdown", "t5_players_at_showdown", "t5_net_bb_A", "t5_net_bb_B", "t1_ap_call_aggressor_A", "t1_ap_call_aggressor_B", "t2_str_river_A", "t2_str_river_B", "t1_a_pre_raise_sm", "t5_vpip_sm"]
for fam in ("directed_transfer", "soft_play"):
    Lf = L.filter(pl.col("family") == fam)
    # pairs where block 0 has an S/C/O-type hand
    pids = Lf.filter((pl.col("block") == 0) & (pl.col("typ") != "F"))["pair_id"].unique().to_list()
    print(f"\n===== {fam}: {len(pids)} lists with a non-fold hand in the FIRST block")
    print(Lf.filter(pl.col("pair_id").is_in(pids)).select(cols))
    # SP: showdown-type block 0 vs block 1 -- who was strong, who was passive
    if fam == "soft_play":
        S = Lf.filter(pl.col("typ") == "S")
        print("SP showdown-type listed hands by block: strength at river of A/B, who bet (bbs aggressor), net")
        print(S.group_by("block").agg(pl.len(), pl.col("t2_str_river_A").mean().round(3), pl.col("t2_str_river_B").mean().round(3), pl.col("t1_bbs_aggressor_A").mean().round(2), pl.col("t1_bbs_aggressor_B").mean().round(2),
                                     (pl.col("t1_ap_call_aggressor_sm") > 0).mean().round(3).alias("call_vs_partner"), (pl.col("t1_bbs_aggressor_A") + pl.col("t1_bbs_aggressor_B") == 0).mean().round(3).alias("no_bet_between"),
                                     pl.col("t5_pot_bb").median().alias("pot_med"), (pl.col("t5_net_bb_A") < 0).mean().round(3).alias("A_loses")).sort("block"))
# all-list type strings by block for the record
rec = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    Lf = L.filter(pl.col("family") == fam)
    rec[fam] = {"typ_by_block": Lf.group_by(["block", "typ"]).len().sort(["block", "typ"]).to_dicts(),
                "f_street_by_block": Lf.filter(pl.col("typ") == "F").group_by(["block", "f_street"]).len().sort(["block", "f_street"]).to_dicts()}
    print(fam, json.dumps(rec[fam]))
json.dump(rec, open(os.path.join(OUT, "s09_reversed_lists.json"), "w"), indent=1, default=str)
