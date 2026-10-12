"""S17 - last exhaustive check: the soft-play SHOWDOWN-type listed hands. Is "passive strong showdown" (a member
holds the best hand at showdown and never bet/raised it) a crisp sub-type inside the block? Same test as before:
recall among listed showdown-type SP hands vs FPR among confirmed non-event SP showdown hands, for the candidate
sub-predicates and their conjunctions (depth <= 2, exhaustive over the small atom set).
"""
import polars as pl, numpy as np, json, os, itertools
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
lab = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet")).select("pair_id", "hand_id", "y2", "B1")
raw = pl.read_parquet(os.path.join(OUT, "raw_atoms.parquet"))
d = base.join(lab, on=["pair_id", "hand_id"]).join(raw, on=["pair_id", "hand_id"]).filter((pl.col("family") == "soft_play") & (pl.col("t5_both_showdown") == 1) & pl.col("y2").is_not_null())
# who won at showdown, who bet
d = d.with_columns(
    winner_A=pl.col("t5_net_bb_A") > pl.col("t5_net_bb_B"),
    bets_A=pl.col("t1_bbs_aggressor_A") > 0, bets_B=pl.col("t1_bbs_aggressor_B") > 0,
    river_str_A=pl.col("t2_str_river_A"), river_str_B=pl.col("t2_str_river_B"),
)
d = d.with_columns(
    winner_bets=pl.when(pl.col("winner_A")).then(pl.col("bets_A")).otherwise(pl.col("bets_B")),
    loser_bets=pl.when(pl.col("winner_A")).then(pl.col("bets_B")).otherwise(pl.col("bets_A")),
    winner_str=pl.when(pl.col("winner_A")).then(pl.col("river_str_A")).otherwise(pl.col("river_str_B")),
    loser_str=pl.when(pl.col("winner_A")).then(pl.col("river_str_B")).otherwise(pl.col("river_str_A")),
)
d = d.with_columns(
    a_passive_winner=~pl.col("winner_bets"),
    a_no_bets=(~pl.col("bets_A")) & (~pl.col("bets_B")),
    a_loser_bets=pl.col("loser_bets"),
    a_winner_strong=pl.col("winner_str") >= 0.7,
    a_both_vpip=pl.col("t5_vpip_sm") == 2,
    a_pot_ge10=pl.col("t5_pot_bb") >= 10,
    a_pot_ge20=pl.col("t5_pot_bb") >= 20,
    a_pair_alone=pl.col("r_pair_alone_end") == 1,
    a_call_partner=pl.col("t1_ap_call_aggressor_sm") > 0,
    a_no_raise_sm=(pl.col("t1_a_flop_raise_sm") + pl.col("t1_a_turn_raise_sm") + pl.col("t1_a_river_raise_sm")) == 0,
    a_transfer_pos=pl.col("t5_transfer_bb") > 0,
    a_bothlive_flop=pl.col("t3_bothlive_flop") == 1,
)
atoms = [c for c in d.columns if c.startswith("a_")]
y = d["y2"].to_numpy().astype(int)
print("SP showdown-type hands: listed", int((y == 1).sum()), "confirmed non-event", int((y == 0).sum()))
X = d.select(atoms).to_numpy().astype(bool)
rows = []
for k in (1, 2):
    for combo in itertools.combinations(range(len(atoms)), k):
        for signs in itertools.product((True, False), repeat=k):
            m = np.ones(len(y), dtype=bool)
            for j, s in zip(combo, signs):
                m &= (X[:, j] == s)
            rec = m[y == 1].mean(); fpr = m[y == 0].mean()
            rows.append((" & ".join(("" if s else "not ") + atoms[j] for j, s in zip(combo, signs)), float(rec), float(fpr), float(m[y == 1].sum() / max(1, m.sum()))))
df = pl.DataFrame(rows, schema=["rule", "rec", "fpr", "prec"], orient="row").sort("rec", descending=True)
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(60)
print("\nbest recall at fpr <= .10 among SP showdown hands:"); print(df.filter(pl.col("fpr") <= .10).head(12))
print("\nbest recall at fpr <= .25:"); print(df.filter(pl.col("fpr") <= .25).head(12))
print("\nhighest J:"); print(df.with_columns(J=pl.col("rec") - pl.col("fpr")).sort("J", descending=True).head(12))
# marginal rates
t = d.group_by("y2").agg([pl.len().alias("n")] + [pl.col(c).cast(pl.Float64).mean().round(3) for c in atoms]).sort("y2")
print("\nmarginal rates by y2:"); print(t.transpose(include_header=True, column_names=["nonevent", "listed"]))
json.dump({"n_listed": int((y == 1).sum()), "n_nonevent": int((y == 0).sum()), "top_fpr10": df.filter(pl.col("fpr") <= .10).head(12).to_dicts(), "top_J": df.with_columns(J=pl.col("rec") - pl.col("fpr")).sort("J", descending=True).head(12).to_dicts(), "marginals": t.to_dicts()}, open(os.path.join(OUT, "s17_sp_showdown_subtype.json"), "w"), indent=1)
