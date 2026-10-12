"""S15 - is there an active-window END? In capped pairs the region after the list is censored: if collusion ran to
the end of the shared history at a constant rate, event-shaped hands should be equally frequent in every part of
that region; if the window closed, the late part should fall back to the honest rate. Rates by position quintile
of the AFTER-LIST region, per family; plus the same for the whole history of uncapped pairs (complete lists).
"""
import polars as pl, numpy as np, json, os
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"), columns=["pair_id", "hand_id", "family", "frac", "idx", "n_hands", "listed", "t1_ap_call_aggressor_sm", "t5_both_showdown", "t5_transfer_bb_A2B", "t1_a_pre_raise_sm", "t1_wgss_pre_sm", "t3_bothlive_flop", "t5_net_bb_A"])
lab = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet"), columns=["pair_id", "hand_id", "y2", "n_listed", "t_us", "t_last", "d_fold_ahead_any"])
raw = pl.read_parquet(os.path.join(OUT, "raw_atoms.parquet"), columns=["pair_id", "hand_id", "r_n_out_fold_vs_mem"])
df = base.join(lab, on=["pair_id", "hand_id"]).join(raw, on=["pair_id", "hand_id"])
df = df.with_columns(e_fold_ahead=pl.col("d_fold_ahead_any") == 1, e_call_partner=pl.col("t1_ap_call_aggressor_sm") > 0, e_both_sd=pl.col("t5_both_showdown") == 1,
                     e_transfer20=pl.col("t5_transfer_bb_A2B") > 20, e_pre_raise_weak_outfold2=(pl.col("t1_a_pre_raise_sm") > 0) & (pl.col("t1_wgss_pre_sm") > .5) & (pl.col("r_n_out_fold_vs_mem") >= 2),
                     e_bothlive_flop=pl.col("t3_bothlive_flop") == 1)
ev = [c for c in df.columns if c.startswith("e_")]
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20)
out = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    d = df.filter((pl.col("family") == fam) & (pl.col("n_listed") == 5) & (pl.col("listed") == 0) & (pl.col("t_us") > pl.col("t_last")))
    # position within the after-list region, per pair: 0 = just after the list, 1 = last shared hand
    d = d.with_columns(pos_in_after=((pl.col("idx") - pl.col("idx").min().over("pair_id")) / (pl.col("idx").max().over("pair_id") - pl.col("idx").min().over("pair_id") + 1e-9)))
    d = d.with_columns(q=(pl.col("pos_in_after") * 5).floor().clip(0, 4).cast(pl.Int8))
    t = d.group_by("q").agg([pl.len().alias("n")] + [(pl.col(c).cast(pl.Float64).mean() * 100).round(2).alias(c[2:] + "_%") for c in ev]).sort("q")
    print(f"\n===== {fam}: after-list region by position quintile (0 = right after the list, 4 = end of history)"); print(t)
    out[fam + "_after_by_quintile"] = t.to_dicts()
    # last 10 shared hands vs the rest of the after region
    d2 = d.with_columns(last10=(pl.col("n_hands") - pl.col("idx")) <= 10)
    t2 = d2.group_by("last10").agg([pl.len().alias("n")] + [(pl.col(c).cast(pl.Float64).mean() * 100).round(2).alias(c[2:] + "_%") for c in ev]).sort("last10")
    print("last 10 hands of the history vs earlier after-list hands:"); print(t2)
    out[fam + "_last10"] = t2.to_dicts()
    # whole history of uncapped pairs by quintile of frac (complete lists): where do the listed hands sit
    u = df.filter((pl.col("family") == fam) & (pl.col("n_listed") < 5)).with_columns(q=(pl.col("frac") * 5).floor().clip(0, 4).cast(pl.Int8))
    if u.shape[0]:
        t3 = u.group_by("q").agg(pl.len().alias("n"), (pl.col("listed").mean() * 100).round(2).alias("listed_%")).sort("q")
        print("uncapped pairs: listed share by history quintile:"); print(t3)
        out[fam + "_uncapped_listed_by_quintile"] = t3.to_dicts()
json.dump(out, open(os.path.join(OUT, "s15_window_end.json"), "w"), indent=1, default=str)
