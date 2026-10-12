"""S02 - censoring-aware labels + univariate atom screen.

Labels (per pair-hand of the 372 positive pairs):
  y = 1  listed (an event, by construction)
  y = 0  confirmed non-event: unlisted AND (list uncapped (<5) OR started before the last-ranked listed hand)
         Under the traversal rule lexsort((block, time)) with cap 5, an unlisted hand earlier than the last
         listed hand of the SECOND block would have been listed whatever its block, so it is a non-event.
         For 0-descent capped lists the second block may be empty (then only block-0 unlisted hands are
         certain); this is flagged (region_assumed) and handled exactly later by the traversal simulation.
  y = NA unknown: unlisted, capped list, started after the last listed hand (censored by the cap)
Targets: any (all listed), b0 (listed with a determined block 0), b1 (determined block 1).
For every atom (662 mechanical + derived crisp atoms) and every threshold/direction: recall on the target,
false-positive rate on the confirmed non-events. Reported: best Youden J, and recall at FPR <= 1% / 0.2%.
"""
import polars as pl, numpy as np, json, os, sys
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
mech = [c for c in base.columns if c[:3] in ("t1_", "t2_", "t3_", "t4_", "t5_")]
assert len(mech) == 662

# ---------------- per-pair censor point
pp = (base.filter(pl.col("listed") == 1).group_by("pair_id")
      .agg(pl.col("evidence_rank").max().alias("n_listed"), pl.col("n_desc").max().alias("n_desc_p"),
           pl.col("t_us").filter(pl.col("evidence_rank") == pl.col("evidence_rank").max()).first().alias("t_last"),
           pl.col("t_us").filter(pl.col("block") == 0).max().alias("t_last_b0"),
           pl.col("t_us").filter(pl.col("block") == 1).max().alias("t_last_b1")))
base = base.join(pp, on="pair_id", how="left")
capped = pl.col("n_listed") == 5
y = (pl.when(pl.col("listed") == 1).then(1)
     .when(pl.col("n_desc_p") == 2).then(None)                       # 3 SP lists violate the 2-block rule: no region
     .when(~capped).then(0)
     .when(pl.col("t_us") < pl.col("t_last")).then(0)
     .otherwise(None))
base = base.with_columns(y=y.cast(pl.Int8),
                         region_assumed=((pl.col("listed") == 0) & capped & (pl.col("n_desc_p") == 0) & (pl.col("t_us") < pl.col("t_last"))).cast(pl.Int8))

# ---------------- derived crisp atoms
def eq_at_fold(member):
    fs = pl.col(f"t5_fold_street_{member}")
    eqA = [pl.col(f"t2_eq_{s}_A") for s in ("pre", "flop", "turn", "river")]
    e = pl.when(fs == 0).then(eqA[0]).when(fs == 1).then(eqA[1]).when(fs == 2).then(eqA[2]).when(fs == 3).then(eqA[3]).otherwise(-1.0)
    if member == "B":
        e = pl.when(e >= 0).then(1.0 - e).otherwise(-1.0)
    return e
base = base.with_columns(
    d_both_vpip=(pl.col("t5_vpip_sm") == 2).cast(pl.Int8),
    d_transfer_pos=(pl.col("t5_transfer_bb") > 0).cast(pl.Int8),
    d_transfer_A2B_pos=(pl.col("t5_transfer_bb_A2B") > 0).cast(pl.Int8),
    d_fold_to_partner_any=(pl.col("t5_fold_to_partner_sm") > 0).cast(pl.Int8),
    d_A_fold_vs_partner=(pl.col("t1_ap_fold_aggressor_A") > 0).cast(pl.Int8),
    d_B_fold_vs_partner=(pl.col("t1_ap_fold_aggressor_B") > 0).cast(pl.Int8),
    d_eq_at_fold_A=eq_at_fold("A"), d_eq_at_fold_B=eq_at_fold("B"),
    d_best_at_fold_any=(pl.col("t2_best_at_fold_sm") > 0).cast(pl.Int8),
    d_call_vs_partner_any=(pl.col("t1_ap_call_aggressor_sm") > 0).cast(pl.Int8),
    d_A_call_vs_partner=(pl.col("t1_ap_call_aggressor_A") > 0).cast(pl.Int8),
    d_raise_over_partner_any=(pl.col("t1_ap_raise_aggressor_sm") > 0).cast(pl.Int8),
    d_pair_pre_raise=(pl.col("t1_a_pre_raise_sm") > 0).cast(pl.Int8),
    d_both_pre_nofold=(pl.col("t1_a_pre_fold_sm") == 0).cast(pl.Int8),
)
base = base.with_columns(
    d_A_fold_ahead=((pl.col("d_A_fold_vs_partner") == 1) & (pl.col("d_eq_at_fold_A") > 0.5)).cast(pl.Int8),
    d_B_fold_ahead=((pl.col("d_B_fold_vs_partner") == 1) & (pl.col("d_eq_at_fold_B") > 0.5)).cast(pl.Int8),
    d_eq_at_fold_max=pl.max_horizontal("d_eq_at_fold_A", "d_eq_at_fold_B"),
)
base = base.with_columns(d_fold_ahead_any=pl.max_horizontal("d_A_fold_ahead", "d_B_fold_ahead"))
derived = [c for c in base.columns if c.startswith("d_")] + ["frac", "idx"]
base.select(["pair_id", "hand_id", "family", "table_id", "t_us", "idx", "n_hands", "frac", "listed", "evidence_rank", "block", "n_desc",
             "n_listed", "t_last", "y", "region_assumed"] + [c for c in base.columns if c.startswith("d_")]).write_parquet(os.path.join(OUT, "labels_derived.parquet"))

# ---------------- label summary
summ = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    b = base.filter(pl.col("family") == fam)
    summ[fam] = {"hands": b.shape[0], "listed": int(b["listed"].sum()),
                 "b0": int((b["block"] == 0).sum()), "b1": int((b["block"] == 1).sum()), "b_amb": int(((b["block"] == -1) & (b["listed"] == 1)).sum()),
                 "nonevent": int((b["y"] == 0).sum()), "nonevent_assumed": int(b["region_assumed"].sum()),
                 "unknown": int(b["y"].is_null().sum())}
print(json.dumps(summ, indent=1))

# ---------------- univariate screen
atoms = mech + derived
def screen(X, pos_mask, neg_mask):
    xp, xn = X[pos_mask], X[neg_mask]
    vals = np.unique(np.concatenate([xp, xn]))
    if len(vals) > 60:
        vals = np.unique(np.quantile(vals, np.linspace(0, 1, 61)))
    best = {"J": (-1, None, None, None, None), "r_fpr1": (-1, None, None, None, None), "r_fpr02": (-1, None, None, None, None)}
    xps, xns = np.sort(xp), np.sort(xn)
    npos, nneg = len(xp), len(xn)
    for thr in vals:
        r_ge = 1 - np.searchsorted(xps, thr, side="left") / npos
        f_ge = 1 - np.searchsorted(xns, thr, side="left") / nneg
        r_le = np.searchsorted(xps, thr, side="right") / npos
        f_le = np.searchsorted(xns, thr, side="right") / nneg
        for d, r, f in ((">=", r_ge, f_ge), ("<=", r_le, f_le)):
            if r - f > best["J"][0]:
                best["J"] = (r - f, d, float(thr), r, f)
            if f <= 0.01 and r > best["r_fpr1"][0]:
                best["r_fpr1"] = (r, d, float(thr), r, f)
            if f <= 0.002 and r > best["r_fpr02"][0]:
                best["r_fpr02"] = (r, d, float(thr), r, f)
    return best

rows = []
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    b = base.filter(pl.col("family") == fam)
    yv = b["y"].to_numpy(); blk = b["block"].to_numpy(); listed = b["listed"].to_numpy()
    neg = (yv == 0)
    targets = {"any": listed == 1, "b0": (listed == 1) & (blk == 0), "b1": (listed == 1) & (blk == 1)}
    Xall = b.select(atoms).to_numpy().astype(np.float64)
    for j, a in enumerate(atoms):
        X = Xall[:, j]
        if np.nanstd(X) == 0:
            continue
        for tname, tm in targets.items():
            if tm.sum() < 10:
                continue
            bst = screen(X, tm, neg)
            rows.append({"family": fam, "target": tname, "atom": a, "n_pos": int(tm.sum()), "n_neg": int(neg.sum()),
                         "J": bst["J"][0], "J_dir": bst["J"][1], "J_thr": bst["J"][2], "J_rec": bst["J"][3], "J_fpr": bst["J"][4],
                         "rec_fpr1": bst["r_fpr1"][0], "fpr1_dir": bst["r_fpr1"][1], "fpr1_thr": bst["r_fpr1"][2], "fpr1_fpr": bst["r_fpr1"][4],
                         "rec_fpr02": bst["r_fpr02"][0], "fpr02_dir": bst["r_fpr02"][1], "fpr02_thr": bst["r_fpr02"][2]})
res = pl.DataFrame(rows)
res.write_csv(os.path.join(OUT, "s02_univariate.csv"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_ascii_tables(True); pl.Config.set_fmt_str_lengths(32)
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    for t in ("any", "b0", "b1"):
        r = res.filter((pl.col("family") == fam) & (pl.col("target") == t)).sort("rec_fpr1", descending=True).head(18)
        if r.shape[0] == 0:
            continue
        print(f"\n===== {fam} / {t}  (n_pos {r['n_pos'][0]}, n_neg {r['n_neg'][0]})  top by recall at FPR<=1%")
        print(r.select("atom", "rec_fpr1", "fpr1_dir", "fpr1_thr", "fpr1_fpr", "rec_fpr02", "J", "J_rec", "J_fpr", "J_dir", "J_thr"))
json.dump(summ, open(os.path.join(OUT, "s02_label_summary.json"), "w"), indent=1)
