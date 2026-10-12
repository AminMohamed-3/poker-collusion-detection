"""S02c - refine the confirmed non-event region with a block predicate (19.230's rule, re-derived here).

Block predicate B1(h) = 1 (second block) iff a member CALLED while the partner was the aggressor AND no member
folded to the partner while ahead (fold-ahead = first block). Checked on the descent-determined listed hands.
Region rules (cap 5, traversal lexsort((block, time))):
  uncapped list            -> every unlisted hand is a non-event
  capped, 1 descent        -> unlisted hands before t(rank 5) are non-events (block-free)
  capped, 0 descents       -> k0_hat = number of listed hands with B1 = 0.
                              k0_hat == 5: unlisted hands before t(rank 5) with B1 = 0 are non-events; B1 = 1 unknown
                              k0_hat <  5: unlisted hands before t(rank 5) are non-events
  capped, 2 descents (3 SP)-> no region
Also an EXTENDED region (variant): in capped lists with k0 < 5 the first block is complete, so unlisted B1 = 0
hands AFTER t(rank 5) are non-events too. Written as y2 (main) and y2x (extended).
"""
import polars as pl, numpy as np, json, os
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
lab = pl.read_parquet(os.path.join(OUT, "labels_derived.parquet"))
lab = lab.with_columns(B1=((pl.col("d_call_vs_partner_any") == 1) & (pl.col("d_fold_ahead_any") == 0)).cast(pl.Int8))
# accuracy of the predicate on determined listed hands
det = lab.filter(pl.col("listed") == 1).filter(pl.col("block").is_in([0, 1]))
acc = {fam: {"n": g.shape[0], "acc": float((g["B1"] == g["block"]).mean()),
             "b0_as_1": float(((g["block"] == 0) & (g["B1"] == 1)).sum() / max(1, (g["block"] == 0).sum())),
             "b1_as_0": float(((g["block"] == 1) & (g["B1"] == 0)).sum() / max(1, (g["block"] == 1).sum()))}
       for fam, g in det.group_by("family")}
acc = {k[0] if isinstance(k, tuple) else k: v for k, v in acc.items()}
print("block predicate accuracy on determined listed hands:", json.dumps(acc, indent=1))
# per-pair k0_hat for 0-descent lists
k0 = (lab.filter(pl.col("listed") == 1).group_by("pair_id")
      .agg((pl.col("B1") == 0).sum().alias("k0_hat"), pl.col("n_desc").max().alias("nd"), pl.col("evidence_rank").max().alias("nl")))
lab = lab.join(k0.select("pair_id", "k0_hat"), on="pair_id", how="left")
capped = pl.col("n_listed") == 5
before = pl.col("t_us") < pl.col("t_last")
unl = pl.col("listed") == 0
y2 = (pl.when(pl.col("listed") == 1).then(1)
      .when(pl.col("n_desc") == 2).then(None)
      .when(~capped).then(0)
      .when((pl.col("n_desc") == 1) & before).then(0)
      .when((pl.col("n_desc") == 0) & before & (pl.col("k0_hat") < 5)).then(0)
      .when((pl.col("n_desc") == 0) & before & (pl.col("k0_hat") == 5) & (pl.col("B1") == 0)).then(0)
      .otherwise(None))
# n_desc is per-hand for listed rows only; propagate per pair
nd = lab.filter(pl.col("listed") == 1).group_by("pair_id").agg(pl.col("n_desc").max().alias("nd_p"))
lab = lab.join(nd, on="pair_id", how="left").with_columns(n_desc=pl.col("nd_p")).drop("nd_p")
lab = lab.with_columns(y2=y2.cast(pl.Int8))
# extended: first block complete (k0 < 5 in capped lists, or 1 descent) -> unlisted B1==0 hands after t_last are non-events
first_complete = (capped & ((pl.col("n_desc") == 1) | ((pl.col("n_desc") == 0) & (pl.col("k0_hat") < 5))))
y2x = pl.when(pl.col("y2").is_not_null()).then(pl.col("y2")).when(unl & first_complete & (pl.col("B1") == 0) & (pl.col("n_desc") != 2)).then(0).otherwise(None)
lab = lab.with_columns(y2x=y2x.cast(pl.Int8))
lab.write_parquet(os.path.join(OUT, "labels_v2.parquet"))
summ = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    b = lab.filter(pl.col("family") == fam)
    summ[fam] = {"listed": int(b["listed"].sum()), "y_nonevent": int((b["y"] == 0).sum()), "y2_nonevent": int((b["y2"] == 0).sum()),
                 "y2x_nonevent": int((b["y2x"] == 0).sum()), "moved_to_unknown": int(((b["y"] == 0) & b["y2"].is_null()).sum()),
                 "lists_k0hat5_0desc": int(k0.join(b.select("pair_id").unique(), on="pair_id").filter((pl.col("nd") == 0) & (pl.col("k0_hat") == 5) & (pl.col("nl") == 5)).shape[0]),
                 "lists_0desc_capped": int(k0.join(b.select("pair_id").unique(), on="pair_id").filter((pl.col("nd") == 0) & (pl.col("nl") == 5)).shape[0])}
print(json.dumps(summ, indent=1))
json.dump({"predicate_accuracy": acc, "regions": summ}, open(os.path.join(OUT, "s02c_labels_v2.json"), "w"), indent=1)
