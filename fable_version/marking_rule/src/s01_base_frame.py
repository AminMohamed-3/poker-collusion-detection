"""S01 - base frame for the marking-rule determinism study.

Positive (confirmed_target) development pairs only. One row per (pair, shared hand):
pair_id, hand_id, family, table_id, started_at, idx (position in the pair's shared history, 0-based),
n_hands, listed, evidence_rank, block (0 = first block, 1 = second block, -1 = ambiguous/unlisted),
plus the 662 STEP254 mechanical columns (read-only from the Claude Version study).
Development phase asserted on every hand.
"""
import polars as pl, numpy as np, json, os
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
S254 = os.path.join(ROOT, "Claude Version", "experiments", "step254_mechanical_tensor_20260917")

lab = pl.read_csv(os.path.join(ROOT, "data", "development_labels.csv"))
ev = pl.read_csv(os.path.join(ROOT, "data", "development_evidence.csv"))
pos = lab.filter(pl.col("label") == 1)
assert pos.shape[0] == 372 and (pos["label_status"] == "confirmed_target").all()

oof = pl.read_parquet(os.path.join(S254, "hand_oof_step254.parquet"), columns=["pair_id", "hand_id", "listed", "label", "family", "fold"])
ph = oof.filter(pl.col("label") == 1).select("pair_id", "hand_id", "listed", "family", "fold")
assert ph["pair_id"].n_unique() == 372
hands = pl.read_parquet(os.path.join(ROOT, "data", "hands.parquet"), columns=["hand_id", "table_id", "started_at", "phase", "big_blind", "final_pot", "players_dealt", "players_at_showdown", "board_cards"])
ph = ph.join(hands, on="hand_id", how="left")
assert (ph["phase"] == "development").all(), "phase assert failed"
ph = ph.drop("phase").with_columns(t_us=pl.col("started_at").dt.replace_time_zone(None).cast(pl.Int64)).drop("started_at")
# evidence rank
ph = ph.join(ev.select("pair_id", "hand_id", "evidence_rank"), on=["pair_id", "hand_id"], how="left")
assert ph.filter(pl.col("listed") == 1)["evidence_rank"].null_count() == 0
assert ph.filter(pl.col("listed") == 0)["evidence_rank"].null_count() == ph.filter(pl.col("listed") == 0).shape[0]
# check every evidence hand of a positive pair is in the frame
n_ev_pos = ev.join(pos.select("pair_id"), on="pair_id").shape[0]
assert ph["listed"].sum() == n_ev_pos == 1817, (ph["listed"].sum(), n_ev_pos)
# chronological index within the pair
ph = ph.sort(["pair_id", "t_us", "hand_id"]).with_columns(
    idx=pl.int_range(pl.len()).over("pair_id"), n_hands=pl.len().over("pair_id"))
ph = ph.with_columns(frac=(pl.col("idx") / (pl.col("n_hands") - 1)).cast(pl.Float64))

# ---- block labels for listed hands from descents of started_at along evidence_rank
L = ph.filter(pl.col("listed") == 1).sort(["pair_id", "evidence_rank"])
rows = []
for pid, g in L.group_by("pair_id", maintain_order=True):
    t = g["t_us"].to_list(); r = g["evidence_rank"].to_list(); h = g["hand_id"].to_list()
    assert r == list(range(1, len(r) + 1)), (pid, r)
    desc = [i for i in range(1, len(t)) if t[i] < t[i - 1]]
    nd = len(desc)
    if nd == 0:
        blk = [-1] * len(t)  # ambiguous: could be all block 0, all block 1, or any split
    elif nd == 1:
        blk = [0] * desc[0] + [1] * (len(t) - desc[0])
    else:
        blk = [-2] * len(t)  # more than one descent: violates the 2-block rule
    for i in range(len(t)):
        rows.append((pid[0] if isinstance(pid, tuple) else pid, h[i], blk[i], nd))
blk = pl.DataFrame(rows, schema=["pair_id", "hand_id", "block", "n_desc"], orient="row")
ph = ph.join(blk, on=["pair_id", "hand_id"], how="left").with_columns(
    block=pl.col("block").fill_null(-1), n_desc=pl.col("n_desc").fill_null(-1))

# ---- mechanical columns
feat = pl.read_parquet(os.path.join(S254, "step254_features.parquet"))
mech_cols = [c for c in feat.columns if c not in ("pair_id", "hand_id")]
assert len(mech_cols) == 662
ph = ph.join(feat, on=["pair_id", "hand_id"], how="left")
assert ph[mech_cols[0]].null_count() == 0
ph.write_parquet(os.path.join(OUT, "base_frame.parquet"))

# ---- summary
summ = {}
summ["rows"] = ph.shape[0]; summ["pairs"] = ph["pair_id"].n_unique(); summ["listed"] = int(ph["listed"].sum())
per_pair = ph.group_by("pair_id").agg(pl.col("family").first(), pl.col("listed").sum().alias("n_listed"), pl.col("n_hands").first(), pl.col("n_desc").max())
summ["lists_by_family_len"] = per_pair.group_by(["family", "n_listed"]).len().sort(["family", "n_listed"]).to_dicts()
summ["descents_by_family"] = per_pair.group_by(["family", "n_desc"]).len().sort(["family", "n_desc"]).to_dicts()
summ["n_hands_quantiles"] = {q: float(per_pair["n_hands"].quantile(q)) for q in (0, .1, .5, .9, 1)}
lst = ph.filter(pl.col("listed") == 1)
summ["listed_frac_quantiles_by_family"] = lst.group_by("family").agg(
    pl.col("frac").quantile(.1).alias("q10"), pl.col("frac").median().alias("q50"), pl.col("frac").quantile(.9).alias("q90")).to_dicts()
summ["block_counts"] = lst.group_by(["family", "block"]).len().sort(["family", "block"]).to_dicts()
json.dump(summ, open(os.path.join(OUT, "s01_summary.json"), "w"), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str))
