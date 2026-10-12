"""S16 - what defines the BLOCK (traversal key) in soft play and coordinated isolation?
Hypotheses for the second sort key, tested on descent-determined lists:
  H1 type   : fold-to-partner vs showdown/call (the production T1 flag)
  H2 loser  : which member lost chips in the hand (direction of the transfer)
  H3 vpip   : both members voluntarily in vs a blind-vs-blind check-down (no VPIP)
  H4 bets   : any bet between the members vs checked down
For each hypothesis: share of 1-descent lists in which the key is constant within each block AND differs across the
blocks (a perfect key gives 1.0); and for 0-descent lists whether the key is monotone along the rank order.
"""
import polars as pl, numpy as np, json, os
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
raw = pl.read_parquet(os.path.join(OUT, "raw_atoms.parquet"))
L = base.join(raw, on=["pair_id", "hand_id"]).filter(pl.col("listed") == 1).sort(["pair_id", "evidence_rank"])
L = L.with_columns(
    k_type=pl.when(pl.col("t5_fold_to_partner_sm") > 0).then(0).otherwise(1),
    k_loser=pl.when(pl.col("t5_net_bb_A") < pl.col("t5_net_bb_B")).then(0).otherwise(1),
    k_vpip=pl.when(pl.col("t5_vpip_sm") == 2).then(0).otherwise(1),
    k_bets=pl.when((pl.col("t1_bbs_aggressor_A") + pl.col("t1_bbs_aggressor_B")) > 0).then(0).otherwise(1),
    k_sd=pl.when(pl.col("t5_both_showdown") == 1).then(1).otherwise(0),
    k_outfold=pl.when(pl.col("r_n_out_fold_vs_mem") >= 2).then(0).otherwise(1),
    k_firstaggr=pl.when(pl.col("r_first_aggr_mem") == 1).then(0).otherwise(1),
)
keys = [c for c in L.columns if c.startswith("k_")]
out = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    res = {k: {"one_desc_perfect": 0, "one_desc_n": 0, "zero_desc_monotone": 0, "zero_desc_n": 0, "two_desc_perfect3": 0, "two_desc_n": 0} for k in keys}
    for pid, g in L.filter(pl.col("family") == fam).group_by("pair_id", maintain_order=True):
        t = g["t_us"].to_list(); nd = int(g["n_desc"][0]); blk = g["block"].to_list()
        for k in keys:
            v = g[k].to_list()
            if nd == 1:
                res[k]["one_desc_n"] += 1
                b0 = [v[i] for i in range(len(v)) if blk[i] == 0]; b1 = [v[i] for i in range(len(v)) if blk[i] == 1]
                if len(set(b0)) == 1 and len(set(b1)) == 1 and b0[0] != b1[0]:
                    res[k]["one_desc_perfect"] += 1
            elif nd == 0:
                res[k]["zero_desc_n"] += 1
                if all(v[i] <= v[i + 1] for i in range(len(v) - 1)):
                    res[k]["zero_desc_monotone"] += 1
            elif nd == 2:
                res[k]["two_desc_n"] += 1
                desc = [i for i in range(1, len(t)) if t[i] < t[i - 1]]
                segs = [v[:desc[0]], v[desc[0]:desc[1]], v[desc[1]:]]
                if all(len(set(s)) == 1 for s in segs) and segs[0][0] < segs[1][0] < segs[2][0]:
                    res[k]["two_desc_perfect3"] += 1
    out[fam] = res
    print(f"\n===== {fam}")
    for k in keys:
        r = res[k]
        print(f"  {k:12s} 1-desc perfect {r['one_desc_perfect']:3d}/{r['one_desc_n']:3d}   0-desc monotone {r['zero_desc_monotone']:3d}/{r['zero_desc_n']:3d}   2-desc 3-level {r['two_desc_perfect3']}/{r['two_desc_n']}")
# SP: combined key (type, then loser) etc.
print("\n--- soft_play combined keys on 1-descent lists (constant within block, different across):")
for fam in ("soft_play", "directed_transfer", "coordinated_isolation"):
    for combo in (("k_type", "k_loser"), ("k_type", "k_vpip"), ("k_type", "k_bets"), ("k_sd", "k_vpip"), ("k_loser", "k_vpip")):
        n = 0; ok = 0
        for pid, g in L.filter((pl.col("family") == fam) & (pl.col("n_desc") == 1)).group_by("pair_id", maintain_order=True):
            blk = g["block"].to_list(); v = list(zip(g[combo[0]].to_list(), g[combo[1]].to_list()))
            b0 = [v[i] for i in range(len(v)) if blk[i] == 0]; b1 = [v[i] for i in range(len(v)) if blk[i] == 1]
            n += 1
            if len(set(b0)) == 1 and len(set(b1)) == 1 and b0[0] != b1[0]:
                ok += 1
        print(f"  {fam:22s} {combo}: {ok}/{n}")
# print the SP two-descent lists and the SP lists with S in block 0, with keys
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
sp = L.filter(pl.col("family") == "soft_play")
print("\nSP 2-descent lists:"); print(sp.filter(pl.col("n_desc") == 2).select("pair_id", "evidence_rank", "idx", "k_type", "k_loser", "k_vpip", "k_bets", "k_sd", "t5_pot_bb", "t5_net_bb_A", "t5_net_bb_B"))
json.dump(out, open(os.path.join(OUT, "s16_subtype_hypotheses.json"), "w"), indent=1)
