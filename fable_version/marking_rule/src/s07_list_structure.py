"""S07 - structure of the lists: what TYPE each block is, whether the type order is universal, and whether the
block order is by first occurrence. Listed hands only.
Type of a listed hand (crisp): FOLD = a member folded facing the partner as aggressor (t5_fold_to_partner_sm>0);
CALL = a member called the partner's aggression and nobody folded to the partner; SD = both to showdown; OTHER.
"""
import polars as pl, numpy as np, json, os, collections
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
lab = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet"))
df = base.join(lab.select(["pair_id", "hand_id", "B1"] + [c for c in lab.columns if c.startswith("d_")]), on=["pair_id", "hand_id"])
L = df.filter(pl.col("listed") == 1).sort(["pair_id", "evidence_rank"])
L = L.with_columns(typ=pl.when(pl.col("t5_fold_to_partner_sm") > 0).then(pl.lit("FOLD"))
                   .when(pl.col("t5_both_showdown") == 1).then(pl.lit("SD"))
                   .when(pl.col("t1_ap_call_aggressor_sm") > 0).then(pl.lit("CALL")).otherwise(pl.lit("OTHER")))
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(40)
out = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    rows = []
    for pid, g in L.filter(pl.col("family") == fam).group_by("pair_id", maintain_order=True):
        pid = pid[0] if isinstance(pid, tuple) else pid
        t = g["t_us"].to_list(); typ = g["typ"].to_list(); blk = g["block"].to_list(); b1 = g["B1"].to_list(); fa = g["d_fold_ahead_any"].to_list()
        nd = g["n_desc"][0]
        rec = {"pair_id": pid, "n": len(t), "n_desc": int(nd), "types": "".join({"FOLD": "F", "SD": "S", "CALL": "C", "OTHER": "O"}[x] for x in typ),
               "B1": "".join(str(x) for x in b1), "foldahead": "".join(str(x) for x in fa)}
        if nd == 1:
            i = [k for k in range(1, len(t)) if t[k] < t[k - 1]][0]
            rec["k0"] = i
            rec["tmin0_lt_tmin1"] = bool(min(t[:i]) < min(t[i:]))
            rec["tmax0_lt_tmax1"] = bool(max(t[:i]) < max(t[i:]))
            rec["types0"] = "".join(sorted(set(rec["types"][:i]))); rec["types1"] = "".join(sorted(set(rec["types"][i:])))
            rec["fa0"] = sum(fa[:i]) / i; rec["fa1"] = sum(fa[i:]) / (len(t) - i)
        rows.append(rec)
    R = pl.DataFrame(rows)
    one = R.filter(pl.col("n_desc") == 1)
    out[fam] = {"lists": R.shape[0], "one_desc": one.shape[0],
                "tmin0_lt_tmin1_share": float(one["tmin0_lt_tmin1"].mean()) if one.shape[0] else None,
                "tmax0_lt_tmax1_share": float(one["tmax0_lt_tmax1"].mean()) if one.shape[0] else None,
                "fold_first_share": float((one["fa0"] > one["fa1"]).mean()) if one.shape[0] else None,
                "k0_dist": one.group_by("k0").len().sort("k0").to_dicts() if one.shape[0] else None,
                "type_pattern_counts_1desc": one.group_by("types0", "types1").len().sort("len", descending=True).head(12).to_dicts() if one.shape[0] else None,
                "type_pattern_counts_0desc": R.filter(pl.col("n_desc") == 0).group_by("types").len().sort("len", descending=True).head(15).to_dicts()}
    print(f"\n===== {fam}"); print(json.dumps(out[fam], indent=1, default=str))
    if one.shape[0]:
        rev = one.filter(pl.col("fa0") < pl.col("fa1"))
        print("REVERSED lists (block 0 less fold-ahead than block 1):"); print(rev.select("pair_id", "k0", "types", "B1", "foldahead", "tmin0_lt_tmin1", "tmax0_lt_tmax1"))
        print("lists where the second block starts EARLIER than the first (tmin0 > tmin1):"); print(one.filter(~pl.col("tmin0_lt_tmin1")).select("pair_id", "k0", "types", "B1", "foldahead"))
json.dump(out, open(os.path.join(OUT, "s07_list_structure.json"), "w"), indent=1, default=str)
