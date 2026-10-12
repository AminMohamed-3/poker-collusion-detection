"""S05 - beam search over short conjunctions of thresholded atoms (depth <= 3), censoring-aware (labels y2).

Targets per family: b0 (determined first-block listed hands), b1 (determined second-block), any (all listed).
Negatives: confirmed non-events (y2 == 0) of the same family's pairs. Unknown (censored) hands excluded.
Objectives: max recall subject to FPR <= 1% / 2% / 3% (soft penalty beyond), and max Youden J.
Honest numbers: 5-fold GroupKFold by table -- the search runs on 4 folds, the chosen rule is scored on the 5th.
Each depth's entry is the best rule of depth <= that depth (monotone).
"""
import polars as pl, numpy as np, json, os, sys, time
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(__file__))
from rulesearch import build_literals, beam_search, apply_spec
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
labf = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet"))
raw = pl.read_parquet(os.path.join(OUT, "raw_atoms.parquet"))
df = base.join(labf.select(["pair_id", "hand_id", pl.col("y2").alias("y")] + [c for c in labf.columns if c.startswith("d_")]), on=["pair_id", "hand_id"]).join(raw, on=["pair_id", "hand_id"])
mech = [c for c in base.columns if c[:3] in ("t1_", "t2_", "t3_", "t4_", "t5_")]
chrono_cols = [c for c in mech if c.startswith("t4_") and any(k in c for k in ("session", "sess_profit", "rebuy", "topup", "stackchg", "stack_bb", "cumabsflow", "shared_idx", "shared_frac"))]
atoms = [c for c in mech if c not in chrono_cols] + [c for c in labf.columns if c.startswith("d_")] + [c for c in raw.columns if c.startswith("r_")]
MAXDEPTH = 3
OBJS = ["0.01", "0.02", "0.03", "J"]
results = {}
t0 = time.time()
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    d = df.filter((pl.col("family") == fam) & pl.col("y").is_not_null())
    X = d.select(atoms).to_numpy().astype(np.float64)
    y = d["y"].to_numpy().astype(int); blk = d["block"].to_numpy(); tables = d["table_id"].to_numpy()
    LIT, desc, spec = build_literals(X, atoms)
    print(f"\n######## {fam}: rows {len(y)}, literals {LIT.shape[1]}, {time.time()-t0:.0f}s", flush=True)
    targets = {"any": y == 1, "b0": (y == 1) & (blk == 0), "b1": (y == 1) & (blk == 1)}
    neg = y == 0
    results[fam] = {"atoms": atoms}
    for tname, pos in targets.items():
        if pos.sum() < 30:
            continue
        results[fam][tname] = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
        for obj in OBJS:
            ins = beam_search(LIT, pos, neg, obj, maxdepth=MAXDEPTH)
            entry = {"in_sample": {int(dd): {"rule": " & ".join(desc[i] for i in k), "lits": [spec[i] for i in k], "score": s, "rec": r, "fpr": f} for dd, (k, s, r, f) in ins.items()}}
            oof_hit = {dd: np.zeros(len(y), dtype=bool) for dd in range(1, MAXDEPTH + 1)}
            fold_rules = {dd: [] for dd in range(1, MAXDEPTH + 1)}
            for fi, (tr, te) in enumerate(GroupKFold(5).split(X, y, tables)):
                trm = np.zeros(len(y), dtype=bool); trm[tr] = True
                bs = beam_search(LIT, pos & trm, neg & trm, obj, maxdepth=MAXDEPTH)
                for dd, (k, s, r, f) in bs.items():
                    oof_hit[dd][te] = np.all(LIT[te][:, list(k)], axis=1)
                    fold_rules[dd].append({"rule": " & ".join(desc[i] for i in k), "lits": [spec[i] for i in k], "test_tables": sorted(set(tables[te].tolist()))})
            entry["oof"] = {}
            for dd in range(1, MAXDEPTH + 1):
                h = oof_hit[dd]
                rec = float(h[pos].mean()); fpr = float(h[neg].mean())
                prec = float(h[pos].sum() / max(1, h[pos].sum() + h[neg].sum()))
                entry["oof"][dd] = {"rec": rec, "fpr": fpr, "prec_in_region": prec, "fold_rules": fold_rules[dd]}
            results[fam][tname][obj] = entry
            b = entry["in_sample"][MAXDEPTH]; o = entry["oof"][MAXDEPTH]
            print(f"  {tname:3s} obj={obj:4s} in-sample: rec {b['rec']:.3f} fpr {b['fpr']:.4f} | OOF: rec {o['rec']:.3f} fpr {o['fpr']:.4f} prec {o['prec_in_region']:.3f}\n      {b['rule']}", flush=True)
    json.dump(results, open(os.path.join(OUT, "s05_conjunctions_v2.json"), "w"), indent=1)
print("done", time.time() - t0)
