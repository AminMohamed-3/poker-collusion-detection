"""S03 - flexible-model bound on the marking rule.

If a gradient-boosted model over all atoms (662 mechanical + derived + raw outsider atoms) cannot separate
listed hands from CONFIRMED non-events of the same pairs at near-zero false-positive rate, then no short
rule over those atoms can either (rules are a subset of what the trees express). Cross-fitted by table
(GroupKFold 5). Unknown (censored) hands are excluded from training and evaluation.
Arms: MECH (no chronology: drops t4 chronology columns, frac, idx), MECH+CHRONO (adds them back).
Reports OOF AUC, AP, recall at FPR 1% / 0.2%, precision at recall .9, per-block recall at FPR 1%.
"""
import polars as pl, numpy as np, json, os, warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
lab = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet")).select(["pair_id", "hand_id", pl.col("y2").alias("y")] + [c for c in pl.read_parquet(os.path.join(OUT, "labels_v2.parquet")).columns if c.startswith("d_")])
raw = pl.read_parquet(os.path.join(OUT, "raw_atoms.parquet"))
df = base.join(lab, on=["pair_id", "hand_id"]).join(raw, on=["pair_id", "hand_id"])
mech = [c for c in base.columns if c[:3] in ("t1_", "t2_", "t3_", "t4_", "t5_")]
chrono_cols = [c for c in mech if c.startswith("t4_") and any(k in c for k in ("session", "sess_profit", "rebuy", "topup", "stackchg", "stack_bb", "cumabsflow", "shared_idx", "shared_frac"))] + ["frac", "idx"]
derived = [c for c in lab.columns if c.startswith("d_")]
rawc = [c for c in raw.columns if c.startswith("r_")]
ARMS = {"MECH": [c for c in mech if c not in chrono_cols] + derived + rawc,
        "MECH+CHRONO": mech + derived + rawc}
print("chrono cols dropped in MECH:", chrono_cols)

def rec_at_fpr(y, s, fpr):
    neg = np.sort(s[y == 0]); thr = neg[int(np.ceil((1 - fpr) * len(neg))) - 1]
    return float((s[y == 1] > thr).mean()), float(thr)
def prec_at_rec(y, s, rec):
    pos = np.sort(s[y == 1]); thr = pos[int(np.floor((1 - rec) * len(pos)))]
    sel = s >= thr
    return float(y[sel].mean())

params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, num_threads=8, seed=7)
res = {}
oof_frames = []
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    d = df.filter((pl.col("family") == fam) & pl.col("y").is_not_null())
    y = d["y"].to_numpy().astype(int); groups = d["table_id"].to_numpy(); blk = d["block"].to_numpy()
    res[fam] = {"n": int(len(y)), "n_pos": int(y.sum())}
    for arm, cols in ARMS.items():
        X = d.select(cols).to_numpy().astype(np.float32)
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=600)
            oof[te] = m.predict(X[te])
        r1, t1 = rec_at_fpr(y, oof, 0.01); r02, t02 = rec_at_fpr(y, oof, 0.002); r5, _ = rec_at_fpr(y, oof, 0.05)
        out = {"auc": float(roc_auc_score(y, oof)), "ap": float(average_precision_score(y, oof)),
               "rec@fpr5%": r5, "rec@fpr1%": r1, "rec@fpr0.2%": r02, "prec@rec90": prec_at_rec(y, oof, 0.9), "prec@rec80": prec_at_rec(y, oof, 0.8),
               "b0_rec@fpr1%": float((oof[(y == 1) & (blk == 0)] > t1).mean()) if ((y == 1) & (blk == 0)).sum() else None,
               "b1_rec@fpr1%": float((oof[(y == 1) & (blk == 1)] > t1).mean()) if ((y == 1) & (blk == 1)).sum() else None,
               "amb_rec@fpr1%": float((oof[(y == 1) & (blk == -1)] > t1).mean())}
        res[fam][arm] = out
        print(fam, arm, json.dumps(out))
        if arm == "MECH":
            oof_frames.append(d.select("pair_id", "hand_id").with_columns(oof_mech=pl.Series(oof), y=pl.Series(y), family=pl.lit(fam)))
pl.concat(oof_frames).write_parquet(os.path.join(OUT, "s03_oof_mech_v2.parquet"))
json.dump(res, open(os.path.join(OUT, "s03_model_bound_v2.json"), "w"), indent=1)
