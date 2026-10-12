"""S11 - the determinism test proper: reproduce each pair's evidence list from a marking rule + the traversal rule.

For a rule pair (E0 = first-block event predicate, E1 = second-block event predicate; CI: one predicate):
  M = E0 | E1 ; block(h) = 0 if E0(h) else 1 ; predicted list = first 5 of M sorted by (block, time).
Scored against the real ordered list: per-pair precision / recall of the predicted set, exact-set rate, AP@5
(official form: mean over hit positions of precision-at-position, divided by min(5, |L|)), and pooled.
Rule sources:
  NAMED   : hand-written predicates (fold-to-partner while ahead; call-vs-partner; CI weak preflop raise + outsider folds)
  BEAM-OOF: S05 cross-fitted rules (each pair scored with the rule found on the other four folds), depth 3, per objective
  BEAM-IN : S05 in-sample rules (optimistic)
  MODEL   : LightGBM on the cleaned labels, cross-fitted by table, thresholded at the training-fold FPR 1% / 0.2%, block by B1
Also the ORACLE-M arm: M = the true listed set -> list is exact by construction (sanity), and ORACLE-M + natural
look-alikes to show the traversal is not the limiting step.
"""
import polars as pl, numpy as np, json, os, sys, warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from rulesearch import apply_spec
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"))
labf = pl.read_parquet(os.path.join(OUT, "labels_v2.parquet"))
raw = pl.read_parquet(os.path.join(OUT, "raw_atoms.parquet"))
df = base.join(labf.select(["pair_id", "hand_id", "y2", "B1", "n_listed"] + [c for c in labf.columns if c.startswith("d_")]), on=["pair_id", "hand_id"]).join(raw, on=["pair_id", "hand_id"])
S05 = json.load(open(os.path.join(OUT, "s05_conjunctions_v2.json")))
# strength-gap fold-ahead predicate
def pick(street_col, names):
    return pl.when(street_col == 0).then(pl.col(names[0])).when(street_col == 1).then(pl.col(names[1])).when(street_col == 2).then(pl.col(names[2])).when(street_col == 3).then(pl.col(names[3])).otherwise(None)
df = df.with_columns(folder=pl.when(pl.col("t1_ap_fold_aggressor_A") > 0).then(pl.lit("A")).when(pl.col("t1_ap_fold_aggressor_B") > 0).then(pl.lit("B")).otherwise(None))
df = df.with_columns(f_street=pl.when(pl.col("folder") == "A").then(pl.col("t5_fold_street_A")).when(pl.col("folder") == "B").then(pl.col("t5_fold_street_B")).otherwise(None))
sg = pick(pl.col("f_street"), ["t2_strgap_pre", "t2_strgap_flop", "t2_strgap_turn", "t2_strgap_river"])
df = df.with_columns(f_strgap=pl.when(pl.col("folder") == "A").then(sg).when(pl.col("folder") == "B").then(-sg).otherwise(None))
df = df.with_columns(n_fold_strgap=(pl.col("folder").is_not_null() & (pl.col("f_strgap") > 0)).cast(pl.Int8),
                     n_fold_eq5=(pl.col("d_fold_ahead_any") == 1).cast(pl.Int8),
                     n_call_partner=(pl.col("t1_ap_call_aggressor_sm") > 0).cast(pl.Int8),
                     n_ci=((pl.col("t1_a_pre_raise_sm") > 0) & (pl.col("r_n_out_fold_vs_mem") >= 2)).cast(pl.Int8),
                     n_ci_weak=((pl.col("t1_a_pre_raise_sm") > 0) & (pl.col("t1_wgss_pre_sm") > 0.5) & (pl.col("r_n_out_fold_vs_mem") >= 2)).cast(pl.Int8))

def simulate(d, E0, E1, single_block=False):
    """d: one family's frame (all hands); E0/E1 boolean arrays aligned with d. Returns per-pair metrics frame."""
    d = d.with_columns(E0=pl.Series(E0.astype(bool)), E1=pl.Series(E1.astype(bool)))
    d = d.with_columns(M=pl.col("E0") | pl.col("E1"), blk=pl.when(pl.col("E0")).then(0).otherwise(1) if not single_block else pl.lit(0))
    rows = []
    for pid, g in d.group_by("pair_id", maintain_order=True):
        pid = pid[0] if isinstance(pid, tuple) else pid
        L = g.filter(pl.col("listed") == 1).sort("evidence_rank")["hand_id"].to_list()
        m = g.filter(pl.col("M")).sort(["blk", "t_us"])
        pred = m["hand_id"].to_list()[:5]
        Ls = set(L); hits = [h in Ls for h in pred]
        ap = sum(sum(hits[:k + 1]) / (k + 1) for k in range(len(pred)) if hits[k]) / min(5, len(L)) if pred else 0.0
        inter = len(Ls.intersection(pred))
        nM = int(g["M"].sum()); nM_listed = int((g["M"] & (g["listed"] == 1)).sum()); nM_conf = int((g["M"] & (g["y2"] == 0)).sum())
        rows.append({"pair_id": pid, "n_listed": len(L), "n_pred": len(pred), "hits": inter, "prec": inter / len(pred) if pred else 0.0, "rec": inter / len(L), "exact": int(set(pred) == Ls),
                     "exact_ordered": int(pred == L), "ap5": ap, "n_M": nM, "M_recall_listed": nM_listed / len(L), "M_fp_confirmed": nM_conf})
    return pl.DataFrame(rows)

def summarize(name, R):
    return {"arm": name, "pairs": R.shape[0], "prec": round(float(R["prec"].mean()), 4), "rec": round(float(R["rec"].mean()), 4), "exact_set": round(float(R["exact"].mean()), 4),
            "exact_ordered": round(float(R["exact_ordered"].mean()), 4), "ap5": round(float(R["ap5"].mean()), 4), "M_per_pair": round(float(R["n_M"].mean()), 2),
            "M_recall_listed": round(float(R["M_recall_listed"].mean()), 4), "M_fp_confirmed_per_pair": round(float(R["M_fp_confirmed"].mean()), 3)}

params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=7)
results = {}
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(220)
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    d = df.filter(pl.col("family") == fam).sort(["pair_id", "t_us"])
    atoms = S05[fam]["atoms"]; X = d.select(atoms).to_numpy().astype(np.float64)
    single = fam == "coordinated_isolation"
    arms = []
    listed = d["listed"].to_numpy() == 1
    # oracle
    arms.append(("ORACLE-M (true set)", listed & (d["block"].to_numpy() != 1), listed & (d["block"].to_numpy() == 1)))
    # named
    fs = d["n_fold_strgap"].to_numpy() == 1; fe = d["n_fold_eq5"].to_numpy() == 1; cp = d["n_call_partner"].to_numpy() == 1
    if not single:
        arms.append(("NAMED fold-strgap | call-partner", fs, cp & ~fs))
        arms.append(("NAMED fold-eq>.5 | call-partner", fe, cp & ~fe))
        arms.append(("NAMED fold-strgap only", fs, np.zeros(len(d), bool)))
    else:
        arms.append(("NAMED pre-raise & outfold>=2", d["n_ci"].to_numpy() == 1, np.zeros(len(d), bool)))
        arms.append(("NAMED weak pre-raise & outfold>=2", d["n_ci_weak"].to_numpy() == 1, np.zeros(len(d), bool)))
    # beam rules
    tables = d["table_id"].to_numpy()
    for obj in ("0.01", "0.02", "0.03", "J"):
        if single:
            r0 = S05[fam]["any"][obj]
            E0 = apply_spec(X, [i for i in range(len(r0["in_sample"]["3"]["lits"]))], [tuple(l) for l in r0["in_sample"]["3"]["lits"]]); E1 = np.zeros(len(d), bool)
            arms.append((f"BEAM-IN any@{obj}", E0, E1))
            E0o = np.zeros(len(d), bool)
            for fr in r0["oof"]["3"]["fold_rules"]:
                te = np.isin(tables, fr["test_tables"]); E0o[te] = apply_spec(X[te], list(range(len(fr["lits"]))), [tuple(l) for l in fr["lits"]])
            arms.append((f"BEAM-OOF any@{obj}", E0o, E1))
        else:
            r0 = S05[fam]["b0"][obj]; r1 = S05[fam]["b1"][obj]
            E0 = apply_spec(X, list(range(len(r0["in_sample"]["3"]["lits"]))), [tuple(l) for l in r0["in_sample"]["3"]["lits"]])
            E1 = apply_spec(X, list(range(len(r1["in_sample"]["3"]["lits"]))), [tuple(l) for l in r1["in_sample"]["3"]["lits"]])
            arms.append((f"BEAM-IN b0|b1@{obj}", E0, E1 & ~E0))
            E0o = np.zeros(len(d), bool); E1o = np.zeros(len(d), bool)
            for fr0, fr1 in zip(r0["oof"]["3"]["fold_rules"], r1["oof"]["3"]["fold_rules"]):
                te = np.isin(tables, fr0["test_tables"])
                E0o[te] = apply_spec(X[te], list(range(len(fr0["lits"]))), [tuple(l) for l in fr0["lits"]])
                E1o[te] = apply_spec(X[te], list(range(len(fr1["lits"]))), [tuple(l) for l in fr1["lits"]])
            arms.append((f"BEAM-OOF b0|b1@{obj}", E0o, E1o & ~E0o))
    # model arm: LightGBM on cleaned labels, cross-fitted by table, predict ALL hands of the test tables
    y2 = d["y2"].to_numpy(); lab_mask = ~np.isnan(y2.astype(float)) if y2.dtype.kind == "f" else d["y2"].is_not_null().to_numpy()
    yv = np.nan_to_num(d["y2"].cast(pl.Float64).to_numpy(), nan=-1)
    score = np.full(len(d), np.nan); thr1 = np.zeros(len(d)); thr02 = np.zeros(len(d))
    Xf = X.astype(np.float32)
    for tr, te in GroupKFold(5).split(Xf, yv, tables):
        trl = tr[yv[tr] >= 0]
        m = lgb.train(params, lgb.Dataset(Xf[trl], yv[trl].astype(int)), num_boost_round=600)
        p_tr = m.predict(Xf[trl]); neg_tr = np.sort(p_tr[yv[trl] == 0])
        t1 = neg_tr[int(np.ceil(0.99 * len(neg_tr))) - 1]; t02 = neg_tr[int(np.ceil(0.998 * len(neg_tr))) - 1]
        score[te] = m.predict(Xf[te]); thr1[te] = t1; thr02[te] = t02
    B1 = d["B1"].to_numpy() == 1
    for nm, thr in (("MODEL@fpr1%", thr1), ("MODEL@fpr0.2%", thr02)):
        M = score > thr
        arms.append((nm + " block=B1", M & ~B1, M & B1) if not single else (nm, M, np.zeros(len(d), bool)))
    # model arm with fixed 9 per pair: top-9 by score then traverse
    d_s = d.with_columns(sc=pl.Series(score)).with_columns(rk=pl.col("sc").rank(descending=True).over("pair_id"))
    M9 = (d_s["rk"] <= 9).to_numpy()
    arms.append(("MODEL top-9/pair block=B1", M9 & ~B1, M9 & B1) if not single else ("MODEL top-9/pair", M9, np.zeros(len(d), bool)))
    M5 = (d_s["rk"] <= 5).to_numpy()
    arms.append(("MODEL top-5/pair (no traversal)", M5, np.zeros(len(d), bool)))
    res = []
    for name, E0, E1 in arms:
        R = simulate(d, E0, E1, single_block=single)
        res.append(summarize(name, R))
    T = pl.DataFrame(res)
    print(f"\n===== {fam} ({T['pairs'][0]} pairs)"); print(T)
    results[fam] = res
json.dump(results, open(os.path.join(OUT, "s11_traversal_simulation.json"), "w"), indent=1)
