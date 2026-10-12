# P3 label-free profile of the evaluation block (task: .opencode/continuous/tasks/pair_ledger.md + NEXT_EXPERIMENTS_PLAN.md P3).
# Read-only inputs from Claude Version/experiments/viewdis_20260911/; writes only next to this script.
# CPU-only, 2 threads, utf-8, seed 7329. No src/outputs/Claude Version/Codex Resume edits.
import os
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import json
from pathlib import Path
import numpy as np
import polars as pl

SEED = 7329
L2 = 10.0
VD = Path(r"C:\Data\Projects\Money Making\work\poker\Claude Version\experiments\viewdis_20260911")
OUT = Path(__file__).resolve().parent
VIEWS = ["FULL", "RAW", "CA", "BASE", "OPP", "V2", "MEAN", "TAILS"]
FS = VIEWS + ["sh"]  # M design (eligible_ledger "M (8 views + sh)")
DEFS = {"A": (["pos", "undisclosed"], ["FP", "undisclosed_R1"]),
        "C": (["pos"], ["FP", "undisclosed_R1", "ambiguous", "mid", "neg"])}


def lg(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def logreg(X, y, l2=L2, iters=200):  # step60 protocol: Newton on standardised inputs
    mu = X.mean(0); sd = X.std(0) + 1e-9; Z = (X - mu) / sd
    n, k = Z.shape; w = np.zeros(k); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = Z.T @ (p - y) + l2 * w; gb = (p - y).sum()
        W = p * (1 - p); H = (Z * W[:, None]).T @ Z + l2 * np.eye(k)
        step = np.linalg.solve(H, g); w -= step; b -= gb / max(W.sum(), 1e-9)
        if np.abs(step).max() < 1e-8:
            break
    return w, b, mu, sd


def X_testbed(c):
    cols = [lg(c["risk_prefix"].to_numpy()) if v == "FULL" else
            np.log(c["shared_hands"].to_numpy().astype(float)) if v == "sh" else
            lg(c[f"r_{v}"].to_numpy()) for v in FS]
    return np.column_stack(cols)


def X_eval(e):
    cols = [lg(e["r_FULL"].to_numpy()) if v == "FULL" else
            np.log(e["shared_hands"].to_numpy().astype(float)) if v == "sh" else
            lg(e[f"r_{v}"].to_numpy()) for v in FS]
    return np.column_stack(cols)


def main():
    mp = pl.read_parquet(VD / "multiview_prefix.parquet")
    tv = pl.read_parquet(VD / "tail_views_prefix.parquet").sort("pair_id", "fold_new") \
        .unique(subset=["pair_id"], keep="first", maintain_order=True).drop("fold_new")
    tb = pl.read_parquet(VD / "evidence_view_prefix_rawonly.parquet").sort("pair_id", "fold_new") \
        .unique(subset=["pair_id"], keep="first", maintain_order=True).select("pair_id", "E_top5")
    r1 = pl.read_parquet(VD / "prefix_rescoring.parquet").select("pair_id", "R1_prefix") \
        .unique(subset=["pair_id"], keep="first", maintain_order=True)
    d = mp.join(tv, on="pair_id", how="left").join(tb, on="pair_id", how="left") \
        .join(r1, on="pair_id", how="left").with_columns(pl.col("R1_prefix").fill_null(False))
    ev = pl.read_parquet(VD / "multiview_eval.parquet")
    blk = ev.filter(pl.col("risk") >= 0.5)
    assert blk.height == 909, f"eval block {blk.height} != 909"
    blk = blk.with_columns(pl.col("R1").fill_null(False))

    fits = {}
    for dn, (posset, negset) in DEFS.items():
        c = d.filter((pl.col("risk_prefix") >= 0.5) & (pl.col("shared_hands") >= 38)
                     & pl.col("cls").is_in(posset + negset)
                     & pl.col("E_top5").is_not_null() & pl.col("r_SHORT3").is_not_null())
        y = c["cls"].is_in(posset).to_numpy().astype(float)
        X = X_testbed(c)
        w, b, mu, sd = logreg(X, y)
        fits[dn] = dict(n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), w=w, b=b, mu=mu, sd=sd,
                        weights={v: round(float(x), 3) for v, x in zip(FS, w)}, b_out=round(float(b), 3))

    Xe = X_eval(blk)
    res = {"eval_block_n": blk.height, "defs": {dn: {"n_pos": f["n_pos"], "n_neg": f["n_neg"],
             "weights": f["weights"], "b": f["b_out"]} for dn, f in fits.items()}}
    s_like = lg(blk["r_FULL"].to_numpy()) + 0.5 * lg(blk["r_RAW"].to_numpy())
    s_like = np.where(blk["R1"].to_numpy(), s_like - 100, s_like)  # S order: R1 rows bottom
    order = np.argsort(-s_like, kind="stable")
    thirds = {}
    for dn, f in fits.items():
        p = 1 / (1 + np.exp(-(((Xe - f["mu"]) / f["sd"]) @ f["w"] + f["b"])))
        F = float(np.sum(1 - p))
        t = {"F": round(F, 1), "mean_p": round(float(p.mean()), 4)}
        names = ["top", "mid", "bottom"]
        for i, nm in enumerate(names):
            idx = order[i * 303:(i + 1) * 303]
            t[nm] = {"F": round(float(np.sum(1 - p[idx])), 1), "mean_p": round(float(p[idx].mean()), 4)}
        res[dn] = t
        if dn == "A":
            res["pA_summary"] = {"min": round(float(p.min()), 4), "median": round(float(np.median(p)), 4),
                                 "max": round(float(p.max()), 4)}
            pA = p
    res["S_order"] = {"score": "lg(r_FULL)+0.5*lg(r_RAW), R1 bottom", "third_n": 303}

    from sklearn.mixture import GaussianMixture
    f = fits["A"]
    Zs = (Xe - f["mu"]) / f["sd"]  # one shared space: testbed-A standardisation; GMM itself label-free
    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=SEED, n_init=10).fit(Zs)
    o = np.argsort(-gm.means_[:, 0], kind="stable")  # comp0 = higher FULL logit
    gm_means, gm_w = gm.means_[o], gm.weights_[o]
    cA = d.filter((pl.col("risk_prefix") >= 0.5) & (pl.col("shared_hands") >= 38)
                  & pl.col("cls").is_in(DEFS["A"][0] + DEFS["A"][1])
                  & pl.col("E_top5").is_not_null() & pl.col("r_SHORT3").is_not_null())
    yA = cA["cls"].is_in(DEFS["A"][0]).to_numpy()
    Zt = (X_testbed(cA) - f["mu"]) / f["sd"]
    mu_pos, mu_neg = Zt[yA == 1].mean(0), Zt[yA == 0].mean(0)
    res["GMM"] = {"converged": bool(gm.converged_), "weights": [round(float(x), 3) for x in gm_w],
                  "means": {v: [round(float(x), 3) for x in col] for v, col in zip(FS, gm_means.T)},
                  "testbed_pos_mean": {v: round(float(x), 3) for v, x in zip(FS, mu_pos)},
                  "testbed_neg_mean": {v: round(float(x), 3) for v, x in zip(FS, mu_neg)}}

    def mahal2(Z, mu):
        C = np.cov(Z, rowvar=False) + 1e-6 * np.eye(Z.shape[1])
        D = Z - mu
        try:
            return np.einsum("ij,ij->i", D, np.linalg.solve(C, D.T).T)
        except np.linalg.LinAlgError:
            return np.einsum("ij,ij->i", D, (np.linalg.pinv(C) @ D.T).T)
    dpos_t, dneg_t = mahal2(Zt[yA == 1], mu_pos), mahal2(Zt[yA == 0], mu_neg)
    t99 = (float(np.quantile(dpos_t, 0.99)), float(np.quantile(dneg_t, 0.99)))
    Cp = np.cov(Zt[yA == 1], rowvar=False) + 1e-6 * np.eye(Zt.shape[1])
    Cn = np.cov(Zt[yA == 0], rowvar=False) + 1e-6 * np.eye(Zt.shape[1])
    Dp, Dn = Zs - mu_pos, Zs - mu_neg
    dp = np.einsum("ij,ij->i", Dp, np.linalg.solve(Cp, Dp.T).T)
    dn_ = np.einsum("ij,ij->i", Dn, np.linalg.solve(Cn, Dn.T).T)
    mask = (dp > t99[0]) & (dn_ > t99[1])
    out = blk.with_columns(pl.Series("pA", pA), pl.Series("d_pos", dp), pl.Series("d_neg", dn_)) \
        .filter(pl.Series("m", mask)).select("pair_id", "shared_hands", "risk", "r_FULL", "r_RAW", "R1", "pA", "d_pos", "d_neg") \
        .sort("d_pos", descending=True)
    out.write_csv(OUT / "p3_outliers.csv")
    res["outliers"] = {"n": out.height, "t99_pos": round(t99[0], 2), "t99_neg": round(t99[1], 2),
                       "median_sh": None if out.height == 0 else round(float(out["shared_hands"].median()), 1),
                       "R1_rate": None if out.height == 0 else round(float(out["R1"].mean()), 3),
                       "median_pA": None if out.height == 0 else round(float(out["pA"].median()), 4)}
    FA = res["A"]["F"]
    res["verdict"] = ("CLOSE block re-ordering" if FA < 150 else
                      "BUILD P4 (with P4 eligible FPs only)" if FA > 300 else
                      "INDETERMINATE: P4-Step A parity only, no new lever")
    res["meta"] = {"seed": SEED, "threads": 2, "FULL_eval_col": "r_FULL (corr(risk,r_FULL)=.956)",
                   "note": "logistic uses testbed labels (validation-allowed); GMM/Mahalanobis on eval block label-free"}
    (OUT / "p3_profile.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("A", "C", "GMM", "outliers", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
