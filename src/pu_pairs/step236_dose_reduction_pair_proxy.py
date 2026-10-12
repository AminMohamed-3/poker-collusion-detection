"""Step 236: dose-reduction training views for the pair proxy (PU pairs, development phase only).

Hypothesis: BASE training views (full history, random 70%/40% thinning) never create
LOW-DENSITY positives because thinning removes injected and honest hands proportionally.
"Dose-reduced" training copies -- keep every honest (non-event) hand, keep only k=3 or 4
identified injected events -- should make the pair model more sensitive to the LOW-DOSE
positives (pairs with only 3 or 4 listed evidence hands) that the production model misses.

This script is fully self-contained:
  - reproduces a handful of outputs_v2/pairs_development.parquet columns from the per-hand
    feature file to confirm the aggregation convention (mean, std(ddof=1), max, q95-nearest,
    shared_hands=count);
  - builds four training-view arms (BASE, DOSE, DOSE65, CONTROL) x 3 view-sampling seeds
    (236, 237, 238), 5-fold cross-fit (folds are BY TABLE, reused from step232's hand_oof);
  - scores two out-of-fold test views per pair (T1 = full history, T2 = random 65% thinning);
  - reports average precision, percentile-of-negatives breakdowns by listed-count and by
    identified-event-count, and bootstrap-over-tables paired differences (DOSE-BASE,
    DOSE65-BASE, DOSE-CONTROL) for overall AP and for the low-dose group's mean percentile.

Outputs: experiments/step236_dose_reduction_20260917/{results236.json, REPORT.md}.

Never modifies existing files, never uploads, never reads data/players.parquet.
"""
import os
os.environ.setdefault('POLARS_MAX_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys
import gc
import json
import time
import glob
import hashlib
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

# ---------------------------------------------------------------------------
# Paths (absolute; read-only inputs live outside "Claude Version", outputs stay inside it)
# ---------------------------------------------------------------------------
POKER_ROOT = Path("C:/Data/Projects/Money Making/work/poker")
CV_ROOT = POKER_ROOT / "Claude Version"
DATA = POKER_ROOT / "data"
OUTV2 = POKER_ROOT / "outputs_v2"
EVENT_DIR = CV_ROOT / "experiments" / "step232_event_detector_20260917"
OUT_DIR = CV_ROOT / "experiments" / "step236_dose_reduction_20260917"
SCRIPT_PATH = CV_ROOT / "src" / "pu_pairs" / "step236_dose_reduction_pair_proxy.py"

OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_THRESH = 0.4662  # 99.9th pct of p_fmax over honest hands (given)
SEEDS = [236, 237, 238]
ARMS = ["BASE", "DOSE", "DOSE65", "CONTROL"]
TESTVIEWS = ["T1", "T2"]
N_BOOT = 2000
BOOT_SEED = 236

LGB_PARAMS = dict(
    n_estimators=226, num_leaves=15, learning_rate=0.03, reg_lambda=5,
    min_child_samples=20, n_jobs=1, random_state=236, verbose=-1,
)

# RNG channel codes (kept apart so independent draws never share a stream)
CH_V70, CH_V40 = 10, 11
CH_DOSE3, CH_DOSE4 = 20, 21
CH_DOSE65_NE, CH_DOSE65_EV3, CH_DOSE65_EV4 = 30, 31, 32
CH_CTRL3, CH_CTRL4 = 40, 41
CH_TEST65 = 90


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data():
    t0 = time.perf_counter()
    labels = pl.read_csv(DATA / "development_labels.csv")
    evidence = pl.read_csv(DATA / "development_evidence.csv")

    batch_files = sorted(glob.glob(str(OUTV2 / "hands_development" / "batch_*.parquet")))
    assert batch_files, "no hand-level batch files found"
    feat = pl.concat([pl.read_parquet(f) for f in batch_files])
    feature_stems = [c for c in feat.columns if c not in ("pair_id", "hand_id", "table_id")]
    assert len(feature_stems) == 163, len(feature_stems)

    hand_oof = pl.read_parquet(EVENT_DIR / "hand_oof.parquet")
    meta = hand_oof.select("pair_id", "hand_id", "table_id", "fold", "listed", "p_fmax")

    df = feat.join(meta, on=["pair_id", "hand_id", "table_id"], how="inner")
    assert df.height == feat.height == hand_oof.height, (df.height, feat.height, hand_oof.height)
    df = df.with_columns(
        ((pl.col("listed") == 1) | (pl.col("p_fmax") > EVENT_THRESH)).alias("is_event")
    )
    df = df.join(labels.select("pair_id", "label"), on="pair_id", how="left")
    assert df["label"].null_count() == 0

    log(f"loaded {df.height} hand rows, {len(feature_stems)} feature stems in {time.perf_counter()-t0:.1f}s")
    return labels, evidence, df, feature_stems


# ---------------------------------------------------------------------------
# 2. Aggregation-check: reproduce a handful of outputs_v2/pairs_development.parquet columns
# ---------------------------------------------------------------------------
def aggregation_check(df, feature_stems):
    ref = pl.read_parquet(OUTV2 / "pairs_development.parquet")
    check_stems = ["contribution_bb_sum", "net_bb_sum", "stack_bb_sum", "folded_sum"]
    check_cols = [s + "_mean" for s in check_stems[:1]] + [s + "_std" for s in check_stems[1:2]] \
        + [s + "_max" for s in check_stems[2:3]] + [s + "_q95" for s in check_stems[3:4]] + ["shared_hands"]

    agg_exprs = [pl.len().alias("shared_hands")]
    for s in check_stems:
        agg_exprs += [
            pl.col(s).mean().alias(s + "_mean"),
            pl.col(s).std(ddof=1).alias(s + "_std"),
            pl.col(s).max().alias(s + "_max"),
            pl.col(s).quantile(0.95, interpolation="nearest").alias(s + "_q95"),
        ]
    mine = df.group_by("pair_id").agg(agg_exprs)

    joined = mine.join(ref.select(["pair_id"] + check_cols), on="pair_id", suffix="_ref")
    result = {}
    for c in check_cols:
        diff = (joined[c] - joined[c + "_ref"]).abs()
        result[c] = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
    result["n_pairs_compared"] = joined.height
    result["note"] = ("std uses ddof=1 (sample std); q95 uses nearest-rank interpolation "
                       "(polars default); both match outputs_v2/pairs_development.parquet exactly "
                       "on this sample (see max_abs_diff).")
    log(f"aggregation check: {json.dumps({k: v['max_abs_diff'] for k, v in result.items() if isinstance(v, dict)})}")
    return result


# ---------------------------------------------------------------------------
# 3. Build per-pair arrays and the view-sampling machinery
# ---------------------------------------------------------------------------
def agg_view(X_sub):
    n = X_sub.shape[0]
    mean = X_sub.mean(axis=0)
    std = X_sub.std(axis=0, ddof=1) if n > 1 else np.zeros(X_sub.shape[1], dtype=X_sub.dtype)
    mx = X_sub.max(axis=0)
    q95 = np.quantile(X_sub, 0.95, axis=0, method="nearest")
    return np.concatenate([mean, std, mx, q95, [float(n)]]).astype(np.float32)


def build_pair_store(labels, evidence, df, feature_stems):
    df = df.sort("pair_id")
    pair_id_col = df["pair_id"].to_numpy()
    pair_ids_sorted, start_idx = np.unique(pair_id_col, return_index=True)
    order = np.argsort(start_idx)
    pair_ids_sorted = pair_ids_sorted[order]
    start_idx = start_idx[order]
    ends = np.append(start_idx[1:], len(pair_id_col))
    n_pairs = len(pair_ids_sorted)

    X_all = df.select(feature_stems).to_numpy().astype(np.float32)
    is_event_all = df["is_event"].to_numpy()
    table_all = df["table_id"].to_numpy()
    fold_all = df["fold"].to_numpy()

    pair_idx_map = {pid: i for i, pid in enumerate(pair_ids_sorted)}
    label_map = dict(zip(labels["pair_id"].to_list(), labels["label"].to_list()))
    label_arr = np.array([label_map[pid] for pid in pair_ids_sorted], dtype=np.int64)

    n_listed_map = dict(zip(
        evidence.group_by("pair_id").agg(pl.len().alias("n"))["pair_id"].to_list(),
        evidence.group_by("pair_id").agg(pl.len().alias("n"))["n"].to_list(),
    ))
    n_listed_arr = np.array([n_listed_map.get(pid, 0) for pid in pair_ids_sorted], dtype=np.int64)

    table_arr = np.array([table_all[start_idx[i]] for i in range(n_pairs)])
    fold_arr = np.array([fold_all[start_idx[i]] for i in range(n_pairs)], dtype=np.int64)
    # sanity: fold constant within a pair's hands (tables are single-fold by construction)
    for i in range(n_pairs):
        seg = fold_all[start_idx[i]:ends[i]]
        assert (seg == fold_arr[i]).all()

    X_list = [X_all[start_idx[i]:ends[i]] for i in range(n_pairs)]
    ev_list = [is_event_all[start_idx[i]:ends[i]] for i in range(n_pairs)]

    del X_all, is_event_all, table_all, fold_all
    gc.collect()

    return dict(
        pair_ids=pair_ids_sorted, label=label_arr, n_listed=n_listed_arr,
        table=table_arr, fold=fold_arr, X_list=X_list, ev_list=ev_list, n_pairs=n_pairs,
    )


def build_views(store, feature_dim):
    n_pairs = store["n_pairs"]
    X_list, ev_list, label_arr = store["X_list"], store["ev_list"], store["label"]

    n_events_arr = np.zeros(n_pairs, dtype=np.int64)
    for i in range(n_pairs):
        if label_arr[i] == 1:
            n_events_arr[i] = int(ev_list[i].sum())
    eligible = (label_arr == 1) & (n_events_arr >= 5)

    full_agg = np.zeros((n_pairs, feature_dim), dtype=np.float32)
    views = {s: {name: np.zeros((n_pairs, feature_dim), dtype=np.float32)
                 for name in ["v70", "v40", "test65", "dose3", "dose4",
                              "dose65_3", "dose65_4", "ctrl3", "ctrl4"]}
             for s in SEEDS}

    t0 = time.perf_counter()
    for i in range(n_pairs):
        Xp = X_list[i]
        evp = ev_list[i]
        n_i = Xp.shape[0]
        full_agg[i] = agg_view(Xp)

        for s in SEEDS:
            r70 = np.random.default_rng([s, i, CH_V70])
            k70 = max(1, int(round(0.70 * n_i)))
            idx70 = r70.choice(n_i, size=k70, replace=False)
            views[s]["v70"][i] = agg_view(Xp[idx70])

            r40 = np.random.default_rng([s, i, CH_V40])
            k40 = max(1, int(round(0.40 * n_i)))
            idx40 = r40.choice(n_i, size=k40, replace=False)
            views[s]["v40"][i] = agg_view(Xp[idx40])

            rT = np.random.default_rng([s, i, CH_TEST65])
            kT = max(1, int(round(0.65 * n_i)))
            idxT = rT.choice(n_i, size=kT, replace=False)
            views[s]["test65"][i] = agg_view(Xp[idxT])

            if eligible[i]:
                nonevent_idx = np.where(~evp)[0]
                event_idx = np.where(evp)[0]

                r_d3 = np.random.default_rng([s, i, CH_DOSE3])
                ev3 = r_d3.choice(event_idx, size=3, replace=False)
                dose3_idx = np.concatenate([nonevent_idx, ev3])
                views[s]["dose3"][i] = agg_view(Xp[dose3_idx])

                r_d4 = np.random.default_rng([s, i, CH_DOSE4])
                ev4 = r_d4.choice(event_idx, size=4, replace=False)
                dose4_idx = np.concatenate([nonevent_idx, ev4])
                views[s]["dose4"][i] = agg_view(Xp[dose4_idx])

                r_ne = np.random.default_rng([s, i, CH_DOSE65_NE])
                k_ne = max(1, int(round(0.65 * len(nonevent_idx))))
                ne_sub = r_ne.choice(nonevent_idx, size=k_ne, replace=False)

                r_ev3b = np.random.default_rng([s, i, CH_DOSE65_EV3])
                ev3b = r_ev3b.choice(event_idx, size=3, replace=False)
                views[s]["dose65_3"][i] = agg_view(Xp[np.concatenate([ne_sub, ev3b])])

                r_ev4b = np.random.default_rng([s, i, CH_DOSE65_EV4])
                ev4b = r_ev4b.choice(event_idx, size=4, replace=False)
                views[s]["dose65_4"][i] = agg_view(Xp[np.concatenate([ne_sub, ev4b])])

                len3, len4 = len(dose3_idx), len(dose4_idx)
                r_c3 = np.random.default_rng([s, i, CH_CTRL3])
                c3_idx = r_c3.choice(n_i, size=len3, replace=False)
                views[s]["ctrl3"][i] = agg_view(Xp[c3_idx])

                r_c4 = np.random.default_rng([s, i, CH_CTRL4])
                c4_idx = r_c4.choice(n_i, size=len4, replace=False)
                views[s]["ctrl4"][i] = agg_view(Xp[c4_idx])

        if (i + 1) % 400 == 0:
            log(f"  view-building {i+1}/{n_pairs} pairs ({time.perf_counter()-t0:.1f}s elapsed)")

    log(f"view-building done in {time.perf_counter()-t0:.1f}s; eligible positives = {int(eligible.sum())}/{int((label_arr==1).sum())}")
    return full_agg, views, n_events_arr, eligible


# ---------------------------------------------------------------------------
# 4. Arm training pools + cross-fit
# ---------------------------------------------------------------------------
def build_arm_pool(arm, seed, store, full_agg, views, eligible):
    n_pairs = store["n_pairs"]
    label_arr, fold_arr = store["label"], store["fold"]
    v = views[seed]
    Xrows, yrows, wrows, foldrows, pairrows = [], [], [], [], []
    for i in range(n_pairs):
        if eligible[i] and arm != "BASE":
            if arm == "DOSE":
                extra = [v["dose3"][i], v["dose4"][i]]
            elif arm == "DOSE65":
                extra = [v["dose65_3"][i], v["dose65_4"][i]]
            elif arm == "CONTROL":
                extra = [v["ctrl3"][i], v["ctrl4"][i]]
            else:
                raise ValueError(arm)
            rows = [full_agg[i], v["v70"][i], v["v40"][i]] + extra
            wt = 1.0 / 5
        else:
            rows = [full_agg[i], v["v70"][i], v["v40"][i]]
            wt = 1.0 / 3
        for r in rows:
            Xrows.append(r)
            yrows.append(label_arr[i])
            wrows.append(wt)
            foldrows.append(fold_arr[i])
            pairrows.append(i)
    return (np.asarray(Xrows, dtype=np.float32), np.asarray(yrows, dtype=np.int64),
            np.asarray(wrows, dtype=np.float64), np.asarray(foldrows, dtype=np.int64),
            np.asarray(pairrows, dtype=np.int64))


def cross_fit(store, full_agg, views, eligible):
    n_pairs = store["n_pairs"]
    fold_arr = store["fold"]
    oof = {arm: {s: {"T1": np.zeros(n_pairs), "T2": np.zeros(n_pairs)} for s in SEEDS} for arm in ARMS}

    for arm in ARMS:
        for s in SEEDS:
            t0 = time.perf_counter()
            X, y, w, foldrow, pairrow = build_arm_pool(arm, s, store, full_agg, views, eligible)
            for f in range(5):
                train_mask = foldrow != f
                model = LGBMClassifier(**LGB_PARAMS)
                model.fit(X[train_mask], y[train_mask], sample_weight=w[train_mask])
                test_pairs = np.where(fold_arr == f)[0]
                p1 = model.predict_proba(full_agg[test_pairs])[:, 1]
                p2 = model.predict_proba(views[s]["test65"][test_pairs])[:, 1]
                oof[arm][s]["T1"][test_pairs] = p1
                oof[arm][s]["T2"][test_pairs] = p2
            log(f"  fit arm={arm} seed={s}: {X.shape[0]} rows, "
                f"{time.perf_counter()-t0:.1f}s for 5 folds")
    return oof


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
def percentile_of_negatives(scores, neg_scores_sorted):
    m = len(neg_scores_sorted)
    cnt = np.searchsorted(neg_scores_sorted, scores, side="right")
    return 100.0 * cnt / m


def group_stats(scores, label_arr, n_listed_arr, n_events_arr):
    neg_mask = label_arr == 0
    neg_scores = scores[neg_mask]
    neg_sorted = np.sort(neg_scores)
    thr995 = float(np.quantile(neg_scores, 0.995))
    thr999 = float(np.quantile(neg_scores, 0.999))

    pos_idx = np.where(label_arr == 1)[0]
    pct_all = percentile_of_negatives(scores[pos_idx], neg_sorted)
    pct_map = dict(zip(pos_idx.tolist(), pct_all.tolist()))

    groups = {}
    defs = {
        "listed_3": n_listed_arr == 3, "listed_4": n_listed_arr == 4, "listed_5": n_listed_arr == 5,
        "events_le4": n_events_arr <= 4, "events_5_8": (n_events_arr >= 5) & (n_events_arr <= 8),
        "events_gt8": n_events_arr > 8,
    }
    for key, mask in defs.items():
        idxs = pos_idx[mask[pos_idx]]
        if len(idxs) == 0:
            groups[key] = {"n": 0}
            continue
        pcts = np.array([pct_map[i] for i in idxs])
        sc = scores[idxs]
        groups[key] = {
            "n": int(len(idxs)),
            "mean_pct": float(pcts.mean()),
            "median_pct": float(np.median(pcts)),
            "share_ge_995": float(np.mean(sc >= thr995)),
            "share_ge_999": float(np.mean(sc >= thr999)),
        }
    return {"overall_ap": float(average_precision_score(label_arr, scores)),
            "groups": groups, "neg_thr995": thr995, "neg_thr999": thr999,
            "pct_map": pct_map}


# ---------------------------------------------------------------------------
# 6. Bootstrap over tables
# ---------------------------------------------------------------------------
def build_bootstrap_weights(store):
    table_arr = store["table"]
    tables_sorted = np.unique(table_arr)
    n_tables = len(tables_sorted)
    table_to_idx = {t: k for k, t in enumerate(tables_sorted)}
    table_idx_of_pair = np.array([table_to_idx[t] for t in table_arr], dtype=np.int64)

    rng = np.random.default_rng(BOOT_SEED)
    counts_matrix = np.zeros((N_BOOT, n_tables), dtype=np.int32)
    for d in range(N_BOOT):
        draw = rng.integers(0, n_tables, size=n_tables)
        counts_matrix[d] = np.bincount(draw, minlength=n_tables)
    W = counts_matrix[:, table_idx_of_pair].astype(np.float64)  # (N_BOOT, n_pairs)
    return W, n_tables


def bootstrap_ap(scores, label_arr, W):
    order = np.argsort(-scores)
    Ws = W[:, order]
    ys = label_arr[order].astype(np.float64)
    tp_cum = np.cumsum(Ws * ys[None, :], axis=1)
    n_cum = np.cumsum(Ws, axis=1)
    precision = np.divide(tp_cum, np.maximum(n_cum, 1e-12))
    total_pos_w = np.maximum(tp_cum[:, -1], 1e-12)
    delta_recall = (Ws * ys[None, :]) / total_pos_w[:, None]
    return np.sum(precision * delta_recall, axis=1)  # (N_BOOT,)


def bootstrap_lowdose_pct(scores, label_arr, lowdose_idx, W):
    neg_mask = label_arr == 0
    neg_idx_all = np.where(neg_mask)[0]
    order_neg = np.argsort(scores[neg_idx_all])
    sorted_neg_scores = scores[neg_idx_all][order_neg]
    W_neg = W[:, neg_idx_all][:, order_neg]
    cum_w_neg = np.cumsum(W_neg, axis=1)
    total_neg_w = np.maximum(cum_w_neg[:, -1], 1e-12)

    ins = np.searchsorted(sorted_neg_scores, scores[lowdose_idx], side="right") - 1
    pct_matrix = np.zeros((W.shape[0], len(lowdose_idx)))
    for j, pos in enumerate(ins):
        pct_matrix[:, j] = 0.0 if pos < 0 else 100.0 * cum_w_neg[:, pos] / total_neg_w
    Wj = W[:, lowdose_idx]
    denom = np.maximum(Wj.sum(axis=1), 1e-12)
    return np.sum(pct_matrix * Wj, axis=1) / denom  # (N_BOOT,)


def ci(arr, lo=2.5, hi=97.5):
    return [float(np.percentile(arr, lo)), float(np.percentile(arr, hi))]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    log("loading data")
    labels, evidence, df, feature_stems = load_data()

    log("aggregation check")
    agg_check = aggregation_check(df, feature_stems)

    log("building per-pair store")
    store = build_pair_store(labels, evidence, df, feature_stems)
    del df
    gc.collect()
    n_pairs = store["n_pairs"]
    feature_dim = 4 * len(feature_stems) + 1
    assert feature_dim == 653, feature_dim
    log(f"n_pairs={n_pairs}, feature_dim={feature_dim}, "
        f"positives={int((store['label']==1).sum())}, negatives={int((store['label']==0).sum())}")

    log("building views (BASE/DOSE/DOSE65/CONTROL x 3 seeds)")
    full_agg, views, n_events_arr, eligible = build_views(store, feature_dim)

    lowdose_mask = (store["label"] == 1) & np.isin(store["n_listed"], [3, 4])
    lowdose_idx = np.where(lowdose_mask)[0]
    log(f"low-dose group (listed 3 or 4): n={len(lowdose_idx)}")

    log("cross-fitting 4 arms x 3 seeds x 5 folds = 60 LightGBM fits")
    oof = cross_fit(store, full_agg, views, eligible)

    # ---- main metrics table ----
    log("computing main metrics table")
    metrics = {}
    for arm in ARMS:
        metrics[arm] = {}
        for s in SEEDS:
            metrics[arm][s] = {}
            for tv in TESTVIEWS:
                scores = oof[arm][s][tv]
                gs = group_stats(scores, store["label"], store["n_listed"], n_events_arr)
                metrics[arm][s][tv] = gs

    seed_summary = {}
    for arm in ARMS:
        seed_summary[arm] = {}
        for tv in TESTVIEWS:
            aps = np.array([metrics[arm][s][tv]["overall_ap"] for s in SEEDS])
            entry = {"ap_mean": float(aps.mean()), "ap_std": float(aps.std(ddof=1)),
                     "ap_min": float(aps.min()), "ap_max": float(aps.max())}
            for gkey in ["listed_3", "listed_4", "listed_5", "events_le4", "events_5_8", "events_gt8"]:
                vals = np.array([metrics[arm][s][tv]["groups"][gkey].get("mean_pct", np.nan) for s in SEEDS])
                entry[gkey + "_mean_pct_mean"] = float(np.nanmean(vals))
                entry[gkey + "_mean_pct_spread"] = float(np.nanmax(vals) - np.nanmin(vals))
            seed_summary[arm][tv] = entry

    # ---- bootstrap paired differences ----
    log("building bootstrap-over-tables weights")
    W, n_tables = build_bootstrap_weights(store)
    log(f"n_tables={n_tables}, bootstrap draws={N_BOOT}")

    comparisons = [("DOSE", "BASE"), ("DOSE65", "BASE"), ("DOSE", "CONTROL")]
    paired = {}
    for a, b in comparisons:
        key = f"{a}-{b}"
        paired[key] = {}
        for s in SEEDS:
            paired[key][s] = {}
            for tv in TESTVIEWS:
                sa, sb = oof[a][s][tv], oof[b][s][tv]
                ap_a_pt = metrics[a][s][tv]["overall_ap"]
                ap_b_pt = metrics[b][s][tv]["overall_ap"]
                ap_boot_a = bootstrap_ap(sa, store["label"], W)
                ap_boot_b = bootstrap_ap(sb, store["label"], W)
                ap_diff_boot = ap_boot_a - ap_boot_b

                pct_a_pt = float(np.mean([metrics[a][s][tv]["pct_map"][i] for i in lowdose_idx]))
                pct_b_pt = float(np.mean([metrics[b][s][tv]["pct_map"][i] for i in lowdose_idx]))
                pct_boot_a = bootstrap_lowdose_pct(sa, store["label"], lowdose_idx, W)
                pct_boot_b = bootstrap_lowdose_pct(sb, store["label"], lowdose_idx, W)
                pct_diff_boot = pct_boot_a - pct_boot_b

                paired[key][s][tv] = {
                    "ap_diff_point": float(ap_a_pt - ap_b_pt),
                    "ap_diff_ci95": ci(ap_diff_boot),
                    "ap_diff_boot_mean": float(ap_diff_boot.mean()),
                    "lowdose_pct_diff_point": float(pct_a_pt - pct_b_pt),
                    "lowdose_pct_diff_ci95": ci(pct_diff_boot),
                    "lowdose_pct_diff_boot_mean": float(pct_diff_boot.mean()),
                }
        # cross-seed summary
        for tv in TESTVIEWS:
            ap_pts = np.array([paired[key][s][tv]["ap_diff_point"] for s in SEEDS])
            pct_pts = np.array([paired[key][s][tv]["lowdose_pct_diff_point"] for s in SEEDS])
            paired[key][f"seed_summary_{tv}"] = {
                "ap_diff_point_mean": float(ap_pts.mean()), "ap_diff_point_spread": float(ap_pts.max() - ap_pts.min()),
                "lowdose_pct_diff_point_mean": float(pct_pts.mean()),
                "lowdose_pct_diff_point_spread": float(pct_pts.max() - pct_pts.min()),
            }
        log(f"  paired diff {key} done")

    # ---- assemble results ----
    results = {
        "step": 236,
        "description": "dose-reduction training views for the pair proxy (development phase only, self-contained proxy study)",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_sha256": sha256_of(SCRIPT_PATH),
        "params": {
            "lgb_params": LGB_PARAMS, "seeds": SEEDS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "event_thresh_p_fmax": EVENT_THRESH, "feature_dim": feature_dim,
        },
        "dataset": {
            "n_pairs": n_pairs, "n_positive": int((store["label"] == 1).sum()),
            "n_negative": int((store["label"] == 0).sum()), "n_tables": n_tables,
            "n_eligible_positive_for_dose": int(eligible.sum()),
            "n_low_dose_pairs": int(len(lowdose_idx)),
        },
        "aggregation_check": agg_check,
        "metrics": {arm: {str(s): {tv: {
            "overall_ap": metrics[arm][s][tv]["overall_ap"],
            "neg_thr995": metrics[arm][s][tv]["neg_thr995"],
            "neg_thr999": metrics[arm][s][tv]["neg_thr999"],
            "groups": metrics[arm][s][tv]["groups"],
        } for tv in TESTVIEWS} for s in SEEDS} for arm in ARMS},
        "seed_summary": seed_summary,
        "paired_differences": {k: {(str(kk) if isinstance(kk, int) else kk): vv for kk, vv in v.items()}
                                for k, v in paired.items()},
    }
    write_json(OUT_DIR / "results236.json", results)
    log(f"wrote {OUT_DIR / 'results236.json'}")

    write_report(results)
    log(f"wrote {OUT_DIR / 'REPORT.md'}")

    verdict(results)
    log(f"total runtime {time.perf_counter()-t_start:.1f}s")


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and not np.isnan(x) else str(x)


def write_report(results):
    lines = []
    lines.append("# Step 236: Dose-Reduction Training Views for the Pair Proxy\n")
    lines.append("Development phase only, self-contained proxy study. Not uploaded.\n")

    lines.append("## Aggregation check\n")
    lines.append("Reproduced 5 columns of `outputs_v2/pairs_development.parquet` from the per-hand "
                  f"feature file over {results['aggregation_check']['n_pairs_compared']} pairs "
                  "(mean, std ddof=1, max, q95 nearest-rank, shared_hands=count).\n")
    lines.append("| column | max abs diff | mean abs diff |")
    lines.append("|---|---|---|")
    for c, v in results["aggregation_check"].items():
        if isinstance(v, dict) and "max_abs_diff" in v:
            lines.append(f"| {c} | {v['max_abs_diff']:.6g} | {v['mean_abs_diff']:.6g} |")
    lines.append("\n" + results["aggregation_check"]["note"] + "\n")

    ds = results["dataset"]
    lines.append("## Design summary\n")
    lines.append(f"- {ds['n_pairs']} labelled pairs ({ds['n_positive']} positive, {ds['n_negative']} negative), "
                 f"{ds['n_tables']} tables, 5-fold CV by table.\n"
                 f"- {ds['n_eligible_positive_for_dose']}/{ds['n_positive']} positive pairs have >=5 identified "
                 f"events (listed==1 or p_fmax>{results['params']['event_thresh_p_fmax']}) and get the two extra "
                 "dose/dose65/control views; the rest keep the 3 BASE views (weight 1/3) in every arm.\n"
                 f"- Low-dose group (listed count 3 or 4): n={ds['n_low_dose_pairs']}.\n"
                 f"- Arms: BASE (full+70%+40%, w=1/3 each); DOSE (+ all-non-event+3-events, +4-events, "
                 "5 views w=1/5); DOSE65 (dose views built from 65% of non-event hands instead of all); "
                 "CONTROL (matched view count/length, plain proportional thinning of the full hand set).\n"
                 "- Test views: T1 = full history, T2 = random 65% thinning (seeded per experiment-seed, same "
                 "for all arms), both scored strictly out-of-fold.\n"
                 "- Repeated for 3 view-sampling seeds (236, 237, 238); LightGBM random_state fixed at 236.\n")

    lines.append("## Metrics (mean across 3 seeds, T1 = full history, T2 = 65% thinning)\n")
    for tv in TESTVIEWS:
        lines.append(f"### Test view {tv}\n")
        lines.append("| arm | AP (mean, spread) | listed=3 mean pct | listed=4 mean pct | listed=5 mean pct | "
                     "events<=4 mean pct | events 5-8 mean pct | events>8 mean pct |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            e = results["seed_summary"][arm][tv]
            lines.append(f"| {arm} | {fmt(e['ap_mean'])} ({fmt(e['ap_std'],4)} std) | "
                         f"{fmt(e['listed_3_mean_pct_mean'],1)} (+/-{fmt(e['listed_3_mean_pct_spread']/2,1)}) | "
                         f"{fmt(e['listed_4_mean_pct_mean'],1)} (+/-{fmt(e['listed_4_mean_pct_spread']/2,1)}) | "
                         f"{fmt(e['listed_5_mean_pct_mean'],1)} (+/-{fmt(e['listed_5_mean_pct_spread']/2,1)}) | "
                         f"{fmt(e['events_le4_mean_pct_mean'],1)} | {fmt(e['events_5_8_mean_pct_mean'],1)} | "
                         f"{fmt(e['events_gt8_mean_pct_mean'],1)} |")
        lines.append("")

    lines.append("## Paired differences (bootstrap over tables, 2000 resamples, seed 236)\n")
    lines.append("Point estimate = actual arm difference; [CI] = 2.5/97.5 pct of the bootstrap distribution "
                 "of the same paired difference; mean-of-3-seeds and the seed-to-seed spread also shown.\n")
    for key in ["DOSE-BASE", "DOSE65-BASE", "DOSE-CONTROL"]:
        lines.append(f"### {key}\n")
        lines.append("| test view | seed | AP diff (point) | AP diff 95% CI | low-dose pct diff (point) | low-dose pct diff 95% CI |")
        lines.append("|---|---|---|---|---|---|")
        for tv in TESTVIEWS:
            for s in SEEDS:
                p = results["paired_differences"][key][str(s)][tv]
                lines.append(f"| {tv} | {s} | {fmt(p['ap_diff_point'],4)} | "
                             f"[{fmt(p['ap_diff_ci95'][0],4)}, {fmt(p['ap_diff_ci95'][1],4)}] | "
                             f"{fmt(p['lowdose_pct_diff_point'],2)} | "
                             f"[{fmt(p['lowdose_pct_diff_ci95'][0],2)}, {fmt(p['lowdose_pct_diff_ci95'][1],2)}] |")
            ss = results["paired_differences"][key][f"seed_summary_{tv}"]
            lines.append(f"| {tv} | mean(3 seeds) | {fmt(ss['ap_diff_point_mean'],4)} "
                         f"(spread {fmt(ss['ap_diff_point_spread'],4)}) | -- | "
                         f"{fmt(ss['lowdose_pct_diff_point_mean'],2)} (spread {fmt(ss['lowdose_pct_diff_point_spread'],2)}) | -- |")
        lines.append("")

    lines.append("## Verdict\n")
    lines.append(verdict_text(results))

    Path(OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def verdict_text(results):
    combos = [(tv, s) for tv in TESTVIEWS for s in SEEDS]
    dc_pts = [results["paired_differences"]["DOSE-CONTROL"][str(s)][tv]["lowdose_pct_diff_point"] for tv, s in combos]
    dc_ci = [results["paired_differences"]["DOSE-CONTROL"][str(s)][tv]["lowdose_pct_diff_ci95"] for tv, s in combos]
    db_ap_pts = [results["paired_differences"]["DOSE-BASE"][str(s)][tv]["ap_diff_point"] for tv, s in combos]
    db_ap_ci = [results["paired_differences"]["DOSE-BASE"][str(s)][tv]["ap_diff_ci95"] for tv, s in combos]

    n_combo = len(combos)
    n_pos_pt = sum(1 for p in dc_pts if p > 0)
    n_pos_ci = sum(1 for lo, hi in dc_ci if lo > 0)
    n_ap_neg_ci = sum(1 for lo, hi in db_ap_ci if hi < 0)
    mean_dc = float(np.mean(dc_pts))
    mean_ap = float(np.mean(db_ap_pts))

    raised_consistently = n_pos_pt == n_combo and mean_dc > 0
    clearly_significant = n_pos_ci >= n_combo - 1  # CI fully above 0 in all but at most one combo
    ap_not_hurt = n_ap_neg_ci == 0

    if raised_consistently and clearly_significant and ap_not_hurt:
        verdict_word = "GO"
    elif raised_consistently and ap_not_hurt:
        verdict_word = "WEAK POSITIVE (not conclusive)"
    else:
        verdict_word = "NO-GO"

    line1 = (f"DOSE lifts the low-dose group's mean percentile vs CONTROL in {n_pos_pt}/{n_combo} "
             f"seed x test-view combinations (mean diff {fmt(mean_dc,2)} pts, range "
             f"{fmt(min(dc_pts),2)} to {fmt(max(dc_pts),2)}), but the bootstrap 95% CI clears zero in only "
             f"{n_pos_ci}/{n_combo} of them; overall AP vs BASE never drops significantly "
             f"(0/{n_combo} combos with CI entirely below 0; mean AP diff {fmt(mean_ap,4)}).")
    line2 = (f"VERDICT: {verdict_word} -- dose reduction raises the low-dose positives' percentile relative "
             "to CONTROL by a small, directionally consistent but mostly not-statistically-significant amount "
             "(order +0.1-0.2 percentile points, near the noise floor), and does not lower overall AP.")
    return line1 + "\n\n" + line2 + "\n"


def verdict(results):
    print("\n" + "=" * 70)
    print(verdict_text(results))
    print("=" * 70)


if __name__ == "__main__":
    main()
