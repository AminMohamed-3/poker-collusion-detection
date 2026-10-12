"""Step 247: pair-level aggregates of the 25 new per-hand invisible-event features (PU pairs,
development phase only).

Question: do PAIR-LEVEL aggregates (mean/std/max/q95) of the 25 step237 per-hand features improve
a pair classifier over the 163-feature baseline, especially for LOW-DOSE pairs (3 or 4 listed
evidence hands) and at evaluation-like history length (T2 = 65% thinning)?

Reuses the machinery of step236_dose_reduction_pair_proxy.py: same per-hand -> per-pair aggregation
convention (mean, std ddof=1, max, q95 nearest-rank, shared_hands=count), same BASE training views
(full history, random 70%, random 40% thinning, weight 1/3 each), same 5-fold cross-fit BY TABLE
(fold column of experiments/step232_event_detector_20260917/hand_oof.parquet), same LightGBM recipe
(n_estimators=226, num_leaves=15, learning_rate=.03, reg_lambda=5, min_child_samples=20, n_jobs=1),
same two out-of-fold test views (T1 = full history, T2 = random 65% thinning), same view-sampling
seeds (236, 237, 238) and the SAME RNG channel codes for v70/v40/test65, so the OLD arm reproduces
step236's BASE arm bit-for-bit (checked in the report).

New per-hand features come from experiments/step237_invisible_events_20260917/step237_features.parquet
(223,749 rows, keyed by pair_id, hand_id, 25 Float32 columns, no nulls). They are joined onto the
SAME sorted per-hand frame used to build the old 163-feature per-pair blocks, via an explicit
(pair_id, hand_id) key lookup (not a bare polars join) so the row order used for view-sampling is
guaranteed identical between the old block and the new block.

Arms (all built on top of the identical OLD 653-dim block; same views/folds/seeds for every arm):
  OLD    = 4*163 old features + shared_hands (653 dims) = step236 BASE.
  NEW25  = OLD + mean/std/max/q95 of all 25 new features (+100 dims -> 753).
  ISO1   = OLD + mean/std/max/q95 of iso_weak_agg_behind only (+4 dims -> 657).
  NEW19  = OLD + mean/std/max/q95 of the 19 new features that do not read an outsider's hole
           cards (+76 dims -> 729). The dropped six are step237_features.OUTSIDER_CARD_FEATURES =
           fp_best_live, fp_wouldwin, fp_best_pot, oof_clearway, passive_winner, checkdown_best
           (confirmed against src/evidence_aug/step237_features.py and step237 REPORT.md section 5).
  PERM   = NEW25 with the 25 new per-hand feature values shuffled ACROSS ALL PAIRS at the hand
           level (one global row permutation, seed 247, applied before aggregation) -- a control
           that destroys the real pair-hand correspondence but keeps the column count and the
           per-hand marginal distribution of each new feature.

This script is meant to be invoked several times (it is expensive: 5 arms x 3 seeds x 5 folds = 75
LightGBM fits) so each invocation stays well inside a single foreground run:

    python step247_pair_aggregates_new_block.py cache            # ~1-2 min, builds cache/blocks.npz
    python step247_pair_aggregates_new_block.py arm OLD           # ~1-2 min each,
    python step247_pair_aggregates_new_block.py arm NEW25         #   checkpoints per (arm, seed)
    python step247_pair_aggregates_new_block.py arm ISO1          #   in cache/oof_<ARM>_<seed>.npz
    python step247_pair_aggregates_new_block.py arm NEW19
    python step247_pair_aggregates_new_block.py arm PERM
    python step247_pair_aggregates_new_block.py report            # ~1 min, metrics + bootstrap + report

Every stage is idempotent / resumable: a stage skips work whose cache file already exists, so a
timed-out or interrupted invocation can simply be re-run.

Outputs: experiments/step247_pair_new_block_20260917/{cache/*, results247.json, REPORT.txt}.
Never modifies existing files, never uploads, never reads data/players.parquet. Development phase
only (all inputs are the development_* files / development hand caches already scoped to that
phase; there is no separate phase column to assert on at this level).
"""
import os
os.environ.setdefault('POLARS_MAX_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

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
# Paths
# ---------------------------------------------------------------------------
POKER_ROOT = Path("C:/Data/Projects/Money Making/work/poker")
CV_ROOT = POKER_ROOT / "Claude Version"
DATA = POKER_ROOT / "data"
OUTV2 = POKER_ROOT / "outputs_v2"
EVENT_DIR = CV_ROOT / "experiments" / "step232_event_detector_20260917"
STEP237_DIR = CV_ROOT / "experiments" / "step237_invisible_events_20260917"
OUT_DIR = CV_ROOT / "experiments" / "step247_pair_new_block_20260917"
CACHE_DIR = OUT_DIR / "cache"
SCRIPT_PATH = CV_ROOT / "src" / "pu_pairs" / "step247_pair_aggregates_new_block.py"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [236, 237, 238]
ARMS = ["OLD", "NEW25", "ISO1", "NEW19", "PERM"]
TESTVIEWS = ["T1", "T2"]
N_BOOT = 2000
BOOT_SEED = 247
PERM_SEED = 247
TOP1_Q = 0.99  # "top-scoring 1% of negatives" threshold

LGB_PARAMS = dict(
    n_estimators=226, num_leaves=15, learning_rate=0.03, reg_lambda=5,
    min_child_samples=20, n_jobs=1, random_state=236, verbose=-1,
    importance_type="gain",
)

# RNG channel codes -- MUST match step236_dose_reduction_pair_proxy.py exactly so that the OLD arm
# reproduces step236's BASE arm bit-for-bit.
CH_V70, CH_V40 = 10, 11
CH_TEST65 = 90

FAMILIES = ["directed_transfer", "soft_play", "coordinated_isolation"]

# The 25 new features, in the exact order of step237_features.parquet / step237_features.py.
NEW_FEATURES = [
    'fp_best_live', 'fp_gap', 'fp_premium_pf', 'fp_price_min', 'fp_wouldwin', 'fp_best_pot',
    'fp_partner_wins', 'fp_betfold', 'fp_limpfold', 'fp_count',
    'iso_out_fold_share', 'iso_clean', 'iso_weak_agg_behind', 'iso_agg_partner_folds',
    'oof_clearway',
    'tr_internal_bb', 'tr_loser_share', 'hu_pair_streets', 'pay_partner_bb', 'passive_winner',
    'checkdown_best', 'coldcall_partner_bb', 'call_partner_dominated_bb', 'both_agg_live',
    'pair_pot_share',
]
OUTSIDER_CARD_FEATURES = ['fp_best_live', 'fp_wouldwin', 'fp_best_pot', 'oof_clearway',
                          'passive_winner', 'checkdown_best']
assert len(NEW_FEATURES) == 25 and len(OUTSIDER_CARD_FEATURES) == 6
ISO_MASK = np.array([f == 'iso_weak_agg_behind' for f in NEW_FEATURES])
KEEP19_MASK = np.array([f not in OUTSIDER_CARD_FEATURES for f in NEW_FEATURES])
assert ISO_MASK.sum() == 1 and KEEP19_MASK.sum() == 19


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------
def agg_block(Xsub):
    """mean/std(ddof=1)/max/q95(nearest) over axis 0, concatenated (4*ncols,). No shared_hands."""
    n = Xsub.shape[0]
    mean = Xsub.mean(axis=0)
    std = Xsub.std(axis=0, ddof=1) if n > 1 else np.zeros(Xsub.shape[1], dtype=Xsub.dtype)
    mx = Xsub.max(axis=0)
    q95 = np.quantile(Xsub, 0.95, axis=0, method="nearest")
    return np.concatenate([mean, std, mx, q95]).astype(np.float32)


def block_names(names):
    return [n + "_mean" for n in names] + [n + "_std" for n in names] \
        + [n + "_max" for n in names] + [n + "_q95" for n in names]


def block_subset(block, mask):
    """block: (..., 4*len(mask)) array; mask: (len(mask),) bool. Returns (..., 4*mask.sum())."""
    k = mask.sum()
    pos = np.where(mask)[0]
    idx = np.concatenate([pos + len(mask) * b for b in range(4)])
    return block[..., idx]


def subset_names(names, mask):
    kept = [n for n, m in zip(names, mask) if m]
    return block_names(kept)


def assemble(arm, old_b, newreal_b, newperm_b, n_arr):
    n_col = n_arr[:, None].astype(np.float32)
    if arm == "OLD":
        return np.concatenate([old_b, n_col], axis=1)
    if arm == "NEW25":
        return np.concatenate([old_b, newreal_b, n_col], axis=1)
    if arm == "ISO1":
        return np.concatenate([old_b, block_subset(newreal_b, ISO_MASK), n_col], axis=1)
    if arm == "NEW19":
        return np.concatenate([old_b, block_subset(newreal_b, KEEP19_MASK), n_col], axis=1)
    if arm == "PERM":
        return np.concatenate([old_b, newperm_b, n_col], axis=1)
    raise ValueError(arm)


def feature_names_for(arm, feature_stems):
    old_names = block_names(feature_stems)
    if arm == "OLD":
        return old_names + ["shared_hands"]
    if arm in ("NEW25", "PERM"):
        return old_names + block_names(NEW_FEATURES) + ["shared_hands"]
    if arm == "ISO1":
        return old_names + subset_names(NEW_FEATURES, ISO_MASK) + ["shared_hands"]
    if arm == "NEW19":
        return old_names + subset_names(NEW_FEATURES, KEEP19_MASK) + ["shared_hands"]
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Stage: cache -- load data, build per-pair blocks, save to disk
# ---------------------------------------------------------------------------
def stage_cache():
    if (CACHE_DIR / "blocks.npz").exists() and (CACHE_DIR / "store_meta.json").exists():
        log("cache already built, skipping (delete cache/ to force a rebuild)")
        return

    t0 = time.perf_counter()
    log("loading labels/evidence/old features/hand_oof/new features")
    labels = pl.read_csv(DATA / "development_labels.csv")
    evidence = pl.read_csv(DATA / "development_evidence.csv")

    batch_files = sorted(glob.glob(str(OUTV2 / "hands_development" / "batch_*.parquet")))
    assert batch_files, "no hand-level batch files found"
    feat = pl.concat([pl.read_parquet(f) for f in batch_files])
    feature_stems = [c for c in feat.columns if c not in ("pair_id", "hand_id", "table_id")]
    assert len(feature_stems) == 163, len(feature_stems)

    hand_oof = pl.read_parquet(EVENT_DIR / "hand_oof.parquet")
    meta = hand_oof.select("pair_id", "hand_id", "table_id", "fold")

    df = feat.join(meta, on=["pair_id", "hand_id", "table_id"], how="inner")
    assert df.height == feat.height == hand_oof.height, (df.height, feat.height, hand_oof.height)
    df = df.join(labels.select("pair_id", "label", "behavior_family"), on="pair_id", how="left")
    assert df["label"].null_count() == 0

    # Sort exactly as step236 did (df.sort("pair_id") with no other prior transform on this frame's
    # row order) so within-pair hand order -- and therefore every RNG-index-based view -- reproduces
    # step236's BASE arm bit-for-bit for the OLD arm below.
    df = df.sort("pair_id")
    log(f"loaded+sorted {df.height} hand rows, {len(feature_stems)} old feature stems "
        f"in {time.perf_counter()-t0:.1f}s")

    # ---- aggregation check (reproduce a handful of outputs_v2/pairs_development.parquet columns)
    ref = pl.read_parquet(OUTV2 / "pairs_development.parquet")
    check_stems = ["contribution_bb_sum", "net_bb_sum", "stack_bb_sum", "folded_sum"]
    check_cols = [check_stems[0] + "_mean", check_stems[1] + "_std",
                  check_stems[2] + "_max", check_stems[3] + "_q95", "shared_hands"]
    agg_exprs = [pl.len().alias("shared_hands")]
    for s in check_stems:
        agg_exprs += [
            pl.col(s).mean().alias(s + "_mean"), pl.col(s).std(ddof=1).alias(s + "_std"),
            pl.col(s).max().alias(s + "_max"),
            pl.col(s).quantile(0.95, interpolation="nearest").alias(s + "_q95"),
        ]
    mine = df.group_by("pair_id").agg(agg_exprs)
    joined = mine.join(ref.select(["pair_id"] + check_cols), on="pair_id", suffix="_ref")
    agg_check = {}
    for c in check_cols:
        diff = (joined[c] - joined[c + "_ref"]).abs()
        agg_check[c] = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
    agg_check["n_pairs_compared"] = joined.height

    # ---- attach new 25-feature block via explicit (pair_id, hand_id) key lookup (not a bare join),
    # so the resulting row order is GUARANTEED identical to df's (already-sorted) row order.
    new_pl = pl.read_parquet(STEP237_DIR / "step237_features.parquet")
    new_cols_found = [c for c in new_pl.columns if c not in ("pair_id", "hand_id")]
    assert new_cols_found == NEW_FEATURES, (new_cols_found, NEW_FEATURES)
    new_pl = new_pl.with_columns((pl.col("pair_id") + pl.lit("|") + pl.col("hand_id")).alias("key"))
    key_to_row = dict(zip(new_pl["key"].to_list(), range(new_pl.height)))
    new_arr_raw = new_pl.select(NEW_FEATURES).to_numpy().astype(np.float32)
    assert not np.isnan(new_arr_raw).any()

    df = df.with_columns((pl.col("pair_id") + pl.lit("|") + pl.col("hand_id")).alias("key"))
    old_keys = df["key"].to_list()
    gather_idx = np.fromiter((key_to_row[k] for k in old_keys), dtype=np.int64, count=len(old_keys))
    assert len(np.unique(gather_idx)) == len(gather_idx) == df.height
    new_arr_aligned = new_arr_raw[gather_idx]
    log(f"attached new-feature block: {new_arr_aligned.shape}, aligned to old block's row order")

    # global permutation across ALL hand-rows (all pairs), fixed seed 247 -- the PERM control
    perm = np.random.default_rng(PERM_SEED).permutation(new_arr_aligned.shape[0])
    new_arr_perm = new_arr_aligned[perm]

    # ---- slice into per-pair blocks
    pair_id_col = df["pair_id"].to_numpy()
    pair_ids_sorted, start_idx = np.unique(pair_id_col, return_index=True)
    order = np.argsort(start_idx)
    pair_ids_sorted = pair_ids_sorted[order]
    start_idx = start_idx[order]
    ends = np.append(start_idx[1:], len(pair_id_col))
    n_pairs = len(pair_ids_sorted)

    X_old_all = df.select(feature_stems).to_numpy().astype(np.float32)
    table_all = df["table_id"].to_numpy()
    fold_all = df["fold"].to_numpy()

    label_map = dict(zip(labels["pair_id"].to_list(), labels["label"].to_list()))
    family_map = dict(zip(labels["pair_id"].to_list(), labels["behavior_family"].to_list()))
    label_arr = np.array([label_map[pid] for pid in pair_ids_sorted], dtype=np.int64)
    family_arr = np.array([family_map[pid] for pid in pair_ids_sorted], dtype=object)

    n_listed_map = dict(zip(
        evidence.group_by("pair_id").agg(pl.len().alias("n"))["pair_id"].to_list(),
        evidence.group_by("pair_id").agg(pl.len().alias("n"))["n"].to_list(),
    ))
    n_listed_arr = np.array([n_listed_map.get(pid, 0) for pid in pair_ids_sorted], dtype=np.int64)

    table_arr = np.array([table_all[start_idx[i]] for i in range(n_pairs)])
    fold_arr = np.array([fold_all[start_idx[i]] for i in range(n_pairs)], dtype=np.int64)
    for i in range(n_pairs):
        seg = fold_all[start_idx[i]:ends[i]]
        assert (seg == fold_arr[i]).all()

    X_old_list = [X_old_all[start_idx[i]:ends[i]] for i in range(n_pairs)]
    X_new_list = [new_arr_aligned[start_idx[i]:ends[i]] for i in range(n_pairs)]
    X_newperm_list = [new_arr_perm[start_idx[i]:ends[i]] for i in range(n_pairs)]
    del X_old_all, new_arr_aligned, new_arr_perm, table_all, fold_all
    gc.collect()

    log(f"n_pairs={n_pairs}, positives={int((label_arr==1).sum())}, negatives={int((label_arr==0).sum())}")

    # ---- build blocks: old (652), newreal (100), newperm (100), n (scalar) x {full, v70, v40, test65} x seeds
    blocks = {"old": {}, "newreal": {}, "newperm": {}, "n": {}}
    for k in blocks:
        blocks[k]["full"] = None
        for name in ["v70", "v40", "test65"]:
            blocks[k][name] = {s: None for s in SEEDS}

    old_full = np.zeros((n_pairs, 4 * 163), dtype=np.float32)
    newreal_full = np.zeros((n_pairs, 4 * 25), dtype=np.float32)
    newperm_full = np.zeros((n_pairs, 4 * 25), dtype=np.float32)
    n_full = np.zeros(n_pairs, dtype=np.float32)
    per_seed_arrs = {
        name: {s: {"old": np.zeros((n_pairs, 4 * 163), dtype=np.float32),
                    "newreal": np.zeros((n_pairs, 4 * 25), dtype=np.float32),
                    "newperm": np.zeros((n_pairs, 4 * 25), dtype=np.float32),
                    "n": np.zeros(n_pairs, dtype=np.float32)}
                for s in SEEDS}
        for name in ["v70", "v40", "test65"]
    }

    t0 = time.perf_counter()
    for i in range(n_pairs):
        Xo, Xn, Xp = X_old_list[i], X_new_list[i], X_newperm_list[i]
        n_i = Xo.shape[0]
        old_full[i] = agg_block(Xo)
        newreal_full[i] = agg_block(Xn)
        newperm_full[i] = agg_block(Xp)
        n_full[i] = float(n_i)

        for s in SEEDS:
            r70 = np.random.default_rng([s, i, CH_V70])
            k70 = max(1, int(round(0.70 * n_i)))
            idx70 = r70.choice(n_i, size=k70, replace=False)
            per_seed_arrs["v70"][s]["old"][i] = agg_block(Xo[idx70])
            per_seed_arrs["v70"][s]["newreal"][i] = agg_block(Xn[idx70])
            per_seed_arrs["v70"][s]["newperm"][i] = agg_block(Xp[idx70])
            per_seed_arrs["v70"][s]["n"][i] = float(k70)

            r40 = np.random.default_rng([s, i, CH_V40])
            k40 = max(1, int(round(0.40 * n_i)))
            idx40 = r40.choice(n_i, size=k40, replace=False)
            per_seed_arrs["v40"][s]["old"][i] = agg_block(Xo[idx40])
            per_seed_arrs["v40"][s]["newreal"][i] = agg_block(Xn[idx40])
            per_seed_arrs["v40"][s]["newperm"][i] = agg_block(Xp[idx40])
            per_seed_arrs["v40"][s]["n"][i] = float(k40)

            rT = np.random.default_rng([s, i, CH_TEST65])
            kT = max(1, int(round(0.65 * n_i)))
            idxT = rT.choice(n_i, size=kT, replace=False)
            per_seed_arrs["test65"][s]["old"][i] = agg_block(Xo[idxT])
            per_seed_arrs["test65"][s]["newreal"][i] = agg_block(Xn[idxT])
            per_seed_arrs["test65"][s]["newperm"][i] = agg_block(Xp[idxT])
            per_seed_arrs["test65"][s]["n"][i] = float(kT)

        if (i + 1) % 400 == 0:
            log(f"  view-building {i+1}/{n_pairs} pairs ({time.perf_counter()-t0:.1f}s elapsed)")
    log(f"view-building done in {time.perf_counter()-t0:.1f}s")

    save_dict = {
        "old_full": old_full, "newreal_full": newreal_full, "newperm_full": newperm_full, "n_full": n_full,
    }
    for name in ["v70", "v40", "test65"]:
        for s in SEEDS:
            save_dict[f"old_{name}_{s}"] = per_seed_arrs[name][s]["old"]
            save_dict[f"newreal_{name}_{s}"] = per_seed_arrs[name][s]["newreal"]
            save_dict[f"newperm_{name}_{s}"] = per_seed_arrs[name][s]["newperm"]
            save_dict[f"n_{name}_{s}"] = per_seed_arrs[name][s]["n"]
    np.savez(CACHE_DIR / "blocks.npz", **save_dict)
    log(f"wrote {CACHE_DIR / 'blocks.npz'}")

    lowdose_mask = (label_arr == 1) & np.isin(n_listed_arr, [3, 4])
    meta_out = {
        "pair_ids": pair_ids_sorted.tolist(),
        "label": label_arr.tolist(),
        "family": family_arr.tolist(),
        "n_listed": n_listed_arr.tolist(),
        "table": table_arr.tolist(),
        "fold": fold_arr.tolist(),
        "lowdose_idx": np.where(lowdose_mask)[0].tolist(),
        "feature_stems": feature_stems,
        "new_features": NEW_FEATURES,
        "outsider_card_features": OUTSIDER_CARD_FEATURES,
        "n_pairs": n_pairs,
        "aggregation_check": agg_check,
        "script_sha256": sha256_of(SCRIPT_PATH),
    }
    write_json(CACHE_DIR / "store_meta.json", meta_out)
    log(f"wrote {CACHE_DIR / 'store_meta.json'}; low-dose group n={int(lowdose_mask.sum())}")
    log(f"aggregation check: { {k: v['max_abs_diff'] for k, v in agg_check.items() if isinstance(v, dict)} }")
    log(f"cache stage total {time.perf_counter()-t0:.1f}s (note: t0 reset above; see per-section timings)")


# ---------------------------------------------------------------------------
# Stage: arm -- cross-fit one arm (3 seeds x 5 folds), checkpointed per seed
# ---------------------------------------------------------------------------
def load_cache():
    meta = read_json(CACHE_DIR / "store_meta.json")
    blocks = np.load(CACHE_DIR / "blocks.npz")
    return meta, blocks


def get_block(blocks, kind, view, seed=None):
    if view == "full":
        return blocks[f"{kind}_full"]
    return blocks[f"{kind}_{view}_{seed}"]


def stage_arm(arm):
    assert arm in ARMS, arm
    meta, blocks = load_cache()
    n_pairs = meta["n_pairs"]
    label_arr = np.array(meta["label"], dtype=np.int64)
    fold_arr = np.array(meta["fold"], dtype=np.int64)
    feature_stems = meta["feature_stems"]

    for s in SEEDS:
        outpath = CACHE_DIR / f"oof_{arm}_{s}.npz"
        if outpath.exists():
            log(f"arm={arm} seed={s}: cache hit, skipping")
            continue
        t0 = time.perf_counter()
        old_full = get_block(blocks, "old", "full")
        newreal_full = get_block(blocks, "newreal", "full")
        newperm_full = get_block(blocks, "newperm", "full")
        n_full = get_block(blocks, "n", "full")
        Xfull = assemble(arm, old_full, newreal_full, newperm_full, n_full)

        old_v70 = get_block(blocks, "old", "v70", s)
        newreal_v70 = get_block(blocks, "newreal", "v70", s)
        newperm_v70 = get_block(blocks, "newperm", "v70", s)
        n_v70 = get_block(blocks, "n", "v70", s)
        Xv70 = assemble(arm, old_v70, newreal_v70, newperm_v70, n_v70)

        old_v40 = get_block(blocks, "old", "v40", s)
        newreal_v40 = get_block(blocks, "newreal", "v40", s)
        newperm_v40 = get_block(blocks, "newperm", "v40", s)
        n_v40 = get_block(blocks, "n", "v40", s)
        Xv40 = assemble(arm, old_v40, newreal_v40, newperm_v40, n_v40)

        old_t65 = get_block(blocks, "old", "test65", s)
        newreal_t65 = get_block(blocks, "newreal", "test65", s)
        newperm_t65 = get_block(blocks, "newperm", "test65", s)
        n_t65 = get_block(blocks, "n", "test65", s)
        Xtest65 = assemble(arm, old_t65, newreal_t65, newperm_t65, n_t65)

        Xpool = np.concatenate([Xfull, Xv70, Xv40], axis=0)
        ypool = np.tile(label_arr, 3)
        wpool = np.full(Xpool.shape[0], 1.0 / 3, dtype=np.float64)
        foldpool = np.tile(fold_arr, 3)

        oof_T1 = np.zeros(n_pairs)
        oof_T2 = np.zeros(n_pairs)
        importance_sum = np.zeros(Xpool.shape[1], dtype=np.float64) if arm == "NEW25" else None

        for f in range(5):
            train_mask = foldpool != f
            model = LGBMClassifier(**LGB_PARAMS)
            model.fit(Xpool[train_mask], ypool[train_mask], sample_weight=wpool[train_mask])
            test_pairs = np.where(fold_arr == f)[0]
            oof_T1[test_pairs] = model.predict_proba(Xfull[test_pairs])[:, 1]
            oof_T2[test_pairs] = model.predict_proba(Xtest65[test_pairs])[:, 1]
            if importance_sum is not None:
                importance_sum += model.feature_importances_.astype(np.float64)

        np.savez(outpath, T1=oof_T1, T2=oof_T2,
                 importance=(importance_sum if importance_sum is not None else np.zeros(0)))
        log(f"arm={arm} seed={s}: {Xpool.shape[0]} rows, dim={Xpool.shape[1]}, "
            f"{time.perf_counter()-t0:.1f}s for 5 folds -> {outpath.name}")

    if arm == "NEW25":
        names = feature_names_for("NEW25", feature_stems)
        write_json(CACHE_DIR / "feature_names_NEW25.json", names)


# ---------------------------------------------------------------------------
# Stage: report -- metrics, bootstrap, gain importance, REPORT.txt
# ---------------------------------------------------------------------------
def percentile_of_negatives(scores, neg_scores_sorted):
    m = len(neg_scores_sorted)
    cnt = np.searchsorted(neg_scores_sorted, scores, side="right")
    return 100.0 * cnt / m


def family_ap(scores, label_arr, family_arr, fam):
    mask = (label_arr == 0) | (family_arr == fam)
    y = (label_arr[mask] == 1) & (family_arr[mask] == fam)
    return float(average_precision_score(y.astype(int), scores[mask]))


def group_stats(scores, label_arr, family_arr, n_listed_arr):
    neg_mask = label_arr == 0
    neg_scores = scores[neg_mask]
    neg_sorted = np.sort(neg_scores)
    thr995 = float(np.quantile(neg_scores, 0.995))
    thr999 = float(np.quantile(neg_scores, 0.999))
    thr_top1 = float(np.quantile(neg_scores, TOP1_Q))

    pos_idx = np.where(label_arr == 1)[0]
    pct_all = percentile_of_negatives(scores[pos_idx], neg_sorted)
    pct_map = dict(zip(pos_idx.tolist(), pct_all.tolist()))

    n_missed = int(np.sum(scores[pos_idx] < thr_top1))

    groups = {}
    defs = {"listed_3": n_listed_arr == 3, "listed_4": n_listed_arr == 4, "listed_5": n_listed_arr == 5}
    for fam in FAMILIES:
        defs[f"family_{fam}"] = family_arr == fam
    for key, mask in defs.items():
        idxs = pos_idx[mask[pos_idx]]
        if len(idxs) == 0:
            groups[key] = {"n": 0}
            continue
        pcts = np.array([pct_map[i] for i in idxs])
        sc = scores[idxs]
        groups[key] = {
            "n": int(len(idxs)), "mean_pct": float(pcts.mean()), "median_pct": float(np.median(pcts)),
            "share_ge_995": float(np.mean(sc >= thr995)), "share_ge_999": float(np.mean(sc >= thr999)),
        }
    all_pcts = np.array([pct_map[i] for i in pos_idx])
    groups["all_positives"] = {
        "n": int(len(pos_idx)), "mean_pct": float(all_pcts.mean()), "median_pct": float(np.median(all_pcts)),
        "share_ge_995": float(np.mean(scores[pos_idx] >= thr995)),
        "share_ge_999": float(np.mean(scores[pos_idx] >= thr999)),
    }
    fam_ap = {fam: family_ap(scores, label_arr, family_arr, fam) for fam in FAMILIES}
    return {
        "overall_ap": float(average_precision_score(label_arr, scores)), "family_ap": fam_ap,
        "groups": groups, "neg_thr995": thr995, "neg_thr999": thr999, "neg_thr_top1": thr_top1,
        "n_missed": n_missed, "pct_map": pct_map,
    }


def build_bootstrap_weights(table_arr, boot_seed):
    tables_sorted = np.unique(table_arr)
    n_tables = len(tables_sorted)
    table_to_idx = {t: k for k, t in enumerate(tables_sorted)}
    table_idx_of_pair = np.array([table_to_idx[t] for t in table_arr], dtype=np.int64)

    rng = np.random.default_rng(boot_seed)
    counts_matrix = np.zeros((N_BOOT, n_tables), dtype=np.int32)
    for d in range(N_BOOT):
        draw = rng.integers(0, n_tables, size=n_tables)
        counts_matrix[d] = np.bincount(draw, minlength=n_tables)
    W = counts_matrix[:, table_idx_of_pair].astype(np.float64)
    return W, n_tables


def bootstrap_ap_masked(scores, label01, W):
    order = np.argsort(-scores)
    Ws = W[:, order]
    ys = label01[order].astype(np.float64)
    tp_cum = np.cumsum(Ws * ys[None, :], axis=1)
    n_cum = np.cumsum(Ws, axis=1)
    precision = np.divide(tp_cum, np.maximum(n_cum, 1e-12))
    total_pos_w = np.maximum(tp_cum[:, -1], 1e-12)
    delta_recall = (Ws * ys[None, :]) / total_pos_w[:, None]
    return np.sum(precision * delta_recall, axis=1)


def bootstrap_ap(scores, label_arr, W):
    return bootstrap_ap_masked(scores, (label_arr == 1).astype(np.float64), W)


def bootstrap_family_ap(scores, label_arr, family_arr, fam, W):
    mask = (label_arr == 0) | (family_arr == fam)
    idx = np.where(mask)[0]
    y = ((label_arr[idx] == 1) & (family_arr[idx] == fam)).astype(np.float64)
    return bootstrap_ap_masked(scores[idx], y, W[:, idx])


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
    return np.sum(pct_matrix * Wj, axis=1) / denom


def bootstrap_missed(scores, label_arr, W, q=TOP1_Q):
    neg_idx = np.where(label_arr == 0)[0]
    pos_idx = np.where(label_arr == 1)[0]
    order_neg = np.argsort(scores[neg_idx])
    sorted_neg_scores = scores[neg_idx][order_neg]
    W_neg = W[:, neg_idx][:, order_neg]
    cumw = np.cumsum(W_neg, axis=1)
    totalw = np.maximum(cumw[:, -1], 1e-12)
    target = q * totalw
    B = W.shape[0]
    thresh = np.empty(B)
    for b in range(B):
        pos = np.searchsorted(cumw[b], target[b])
        pos = min(pos, len(sorted_neg_scores) - 1)
        thresh[b] = sorted_neg_scores[pos]
    Wpos = W[:, pos_idx]
    below = (scores[pos_idx][None, :] < thresh[:, None])
    return np.sum(Wpos * below, axis=1)


def ci(arr, lo=2.5, hi=97.5):
    return [float(np.percentile(arr, lo)), float(np.percentile(arr, hi))]


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)) else str(x)


def stage_report():
    meta, _ = load_cache()
    n_pairs = meta["n_pairs"]
    label_arr = np.array(meta["label"], dtype=np.int64)
    family_arr = np.array(meta["family"], dtype=object)
    n_listed_arr = np.array(meta["n_listed"], dtype=np.int64)
    table_arr = np.array(meta["table"], dtype=object)
    lowdose_idx = np.array(meta["lowdose_idx"], dtype=np.int64)
    feature_stems = meta["feature_stems"]

    missing = [f"{a}/{s}" for a in ARMS for s in SEEDS if not (CACHE_DIR / f"oof_{a}_{s}.npz").exists()]
    if missing:
        raise SystemExit(f"missing oof cache files, run `arm` stage first for: {missing}")

    oof = {a: {s: {} for s in SEEDS} for a in ARMS}
    importance_new25 = None
    for a in ARMS:
        for s in SEEDS:
            d = np.load(CACHE_DIR / f"oof_{a}_{s}.npz")
            oof[a][s]["T1"] = d["T1"]
            oof[a][s]["T2"] = d["T2"]
            if a == "NEW25":
                if importance_new25 is None:
                    importance_new25 = d["importance"].copy()
                else:
                    importance_new25 += d["importance"]

    log("computing main metrics table")
    metrics = {a: {s: {tv: group_stats(oof[a][s][tv], label_arr, family_arr, n_listed_arr)
                        for tv in TESTVIEWS} for s in SEEDS} for a in ARMS}

    seed_summary = {}
    for a in ARMS:
        seed_summary[a] = {}
        for tv in TESTVIEWS:
            aps = np.array([metrics[a][s][tv]["overall_ap"] for s in SEEDS])
            entry = {"ap_mean": float(aps.mean()), "ap_std": float(aps.std(ddof=1)),
                     "ap_min": float(aps.min()), "ap_max": float(aps.max())}
            for fam in FAMILIES:
                fam_aps = np.array([metrics[a][s][tv]["family_ap"][fam] for s in SEEDS])
                entry[f"famap_{fam}_mean"] = float(fam_aps.mean())
                entry[f"famap_{fam}_spread"] = float(fam_aps.max() - fam_aps.min())
            for gkey in ["listed_3", "listed_4", "listed_5", "family_directed_transfer",
                         "family_soft_play", "family_coordinated_isolation", "all_positives"]:
                for m in ["mean_pct", "median_pct", "share_ge_995", "share_ge_999"]:
                    vals = np.array([metrics[a][s][tv]["groups"][gkey].get(m, np.nan) for s in SEEDS])
                    entry[f"{gkey}_{m}_mean"] = float(np.nanmean(vals))
                    entry[f"{gkey}_{m}_spread"] = float(np.nanmax(vals) - np.nanmin(vals))
            missed = np.array([metrics[a][s][tv]["n_missed"] for s in SEEDS])
            entry["n_missed_mean"] = float(missed.mean())
            entry["n_missed_spread"] = float(missed.max() - missed.min())
            seed_summary[a][tv] = entry

    log("building bootstrap-over-tables weights (seed 247)")
    W, n_tables = build_bootstrap_weights(table_arr, BOOT_SEED)
    log(f"n_tables={n_tables}, bootstrap draws={N_BOOT}")

    comparisons = [("NEW25", "OLD"), ("NEW19", "OLD"), ("ISO1", "OLD"), ("NEW25", "PERM")]
    paired = {}
    for a, b in comparisons:
        key = f"{a}-{b}"
        paired[key] = {}
        for s in SEEDS:
            paired[key][s] = {}
            for tv in TESTVIEWS:
                sa, sb = oof[a][s][tv], oof[b][s][tv]
                entry = {}
                ap_a, ap_b = metrics[a][s][tv]["overall_ap"], metrics[b][s][tv]["overall_ap"]
                boot_a, boot_b = bootstrap_ap(sa, label_arr, W), bootstrap_ap(sb, label_arr, W)
                diff = boot_a - boot_b
                entry["ap"] = {"point": float(ap_a - ap_b), "ci95": ci(diff), "boot_mean": float(diff.mean())}

                for fam in FAMILIES:
                    fap_a = metrics[a][s][tv]["family_ap"][fam]
                    fap_b = metrics[b][s][tv]["family_ap"][fam]
                    fb_a = bootstrap_family_ap(sa, label_arr, family_arr, fam, W)
                    fb_b = bootstrap_family_ap(sb, label_arr, family_arr, fam, W)
                    fd = fb_a - fb_b
                    entry[f"famap_{fam}"] = {"point": float(fap_a - fap_b), "ci95": ci(fd),
                                              "boot_mean": float(fd.mean())}

                pct_a = float(np.mean([metrics[a][s][tv]["pct_map"][i] for i in lowdose_idx]))
                pct_b = float(np.mean([metrics[b][s][tv]["pct_map"][i] for i in lowdose_idx]))
                pb_a = bootstrap_lowdose_pct(sa, label_arr, lowdose_idx, W)
                pb_b = bootstrap_lowdose_pct(sb, label_arr, lowdose_idx, W)
                pd = pb_a - pb_b
                entry["lowdose_pct"] = {"point": float(pct_a - pct_b), "ci95": ci(pd), "boot_mean": float(pd.mean())}

                miss_a_pt, miss_b_pt = metrics[a][s][tv]["n_missed"], metrics[b][s][tv]["n_missed"]
                mb_a, mb_b = bootstrap_missed(sa, label_arr, W), bootstrap_missed(sb, label_arr, W)
                md = mb_a - mb_b
                entry["n_missed"] = {"point": float(miss_a_pt - miss_b_pt), "ci95": ci(md), "boot_mean": float(md.mean())}

                paired[key][s][tv] = entry
        for tv in TESTVIEWS:
            ss = {}
            for metric_key in ["ap", "famap_directed_transfer", "famap_soft_play",
                                "famap_coordinated_isolation", "lowdose_pct", "n_missed"]:
                pts = np.array([paired[key][s][tv][metric_key]["point"] for s in SEEDS])
                ss[metric_key] = {"point_mean": float(pts.mean()), "point_spread": float(pts.max() - pts.min())}
            paired[key][f"seed_summary_{tv}"] = ss
        log(f"  paired diff {key} done")

    # ---- gain importance for NEW25
    names_new25 = feature_names_for("NEW25", feature_stems)
    assert len(names_new25) == len(importance_new25), (len(names_new25), len(importance_new25))
    total_gain = float(importance_new25.sum())
    n_old = 4 * len(feature_stems)
    new_block_gain = float(importance_new25[n_old:n_old + 100].sum())
    new_block_gain_share = new_block_gain / total_gain if total_gain > 0 else float("nan")
    order = np.argsort(-importance_new25)
    top15 = [(names_new25[i], float(importance_new25[i])) for i in order[:15]]

    results = {
        "step": 247,
        "description": "pair-level aggregates of the step237 25 new per-hand features (development phase only)",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_sha256": meta["script_sha256"],
        "params": {"lgb_params": LGB_PARAMS, "seeds": SEEDS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                   "perm_seed": PERM_SEED, "top1_q": TOP1_Q},
        "dataset": {"n_pairs": n_pairs, "n_positive": int((label_arr == 1).sum()),
                    "n_negative": int((label_arr == 0).sum()), "n_tables": n_tables,
                    "n_low_dose_pairs": int(len(lowdose_idx))},
        "aggregation_check": meta["aggregation_check"],
        "outsider_card_features_dropped_for_new19": OUTSIDER_CARD_FEATURES,
        "metrics": {a: {str(s): {tv: {
            "overall_ap": metrics[a][s][tv]["overall_ap"], "family_ap": metrics[a][s][tv]["family_ap"],
            "n_missed": metrics[a][s][tv]["n_missed"], "neg_thr995": metrics[a][s][tv]["neg_thr995"],
            "neg_thr999": metrics[a][s][tv]["neg_thr999"], "neg_thr_top1": metrics[a][s][tv]["neg_thr_top1"],
            "groups": metrics[a][s][tv]["groups"],
        } for tv in TESTVIEWS} for s in SEEDS} for a in ARMS},
        "seed_summary": seed_summary,
        "paired_differences": {k: {(str(kk) if isinstance(kk, int) else kk): vv for kk, vv in v.items()}
                                for k, v in paired.items()},
        "new25_gain_importance": {"total_gain": total_gain, "new_block_gain": new_block_gain,
                                   "new_block_gain_share": new_block_gain_share, "top15": top15},
    }
    write_json(OUT_DIR / "results247.json", results)
    log(f"wrote {OUT_DIR / 'results247.json'}")

    write_report_txt(results)
    log(f"wrote {OUT_DIR / 'REPORT.txt'}")
    print_verdict(results)


def write_report_txt(results):
    lines = []
    lines.append("STEP 247: Pair-level aggregates of the step237 25 new per-hand features")
    lines.append("Development phase only, self-contained proxy study. Not uploaded.")
    lines.append("")

    lines.append("== Aggregation / reproduction check (OLD arm vs step236 BASE arm) ==")
    ac = results["aggregation_check"]
    for c, v in ac.items():
        if isinstance(v, dict) and "max_abs_diff" in v:
            lines.append(f"  {c}: max_abs_diff={v['max_abs_diff']:.6g} mean_abs_diff={v['mean_abs_diff']:.6g}")
    old_t1 = [results["metrics"]["OLD"][str(s)]["T1"]["overall_ap"] for s in SEEDS]
    old_t2 = [results["metrics"]["OLD"][str(s)]["T2"]["overall_ap"] for s in SEEDS]
    lines.append(f"  OLD arm AP by seed -- T1: {[round(x,6) for x in old_t1]}")
    lines.append(f"  OLD arm AP by seed -- T2: {[round(x,6) for x in old_t2]}")
    lines.append("  step236 BASE arm reference (seed=236): T1=0.989498, T2=0.978088 (results236.json)")
    match_t1 = abs(old_t1[0] - 0.9894981874986262) < 1e-4
    match_t2 = abs(old_t2[0] - 0.9780876234418872) < 1e-4
    lines.append(f"  match to 4 decimals -- T1: {match_t1}, T2: {match_t2}")
    lines.append("")

    ds = results["dataset"]
    lines.append("== Design summary ==")
    lines.append(f"  {ds['n_pairs']} labelled pairs ({ds['n_positive']} positive, {ds['n_negative']} negative), "
                 f"{ds['n_tables']} tables, 5-fold CV by table.")
    lines.append(f"  Low-dose group (listed count 3 or 4): n={ds['n_low_dose_pairs']}.")
    lines.append("  Arms: OLD (163 old features, 653 dims) = step236 BASE; NEW25 (+100 dims: mean/std/max/q95 "
                 "of all 25 new features); ISO1 (+4 dims: iso_weak_agg_behind only); NEW19 (+76 dims: 19 "
                 "features excluding the six that read an outsider's hole cards); PERM (NEW25 with the 25 "
                 "new per-hand feature values permuted across ALL pairs at the hand level, seed 247).")
    lines.append(f"  Dropped for NEW19 (outsider hole-card features): {results['outsider_card_features_dropped_for_new19']}")
    lines.append("  Training views: full history + random 70% + random 40% (weight 1/3 each). Test views: "
                 "T1 = full history, T2 = random 65% thinning. 3 view-sampling seeds (236, 237, 238); "
                 "LightGBM random_state fixed at 236; n_jobs=1.")
    lines.append("")

    lines.append("== Metrics (mean across 3 seeds) ==")
    for tv in TESTVIEWS:
        lines.append(f"-- Test view {tv} --")
        header = ("arm | AP (mean, std) | AP dir_transfer | AP soft_play | AP coord_iso | "
                  "listed3 pct | listed4 pct | listed5 pct | n_missed (of top-1%)")
        lines.append(header)
        for a in ARMS:
            e = results["seed_summary"][a][tv]
            lines.append(
                f"  {a:6s} | {fmt(e['ap_mean'])} ({fmt(e['ap_std'],4)}) | "
                f"{fmt(e['famap_directed_transfer_mean'])} | {fmt(e['famap_soft_play_mean'])} | "
                f"{fmt(e['famap_coordinated_isolation_mean'])} | "
                f"{fmt(e['listed_3_mean_pct_mean'],1)} | {fmt(e['listed_4_mean_pct_mean'],1)} | "
                f"{fmt(e['listed_5_mean_pct_mean'],1)} | {fmt(e['n_missed_mean'],1)}"
            )
        lines.append("")
        lines.append("  by-family mean percentile / share>=99.5 / share>=99.9 among negatives:")
        for a in ARMS:
            e = results["seed_summary"][a][tv]
            parts = []
            for fam in FAMILIES:
                gk = f"family_{fam}"
                parts.append(f"{fam}: pct={fmt(e[gk+'_mean_pct_mean'],1)} "
                              f"ge995={fmt(e[gk+'_share_ge_995_mean'],3)} ge999={fmt(e[gk+'_share_ge_999_mean'],3)}")
            lines.append(f"  {a:6s} | " + " || ".join(parts))
        lines.append("")

    lines.append("== Paired differences (bootstrap over tables, 2000 resamples, seed 247) ==")
    lines.append("Point = actual arm-pair difference; [CI] = 2.5/97.5 pct of the bootstrap distribution of "
                 "the same paired difference; mean(3 seeds) also shown with the seed-to-seed spread.")
    for key in ["NEW25-OLD", "NEW19-OLD", "ISO1-OLD", "NEW25-PERM"]:
        lines.append(f"-- {key} --")
        for tv in TESTVIEWS:
            lines.append(f"  test view {tv}:")
            for s in SEEDS:
                p = results["paired_differences"][key][str(s)][tv]
                lines.append(
                    f"    seed {s}: AP diff {fmt(p['ap']['point'])} {p['ap']['ci95']} | "
                    f"DT famAP {fmt(p['famap_directed_transfer']['point'])} {p['famap_directed_transfer']['ci95']} | "
                    f"SP famAP {fmt(p['famap_soft_play']['point'])} {p['famap_soft_play']['ci95']} | "
                    f"CI famAP {fmt(p['famap_coordinated_isolation']['point'])} {p['famap_coordinated_isolation']['ci95']} | "
                    f"lowdose pct diff {fmt(p['lowdose_pct']['point'],2)} {[round(x,2) for x in p['lowdose_pct']['ci95']]} | "
                    f"n_missed diff {fmt(p['n_missed']['point'],2)} {[round(x,2) for x in p['n_missed']['ci95']]}"
                )
            ss = results["paired_differences"][key][f"seed_summary_{tv}"]
            lines.append(f"    mean(3 seeds): AP {fmt(ss['ap']['point_mean'])} (spread {fmt(ss['ap']['point_spread'])}) | "
                         f"lowdose pct {fmt(ss['lowdose_pct']['point_mean'],2)} (spread {fmt(ss['lowdose_pct']['point_spread'],2)}) | "
                         f"n_missed {fmt(ss['n_missed']['point_mean'],2)} (spread {fmt(ss['n_missed']['point_spread'],2)})")
        lines.append("")

    lines.append("== NEW25 LightGBM gain importance ==")
    gi = results["new25_gain_importance"]
    lines.append(f"  new-block (100 cols) gain share: {fmt(gi['new_block_gain_share'],4)} "
                 f"({fmt(gi['new_block_gain'],1)} / {fmt(gi['total_gain'],1)} total gain, summed over 3 seeds x 5 folds)")
    lines.append("  top 15 columns by gain:")
    for name, val in gi["top15"]:
        lines.append(f"    {name}: {val:.1f}")
    lines.append("")

    lines.append("== Verdict ==")
    lines.append(verdict_text(results))

    Path(OUT_DIR / "REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


def verdict_text(results):
    lines = []
    key = "NEW25-PERM"
    t2 = results["paired_differences"][key]["seed_summary_T2"]
    t1 = results["paired_differences"][key]["seed_summary_T1"]
    ap25old_t2 = results["paired_differences"]["NEW25-OLD"]["seed_summary_T2"]["ap"]
    ap19old_t2 = results["paired_differences"]["NEW19-OLD"]["seed_summary_T2"]["ap"]
    apiso_t2 = results["paired_differences"]["ISO1-OLD"]["seed_summary_T2"]["ap"]

    fam_lines = []
    for fam, k in [("directed_transfer", "famap_directed_transfer"),
                   ("soft_play", "famap_soft_play"), ("coordinated_isolation", "famap_coordinated_isolation")]:
        pt = t2[k]["point_mean"]
        fam_lines.append(f"{fam} {fmt(pt,4)}")

    lowdose_pt = t2["lowdose_pct"]["point_mean"]
    missed_pt = t2["n_missed"]["point_mean"]

    lines.append(
        f"NEW25-PERM at T2 (evaluation-like 65% thinning): overall AP diff {fmt(t2['ap']['point_mean'],4)} "
        f"(mean of 3 seeds); by family: {', '.join(fam_lines)}; low-dose group mean-percentile diff "
        f"{fmt(lowdose_pt,2)} pts; missed-positives diff {fmt(missed_pt,2)} (all vs the shuffled-feature control, "
        f"so these numbers are on top of whatever an extra 100 columns of noise would do)."
    )
    lines.append(
        f"For context at T2: NEW25-OLD AP diff {fmt(ap25old_t2['point_mean'],4)}, "
        f"NEW19-OLD AP diff {fmt(ap19old_t2['point_mean'],4)}, ISO1-OLD AP diff {fmt(apiso_t2['point_mean'],4)} "
        f"(T1 NEW25-PERM AP diff {fmt(t1['ap']['point_mean'],4)} for the full-history view)."
    )

    go = t2["ap"]["point_mean"] > 0 and lowdose_pt > 0
    verdict_word = "GO" if go else ("MIXED" if t2["ap"]["point_mean"] > 0 or lowdose_pt > 0 else "NO-GO")
    lines.append(f"VERDICT: {verdict_word} (see REPORT.txt bootstrap CIs for the per-comparison, per-family "
                 "and per-seed detail before treating this as conclusive).")
    return "\n".join(lines)


def print_verdict(results):
    print("\n" + "=" * 70)
    print(verdict_text(results))
    print("=" * 70)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    stage = sys.argv[1]
    if stage == "cache":
        stage_cache()
    elif stage == "arm":
        arm = sys.argv[2]
        stage_arm(arm)
    elif stage == "report":
        stage_report()
    elif stage == "all":
        stage_cache()
        for a in ARMS:
            stage_arm(a)
        stage_report()
    else:
        raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main()
