"""STEP268 C -- the PAIR-LEVEL read: does a pooled Qwen3-0.6B embedding help the pair model?

The STEP247 / STEP254-D2 / STEP261 pair-proxy harness, reproduced here WITHOUT touching
src/evidence_aug/step254_D2_pair.py or src/pu_pairs/step261_pair_feature_expansion_proxy.py:
the same 1,860 labelled development pairs in the step247 order, the same per-pair view RNG
channels (seeds 236/237/238, CH_V70 = 10, CH_V40 = 11, CH_TEST65 = 90), the same pooled training
set (full + 70 % + 40 % views, weight 1/3 each), the same LightGBM 226/15/.03, 5 folds by table,
T1 = full test view, T2 = 65 % thinned test view, halves by step5_evaluate.half, and the step247
low-dose group (the 32 positives with 3 or 4 listed hands).

  OLD     the cached 653-column step247 OLD block
  M6      OLD + the six pooled MECHPLUS aggregates            <- THE BAR, T2 AP must be .98459
  NOISE70 M6 + 70 Gaussian columns (the matched dilution reference for EMB)
  EMB     M6 + gpu/fit_heads.py's per-hand H1-OOF probability and PCA-32 coordinates, pooled into
          the same six D2 aggregates + 32 PCA mean + 32 PCA max = 70 columns, joined uid ->
          (pair_id, hand_id) through experiments/step268_.../data/keys_universe.parquet

TWO modes for the EMB (and NOISE70) block:

  --emb-per-hand (THE DOCUMENTED MAIN PATH, once heads/h1_oof.npz and heads/emb_pca32.npz come
  back).  heads/fit_heads.py's H1 p_oof and H2 PCA-32 are saved PER HAND (per uid), so this
  script re-pools them itself on EVERY view the harness builds (full, 70 %, 40 %, T1 = full test
  view, T2 = 65 % thinned test view), using the harness's OWN per-pair RNG channel draws so a
  hand survives a view identically to the six MECHPLUS aggregates.  Only in_pool hands (the
  pair's fixed top-40, ranked on the box) carry a p_oof / PCA-32 at all, so the pool a thinned
  view draws from is the top-40 of the pair's FULL history filtered down by that view: a hand
  outside the original top-40 can never enter a thinned view's pool even where the harness's RNG
  draw would have kept it.  This is slightly PESSIMISTIC for EMB (the opposite bias from below).
  --noise-control redraws NOISE70's 70 Gaussian columns per view in this mode too, matching
  STEP261's own NOISE arm rather than the single per-pair draw below.

  --emb (kept for the wire test only; NOT the documented path).  gpu/fit_heads.py's
  heads/pair_emb_features.parquet is ONE constant 70-column block per pair, computed on the box
  over the pair's fixed top-40 pool, and CANNOT be re-pooled on a thinned view.  T2 therefore
  thins the OLD block and the six MECHPLUS aggregates while this EMB block keeps full-history
  information: the comparison is BIASED IN FAVOUR OF EMB at T2, which is why --noise-control
  draws NOISE70 once per pair here too - the matched dilution reference under exactly the same
  (constant-across-views) treatment.

Registered pass rule (registration268.json, "risk", the STEP261 rule):
  M6+EMB - M6 >= +.003 AP at T2, both table halves >= 0, and low-dose AP not below M6.

Usage
  python step268_pair_emb_read.py --baseline                              (reproduce M6 only)
  python step268_pair_emb_read.py --baseline --noise-control              (+ dilution reference)
  python step268_pair_emb_read.py --baseline --emb-per-hand               (main path, once heads/
                                                                            h1_oof.npz + emb_pca32
                                                                            .npz come back)
  python step268_pair_emb_read.py --baseline --emb-per-hand --noise-control
  python step268_pair_emb_read.py --emb                                   (old constant-block path)
Development phase only.  Builds no submission.  Writes experiments/step268_.../eval/pair_read.json.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import argparse
import glob
import json
import sys
import time

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

ROOT = r'C:\Data\Projects\Money Making\work\poker'
CV = ROOT + r'\Claude Version'
if CV + r'\src' not in sys.path:
    sys.path.insert(0, CV + r'\src')

EXP = CV + r'\experiments\step268_qwen06b_encoder_heads_20260918'
EVAL = EXP + r'\eval'
KEYS_UNIVERSE = EXP + r'\data\keys_universe.parquet'
# fit_heads.py writes the .csv unconditionally and the .parquet only when the box happens to
# have pyarrow; the parquet is preferred when both are there.
PAIR_EMB_CANDIDATES = (EXP + r'\heads\pair_emb_features.parquet',
                       EXP + r'\heads\pair_emb_features.csv')
S247 = CV + r'\experiments\step247_pair_new_block_20260917\cache'
S254 = CV + r'\experiments\step254_mechanical_tensor_20260917'
S261 = CV + r'\experiments\step261_pair_feature_expansion_20260918'
EVENT = CV + r'\experiments\step232_event_detector_20260917\hand_oof.parquet'

SEEDS = [236, 237, 238]
CH = {'v70': (10, .70), 'v40': (11, .40), 'test65': (90, .65)}
VIEW_KEYS = ['full'] + [f'{nm}_{s}' for nm in CH for s in SEEDS]      # the 10 views the harness builds
NJ = int(os.environ.get('STEP268_NJOBS', '3'))
LGB = dict(n_estimators=226, num_leaves=15, learning_rate=0.03, reg_lambda=5,
           min_child_samples=20, n_jobs=NJ, random_state=236, verbose=-1)
M6_T2_AP_REGISTERED = 0.98459          # results254_D2.json / results261_arms.json
REPRO_TOL = 1e-4
NOISE_COLS = 70
NOISE_SEED = 268
RULE = ('M6+EMB - M6 >= +.003 AP at T2 with both table halves >= 0 and low-dose AP not below M6')


# --------------------------------------------------------------------------- the D2 recipe
def pair_aggs6(p, q99, q999):
    """step254_D2_pair.pair_aggs / step261.pair_aggs6, character for character."""
    c = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(c / (1 - c))
    s = np.sort(lg)[::-1]
    return np.array([float((p > q999).sum()), float((p > q99).sum()), float(s[:1].mean()),
                     float(s[:3].mean()), float(s[:5].mean()),
                     float(-np.log1p(-np.clip(p, 0, 1 - 1e-9)).sum())], dtype=np.float32)


def hand_frame():
    """The step247 hand row order: concatenated dev batches joined to hand_oof, sorted by
    pair_id.  Returns (df, ids, start, ends) where df[start[i]:ends[i]] is pair i's full listed
    hand block (i matching meta['pair_ids'][i]) -- the SAME slicing step254_D2_pair.py /
    step261_pair_feature_expansion_proxy.py use to build the per-pair MECHPLUS aggregates."""
    feat = pl.concat([pl.read_parquet(f) for f in
                      sorted(glob.glob(ROOT + r'\outputs_v2\hands_development\batch_*.parquet'))]
                     ).select('pair_id', 'hand_id', 'table_id')
    hand_oof = pl.read_parquet(EVENT).select('pair_id', 'hand_id', 'table_id', 'fold')
    df = feat.join(hand_oof, on=['pair_id', 'hand_id', 'table_id'], how='inner')
    assert df.height == feat.height
    df = df.sort('pair_id')
    pid = df['pair_id'].to_numpy()
    ids, start = np.unique(pid, return_index=True)
    o = np.argsort(start)
    ids, start = ids[o], start[o]
    ends = np.append(start[1:], len(pid))
    return df, ids, start, ends


def pair_order_check(meta) -> None:
    """The step247 pair order, re-derived exactly as step254_D2_pair / step261 do it."""
    df, ids, start, ends = hand_frame()
    assert (ids == np.array(meta['pair_ids'])).all(), 'pair order differs from step247'
    ph = (pl.read_parquet(ROOT + r'\data\hands.parquet', columns=['hand_id', 'phase'])
          .join(df.select('hand_id').unique(), on='hand_id'))
    assert ph['phase'].unique().to_list() == ['development'], 'non-development hand in the frame'


def load_harness(extra: dict | None, extra_view: dict | None = None):
    """extra: arm name -> (n_pairs, k) float32 block, constant across views (the old EMB path).
    extra_view: arm name -> {view_key -> (n_pairs, k) float32 block}, ONE block per view (the
    --emb-per-hand path); view_key is 'full' or f'{view}_{seed}', same keys as VIEW_KEYS."""
    meta = json.load(open(S247 + r'\store_meta.json', encoding='utf-8'))
    blocks = np.load(S247 + r'\blocks.npz')
    Z = np.load(S261 + r'\mech_aggs.npz')            # the SIX aggregates, per view (read only)
    from evidence_aug.step5_evaluate import half
    hv = np.array([half(t) for t in meta['table']])
    lab = np.array(meta['label'], dtype=np.int64)
    fold = np.array(meta['fold'], dtype=np.int64)
    n_pairs = meta['n_pairs']
    low = np.zeros(n_pairs, bool)
    low[np.array(meta['lowdose_idx'], dtype=np.int64)] = True
    extra = extra or {}
    extra_view = extra_view or {}

    def X_of(arm, view, seed=None):
        key = 'full' if view == 'full' else f'{view}_{seed}'
        old = blocks['old_full'] if view == 'full' else blocks[f'old_{view}_{seed}']
        n = (blocks['n_full'] if view == 'full' else blocks[f'n_{view}_{seed}'])[:, None]
        parts = [old]
        if arm != 'OLD':
            parts.append(Z['six_' + key])
        if arm in extra:
            parts.append(extra[arm])
        if arm in extra_view:
            parts.append(extra_view[arm][key])
        return np.concatenate(parts + [n], axis=1)

    def score(ts):
        r = {'AP': round(float(np.mean([average_precision_score(lab, o) for o in ts])), 5),
             'AP_seeds': [round(float(average_precision_score(lab, o)), 5) for o in ts]}
        keep = low | (lab == 0)
        r['lowdose_AP'] = round(float(np.mean(
            [average_precision_score(lab[keep], o[keep]) for o in ts])), 5)
        for h in (0, 1):
            m = hv == h
            r[f'AP_half{h}'] = round(float(np.mean(
                [average_precision_score(lab[m], o[m]) for o in ts])), 5)
            r[f'lowdose_AP_half{h}'] = round(float(np.mean(
                [average_precision_score(lab[keep & m], o[keep & m]) for o in ts])), 5)
        return r

    return meta, lab, fold, n_pairs, X_of, score


def fit_predict(arm, s, lab, fold, X_of, n_pairs):
    """step261.fit_predict, LightGBM branch: pooled full + 70 % + 40 % views, weight 1/3."""
    Xpool = np.concatenate([X_of(arm, 'full'), X_of(arm, 'v70', s), X_of(arm, 'v40', s)], axis=0)
    ypool = np.tile(lab, 3)
    fpool = np.tile(fold, 3)
    w = np.full(len(ypool), 1 / 3)
    Xf, Xt = X_of(arm, 'full'), X_of(arm, 'test65', s)
    o1, o2 = np.zeros(n_pairs), np.zeros(n_pairs)
    for f in range(5):
        tr = fpool != f
        te = np.where(fold == f)[0]
        m = LGBMClassifier(**LGB).fit(Xpool[tr], ypool[tr], sample_weight=w[tr])
        o1[te] = m.predict_proba(Xf[te])[:, 1]
        o2[te] = m.predict_proba(Xt[te])[:, 1]
    return o1, o2


# --------------------------------------------------------------------------- the EMB block
def emb_block(meta, override: str | None = None) -> tuple[np.ndarray, dict]:
    """heads/pair_emb_features.parquet, put into the step247 pair order via keys_universe."""
    src = override or next((p for p in PAIR_EMB_CANDIDATES if os.path.exists(p)), None)
    assert src, (
        'heads/pair_emb_features.{parquet,csv} is not here yet - it comes back from the GPU box; '
        'run --baseline (and --noise-control) until then')
    ku = pl.read_parquet(KEYS_UNIVERSE, columns=['pair_id', 'pair_uid']).unique()
    assert ku['pair_uid'].n_unique() == ku.height == len(meta['pair_ids']), \
        'keys_universe pair_uid <-> pair_id is not 1:1 over the 1,860 pairs'
    pe = (pl.read_parquet(src) if src.endswith('.parquet')
          else pl.read_csv(src, schema_overrides={'pair_uid': pl.String}))
    assert 'pair_uid' in pe.columns, 'pair_emb_features.parquet has no pair_uid column'
    for forbidden in ('pair_id', 'hand_id', 'table_id', 'player_id'):
        assert forbidden not in pe.columns, f'{forbidden} came back from the box'
    cols = [c for c in pe.columns if c not in ('pair_uid', 'fold')]
    j = (pl.DataFrame({'pair_id': list(meta['pair_ids'])})
         .join(ku, on='pair_id', how='left', validate='1:1')
         .join(pe, on='pair_uid', how='left', validate='1:1'))
    assert j.height == len(meta['pair_ids'])
    miss = int(j[cols[0]].null_count())
    assert miss == 0, f'{miss} pairs have no embedding features'
    if 'fold' in pe.columns:
        assert (j['fold'].to_numpy() == np.array(meta['fold'])).all(), \
            'the fold column that came back disagrees with the step247 folds'
    X = j.select(cols).to_numpy().astype(np.float32)
    assert np.isfinite(X).all(), 'non-finite value in the embedding features'
    return X, {'n_cols': len(cols), 'cols': cols, 'source': src}


# --------------------------------------------------------------------------- the per-hand EMB block
def build_emb_per_hand(meta, h1_oof_path: str, pca_path: str) -> tuple[dict, dict]:
    """THE DOCUMENTED MAIN PATH.  Re-pool heads/h1_oof.npz (uid, p_oof) and heads/emb_pca32.npz
    (uid, z) on EVERY view the harness builds, using the harness's OWN per-pair RNG channel draws
    (the same [s, i, ch]).choice(n_i, ...) as step254_D2_pair.pair_aggs / this file's six MECHPLUS
    aggregates), so a hand survives a thinned view identically to those six aggregates.

    Only in_pool hands (the pair's fixed top-40, ranked on the GPU box) carry a p_oof / PCA-32 at
    all, so the pool a view draws from is the top-40 of the pair's FULL history filtered down by
    that view: a hand outside the original top-40 can never enter a thinned view's pool even
    where the harness's own RNG draw would have kept it -- PESSIMISTIC for EMB, the opposite bias
    from the old constant-block (--emb) mode.

    Returns ({view_key -> (n_pairs, 70) float32}, info_dict).  A pair with zero surviving pooled
    hands on a view gets an all-NaN row for that view (LightGBM treats NaN as missing)."""
    assert os.path.exists(h1_oof_path), (
        f'{h1_oof_path} is not here yet - it comes back from the GPU box; '
        'run --baseline (and --noise-control) until then')
    assert os.path.exists(pca_path), (
        f'{pca_path} is not here yet - it comes back from the GPU box; '
        'run --baseline (and --noise-control) until then')
    t0 = time.time()
    h1 = np.load(h1_oof_path)
    pcaz = np.load(pca_path)
    u_p, p_oof_raw = h1['uid'], h1['p_oof'].astype(np.float64)
    u_z, Zraw = pcaz['uid'], pcaz['z'].astype(np.float64)
    assert Zraw.ndim == 2 and Zraw.shape[1] == 32, f'emb_pca32.npz z has shape {Zraw.shape}'
    assert len(u_p) == len(set(u_p.tolist())), 'duplicate uid in h1_oof.npz'
    assert len(u_z) == len(set(u_z.tolist())), 'duplicate uid in emb_pca32.npz'

    ku = pl.read_parquet(KEYS_UNIVERSE,
                         columns=['pair_id', 'hand_id', 'uid', 'in_pool', 'label_pair'])
    df, ids, start, ends = hand_frame()
    assert (ids == np.array(meta['pair_ids'])).all(), 'pair order differs from step247'
    n_pairs = len(ids)

    # keys_universe.parquet is a curated candidate universe, NOT the harness's full per-pair hand
    # set: it is a genuine SUBSET of df (verified: every keys_universe row is found in df, but
    # df carries far more hands per pair -- avg ~120, min 57 -- than keys_universe's avg ~42), so
    # most df hand positions legitimately have no keys_universe match at all (not an error).
    dfe = df.select('pair_id', 'hand_id').join(ku, on=['pair_id', 'hand_id'], how='left',
                                                maintain_order='left')
    assert dfe.height == df.height
    n_matched = int(dfe['uid'].is_not_null().sum())

    h1_df = pl.DataFrame({'uid': u_p, 'p_oof': p_oof_raw})
    pca_df = pl.DataFrame(Zraw, schema=[f'z{k:02d}' for k in range(32)]).with_columns(
        pl.Series('uid', u_z))
    j = (dfe.select('uid', 'in_pool', 'label_pair')
         .join(h1_df, on='uid', how='left', maintain_order='left')
         .join(pca_df, on='uid', how='left', maintain_order='left'))
    assert j.height == dfe.height
    in_pool_arr = j['in_pool'].to_numpy()
    label_pair_arr = j['label_pair'].to_numpy()
    p_pos = j['p_oof'].to_numpy().astype(np.float64)
    z_pos = j.select([f'z{k:02d}' for k in range(32)]).to_numpy().astype(np.float64)
    have = (~np.isnan(p_pos)) & (~np.isnan(z_pos).any(axis=1))
    assert have.any(), 'h1_oof.npz / emb_pca32.npz share no usable uid with keys_universe'
    assert bool(in_pool_arr[have].all()), (
        'a usable uid maps to a hand that keys_universe marks NOT in_pool - '
        'h1_oof.npz / emb_pca32.npz should only ever carry in_pool rows')

    # the clean-pair (label_pair == 0) q99.9 / q99 thresholds, fit_heads.py's H2 recipe, computed
    # ONCE here on the full (untrimmed) usable set -- never recomputed per view.
    clean = have & (label_pair_arr == 0)
    assert clean.sum() > 100, 'too few clean (label_pair==0) rows to fix q99/q999'
    q99 = float(np.quantile(p_pos[clean], .99))
    q999 = float(np.quantile(p_pos[clean], .999))

    emb_view = {k: np.full((n_pairs, 70), np.nan, dtype=np.float32) for k in VIEW_KEYS}
    counts = {k: np.zeros(n_pairs, dtype=np.int64) for k in VIEW_KEYS}
    p_i = z_i = None                              # bound inside the loop, read by pool()

    def pool(mask):
        k = int(mask.sum())
        if k == 0:
            return None, 0
        agg6 = pair_aggs6(p_i[mask], q99, q999)
        zz = z_i[mask]
        return np.concatenate([agg6, zz.mean(0).astype(np.float32),
                               zz.max(0).astype(np.float32)]), k

    for i in range(n_pairs):
        a, b = int(start[i]), int(ends[i])
        n_i = b - a
        have_i = have[a:b]
        p_i = p_pos[a:b]
        z_i = z_pos[a:b]

        row, k = pool(have_i)                     # 'full' view: every in_pool hand of the pair
        if row is not None:
            emb_view['full'][i] = row
        counts['full'][i] = k

        for s in SEEDS:
            for nm, (ch, frac) in CH.items():
                key = f'{nm}_{s}'
                kk = max(1, int(round(frac * n_i)))
                idx = np.random.default_rng([s, i, ch]).choice(n_i, size=kk, replace=False)
                mask = np.zeros(n_i, dtype=bool)
                mask[idx] = True
                mask &= have_i                     # only in_pool hands that also survived
                row, k = pool(mask)
                if row is not None:
                    emb_view[key][i] = row
                counts[key][i] = k

    qs = (0.0, .05, .25, .5, .75, .95, 1.0)
    info = {
        'n_cols': 70, 'mode': 'per_hand',
        'source_h1_oof': h1_oof_path, 'source_pca': pca_path,
        'n_h1_oof_uids': int(len(u_p)), 'n_pca_uids': int(len(u_z)),
        'n_usable_uids': int(have.sum()),
        'thresholds': {'clean_q99': q99, 'clean_q999': q999, 'n_clean_rows': int(clean.sum())},
        'counts_quantiles': {k: {str(q): float(np.quantile(counts[k], q)) for q in qs}
                             for k in VIEW_KEYS},
        'seconds': round(time.time() - t0, 1),
        'note': ('the pool is the top-40 of the pair\'s FULL history filtered by the view: a '
                 'hand outside the original top-40 in_pool set cannot enter a thinned view\'s '
                 'pool even where the harness RNG draw would have kept it, so this mode is '
                 'slightly PESSIMISTIC for EMB (the opposite bias from the --emb constant-block '
                 'mode).  Pooled with the same per-pair RNG channel draws as the six MECHPLUS '
                 'aggregates (236/237/238, CH_V70=10, CH_V40=11, CH_TEST65=90) so a hand '
                 'survives a view identically.  Pairs with zero surviving pooled hands on a view '
                 'get NaN (LightGBM treats it as missing) - see counts_quantiles above.')}
    return emb_view, info


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', action='store_true', help='OLD and M6 only (reproduction)')
    ap.add_argument('--noise-control', action='store_true', dest='noise',
                    help='add NOISE70: 70 Gaussian columns instead of EMB')
    ap.add_argument('--emb', action='store_true',
                    help='add EMB via the OLD constant-block path (needs heads/'
                         'pair_emb_features.*; wire-test / fallback only, not the main path)')
    ap.add_argument('--emb-file', default=None, dest='emb_file',
                    help='override the pair_emb_features path (used to WIRE-TEST the --emb path '
                         'against a synthetic file; point --out somewhere disposable)')
    ap.add_argument('--emb-per-hand', action='store_true', dest='emb_per_hand',
                    help='add EMB via the per-hand path (THE MAIN PATH): re-pool heads/h1_oof.npz '
                         'and heads/emb_pca32.npz on EVERY view the harness builds instead of '
                         'using a view-blind constant block.  --noise-control redraws NOISE70 '
                         'per view too in this mode.')
    ap.add_argument('--h1-oof', default=EXP + r'\heads\h1_oof.npz', dest='h1_oof',
                    help='--emb-per-hand: uid, p_oof over the in_pool universe (fit_heads.py H1). '
                         'Overridable to WIRE-TEST against a synthetic file.')
    ap.add_argument('--pca', default=EXP + r'\heads\emb_pca32.npz', dest='pca',
                    help='--emb-per-hand: uid, z (float16 N x 32), per-fold OOF PCA-32 '
                         '(fit_heads.py H2). Overridable to WIRE-TEST against a synthetic file.')
    ap.add_argument('--out', default=EVAL + r'\pair_read.json')
    a = ap.parse_args(argv)
    assert a.baseline or a.noise or a.emb or a.emb_per_hand, (
        'pass --baseline, --noise-control, --emb and/or --emb-per-hand')
    os.makedirs(EVAL, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    t0 = time.time()

    meta0 = json.load(open(S247 + r'\store_meta.json', encoding='utf-8'))
    pair_order_check(meta0)
    print('pair order and development phase verified', round(time.time() - t0, 1), 's', flush=True)

    extra, extra_view, extra_info = {}, {}, {}
    if a.emb_per_hand:
        if a.noise:
            rng = np.random.default_rng(NOISE_SEED)
            extra_view['NOISE70'] = {k: rng.standard_normal(
                (meta0['n_pairs'], NOISE_COLS)).astype(np.float32) for k in VIEW_KEYS}
            extra_info['NOISE70'] = {'n_cols': NOISE_COLS, 'seed': NOISE_SEED,
                                     'note': 'per-hand mode: one fresh draw PER VIEW (10 view '
                                             'keys), matching the STEP261 NOISE arm rather than '
                                             'the constant mode\'s single per-pair draw'}
        extra_view['EMB'], extra_info['EMB'] = build_emb_per_hand(meta0, a.h1_oof, a.pca)
    else:
        if a.noise:
            extra['NOISE70'] = np.random.default_rng(NOISE_SEED).standard_normal(
                (meta0['n_pairs'], NOISE_COLS)).astype(np.float32)
            extra_info['NOISE70'] = {'n_cols': NOISE_COLS, 'seed': NOISE_SEED,
                                     'note': 'one draw per pair, identical in every view - the '
                                             'matched dilution reference for EMB'}
        if a.emb:
            extra['EMB'], extra_info['EMB'] = emb_block(meta0, a.emb_file)

    meta, lab, fold, n_pairs, X_of, score = load_harness(extra, extra_view)
    arms = ['OLD', 'M6'] + [k for k in ('NOISE70', 'EMB') if k in extra or k in extra_view]

    res = {}
    if os.path.exists(a.out):
        res = json.load(open(a.out, encoding='utf-8'))
    res.setdefault('step', 'STEP268 C pair-proxy read (STEP247 / STEP254-D2 / STEP261 harness)')
    res['registered_rule'] = RULE
    res['mode'] = 'per_hand' if a.emb_per_hand else 'constant'
    res['harness'] = {'n_pairs': int(n_pairs), 'n_lowdose': int(len(meta['lowdose_idx'])),
                      'seeds': SEEDS, 'view_channels': {k: v[0] for k, v in CH.items()},
                      'lgb': {k: v for k, v in LGB.items() if k != 'n_jobs'},
                      'M6_T2_AP_registered': M6_T2_AP_REGISTERED}
    if a.emb_per_hand:
        res['harness']['note_emb_view_thinning'] = (
            'in --emb-per-hand mode the EMB (and NOISE70) block IS re-pooled on every thinned '
            'view, using the harness\'s own per-pair RNG channel draws so a hand survives a '
            'view identically to the six MECHPLUS aggregates.  The pool a view draws from is '
            'the top-40 of the pair\'s FULL history filtered down by that view: a hand outside '
            'the original top-40 in_pool set can never enter a thinned view\'s pool even where '
            'the harness RNG draw would have kept it, so this mode is slightly PESSIMISTIC for '
            'EMB (the opposite bias from the old --emb constant-block mode)')
    else:
        res['harness']['caveat_emb_not_view_thinned'] = (
            'the EMB and NOISE70 blocks are pooled over the pair\'s fixed top-40 '
            'pool on the GPU box and cannot be re-pooled on a thinned view, so '
            'they are IDENTICAL in T1 and T2 while the OLD block and the six '
            'MECHPLUS aggregates are thinned - T2 is biased in their favour')
    res.setdefault('arms', {})
    res.setdefault('extra_blocks', {}).update(extra_info)

    for arm in arms:
        if arm in res['arms'] and arm in ('OLD', 'M6') and not (
                a.baseline or a.emb or a.noise or a.emb_per_hand):
            continue
        t1s, t2s = [], []
        t1 = time.time()
        for s in SEEDS:
            o1, o2 = fit_predict(arm, s, lab, fold, X_of, n_pairs)
            t1s.append(o1)
            t2s.append(o2)
        res['arms'][arm] = {'dim': int(X_of(arm, 'full').shape[1]),
                            'T1': score(t1s), 'T2': score(t2s),
                            'seconds': round(time.time() - t1, 1)}
        print(arm, json.dumps(res['arms'][arm]), flush=True)

    # ---- the M6 reproduction, asserted
    m6 = res['arms']['M6']['T2']['AP']
    res['reproduction'] = {'M6_T2_AP': m6, 'registered': M6_T2_AP_REGISTERED,
                           'abs_diff': round(abs(m6 - M6_T2_AP_REGISTERED), 6),
                           'tolerance': REPRO_TOL,
                           'OK': abs(m6 - M6_T2_AP_REGISTERED) < REPRO_TOL}
    print('M6 T2 AP %.5f   registered %.5f   |diff| %.6f   %s'
          % (m6, M6_T2_AP_REGISTERED, abs(m6 - M6_T2_AP_REGISTERED),
             'OK' if res['reproduction']['OK'] else 'MISMATCH'), flush=True)

    # ---- deltas and the registered rule, evaluated mechanically
    res['deltas_vs_M6'] = {}
    res['rule'] = {}
    for arm in ('NOISE70', 'EMB'):
        if arm not in res['arms']:
            continue
        d = {v: {k: round(res['arms'][arm][v][k] - res['arms']['M6'][v][k], 5)
                 for k in res['arms']['M6'][v] if k != 'AP_seeds'} for v in ('T1', 'T2')}
        res['deltas_vs_M6'][arm] = d
        conds = {'T2_AP_delta>=+.003': d['T2']['AP'] >= .003,
                 'T2_AP_half0_delta>=0': d['T2']['AP_half0'] >= 0,
                 'T2_AP_half1_delta>=0': d['T2']['AP_half1'] >= 0,
                 'T2_lowdose_AP_delta>=0': d['T2']['lowdose_AP'] >= 0}
        res['rule'][arm] = {'conditions': {k: bool(v) for k, v in conds.items()},
                            'verdict': 'PASS' if all(conds.values()) else 'FAIL',
                            'rule': RULE}
        print(arm, 'vs M6  T2', json.dumps(d['T2']), '->', res['rule'][arm]['verdict'], flush=True)

    res['wall_seconds_last_run'] = round(time.time() - t0, 1)
    res['arms_this_run'] = arms
    json.dump(res, open(a.out, 'w', encoding='utf-8'), indent=1)
    print('wrote', a.out)
    assert res['reproduction']['OK'], (
        'M6 did not reproduce the registered T2 AP %.5f (got %.5f) - the harness recipe drifted'
        % (M6_T2_AP_REGISTERED, m6))


if __name__ == '__main__':
    main()
