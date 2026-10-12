"""STEP268 B2 -- the three heads on top of the frozen embeddings.  numpy + sklearn + lightgbm only.

NOTHING IN THIS DIRECTORY HAS EVER RUN ON A GPU (or on a real embedding matrix).  Only
`--dry-run`, which synthesises random unit vectors when emb_universe.npy is absent, and
`py_compile` have been executed.  See RUNBOOK_268.txt, "VERIFY ON THE GPU BOX".

INPUTS   emb_universe.npy (float16, N x 1024, L2-normalised), emb_uids.txt (its row order),
         universe_meta.jsonl, heldout_scope57_uids.txt, heldout_subset_uids.txt
         Not one of these carries a pair_id, hand_id, table_id or player_id.  `pair_uid` is an
         opaque 48-bit token; it groups a pair's rows and says nothing else.

H1  EVIDENCE.  label = `listed`.
    train rows  = (in_pool OR extra_listed) AND fold in {0,1,2}
    heads       = LogisticRegression(C=1.0)            [deterministic, fitted once]
                  LGBMClassifier(300 trees, 31 leaves, lr .05, colsample_bytree .5) x 3 seeds
                  MLPClassifier((256, 64), early_stopping=True)                     x 3 seeds
                  (LR and the MLP see a StandardScaler fitted on the training rows; LightGBM
                  does not need one.  That scaler is the only thing added to the registered
                  head list and it is fitted inside every train split, never across one.)
    p_lgb / p_mlp are the mean over their seeds, mean_conf = 100 * mean(p_lr, p_lgb, p_mlp).
    -> heads/h1_scope57_scores.jsonl, heads/h1_subset_scores.jsonl   (harness score-file format)
    and a 5-fold OOF by `fold` over the WHOLE in_pool universe -> heads/h1_oof.npz (uid, p_oof).

H2  RISK.  Per pair_uid, over that pair's in_pool rows:
      * the six STEP254-D2 aggregates of p_oof: count above the CLEAN-PAIR q99.9 and q99 of
        p_oof, top-1 / top-3 / top-5 mean logit, noisy-or;
      * mean and max of a PCA-32 of the embedding.  The PCA is fitted PER FOLD on the in_pool
        rows of the OTHER folds, so a pair's 32+32 columns never saw its own fold.
    -> heads/pair_emb_features.parquet (and .csv, always) - pair_uid, fold, 6 + 64 columns.
    -> heads/emb_pca32.npz - the same per-fold OOF PCA-32 coordinates, PER HAND (uid, z) rather
       than pre-pooled per pair, so step268_pair_emb_read.py --emb-per-hand can re-pool them
       (with h1_oof.npz's p_oof) on every view the pair-proxy harness builds.

H3  BEHAVIOUR.  Positive pairs only: pooled MEAN embedding -> 3-class LogisticRegression,
    5-fold OOF by fold -> heads/h3_family_oof.jsonl + accuracy against the true family.

USAGE
  python fit_heads.py --emb emb_universe.npy --uids emb_uids.txt --meta universe_meta.jsonl \\
      --scope57 heldout_scope57_uids.txt --subset heldout_subset_uids.txt --out heads/ \\
      --seeds 268,269,270
  python fit_heads.py --dry-run --out heads_dry/            (IO + shapes, tiny models)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FAMS = ('coordinated_isolation', 'directed_transfer', 'soft_play')   # sorted = LR class order
TRAIN_FOLDS = (0, 1, 2)
N_FOLDS = 5
PCA_K = 32
AGG_NAMES = ('cnt_q999', 'cnt_q99', 'top1_logit', 'top3_logit', 'top5_logit', 'noisy_or')
LR_C = 1.0
LGB_KW = dict(n_estimators=300, num_leaves=31, learning_rate=.05, colsample_bytree=.5,
              verbose=-1)
MLP_KW = dict(hidden_layer_sizes=(256, 64), early_stopping=True, max_iter=200)


# --------------------------------------------------------------------------- IO
def read_meta(path):
    rows = []
    with open(path, encoding='utf-8') as fh:
        for ln in fh:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def read_list(path):
    with open(path, encoding='utf-8') as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def pair_aggs6(p, q99, q999):
    """The STEP254-D2 aggregate set, identical to src/evidence_aug/step254_D2_pair.pair_aggs."""
    c = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(c / (1 - c))
    s = np.sort(lg)[::-1]
    return np.array([float((p > q999).sum()), float((p > q99).sum()), float(s[:1].mean()),
                     float(s[:3].mean()), float(s[:5].mean()),
                     float(-np.log1p(-np.clip(p, 0, 1 - 1e-9)).sum())], dtype=np.float64)


def write_table(path_noext, colnames, rows):
    """CSV always; parquet too when pyarrow (or pandas) happens to be installed."""
    out = {}
    with open(path_noext + '.csv', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(','.join(colnames) + '\n')
        for r in rows:
            fh.write(','.join(str(x) for x in r) + '\n')
    out['csv'] = path_noext + '.csv'
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        cols = list(zip(*rows)) if rows else [[] for _ in colnames]
        arrs = []
        for name, col in zip(colnames, cols):
            arrs.append(pa.array(list(col), type=pa.string() if name == 'pair_uid'
                                 else (pa.int64() if name == 'fold' else pa.float64())))

        pq.write_table(pa.Table.from_arrays(arrs, names=colnames), path_noext + '.parquet')
        out['parquet'] = path_noext + '.parquet'
    except Exception as e:                                          # noqa: BLE001
        print(f'  (no parquet: {type(e).__name__}: {e}; the .csv is the deliverable)', flush=True)
    return out


# --------------------------------------------------------------------------- the head bank
def fit_bank(Xtr, ytr, Xte_list, seeds, fast=False):
    """Returns (p_lr, p_lgb, p_mlp) for every matrix in Xte_list."""
    from lightgbm import LGBMClassifier
    lgb_kw = dict(LGB_KW)
    mlp_kw = dict(MLP_KW)
    if fast:
        lgb_kw['n_estimators'] = 5
        mlp_kw['max_iter'] = 3
        mlp_kw['hidden_layer_sizes'] = (16,)
    out = [{} for _ in Xte_list]

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=2000))
    lr.fit(Xtr, ytr)
    for i, Xe in enumerate(Xte_list):
        out[i]['p_lr'] = lr.predict_proba(Xe)[:, 1]

    acc = [np.zeros(len(Xe)) for Xe in Xte_list]
    for s in seeds:
        m = LGBMClassifier(random_state=int(s), n_jobs=-1, **lgb_kw).fit(Xtr, ytr)
        for i, Xe in enumerate(Xte_list):
            acc[i] += m.predict_proba(Xe)[:, 1]
    for i in range(len(Xte_list)):
        out[i]['p_lgb'] = acc[i] / len(seeds)

    acc = [np.zeros(len(Xe)) for Xe in Xte_list]
    for s in seeds:
        m = make_pipeline(StandardScaler(),
                          MLPClassifier(random_state=int(s), **mlp_kw)).fit(Xtr, ytr)
        for i, Xe in enumerate(Xte_list):
            acc[i] += m.predict_proba(Xe)[:, 1]
    for i in range(len(Xte_list)):
        out[i]['p_mlp'] = acc[i] / len(seeds)
        out[i]['p'] = (out[i]['p_lr'] + out[i]['p_lgb'] + out[i]['p_mlp']) / 3.0
    return out


def metrics(y, s):
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {'n': int(len(y)), 'n_pos': int(y.sum()), 'note': 'degenerate'}
    return {'n': int(len(y)), 'n_pos': int(y.sum()),
            'ap': round(float(average_precision_score(y, s)), 5),
            'auc': round(float(roc_auc_score(y, s)), 5)}


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', default='emb_universe.npy')
    ap.add_argument('--uids', default='emb_uids.txt')
    ap.add_argument('--meta', default='universe_meta.jsonl')
    ap.add_argument('--scope57', default='heldout_scope57_uids.txt')
    ap.add_argument('--subset', default='heldout_subset_uids.txt')
    ap.add_argument('--out', default='heads/')
    ap.add_argument('--seeds', default='268,269,270')
    ap.add_argument('--tag', default='qwen3emb06b_last')
    ap.add_argument('--dry-run', action='store_true', dest='dry')
    ap.add_argument('--dry-pairs', type=int, default=120, dest='dry_pairs')
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(',') if x.strip()]
    os.makedirs(a.out, exist_ok=True)
    T0 = time.time()
    timings = {}

    meta = read_meta(a.meta)
    m_uid = [r['uid'] for r in meta]
    scope57 = read_list(a.scope57)
    subset = read_list(a.subset)

    fold = np.array([r['fold'] for r in meta])
    listed = np.array([r['listed'] for r in meta])
    label_pair = np.array([r['label_pair'] for r in meta])
    in_pool = np.array([r['in_pool'] for r in meta], dtype=bool)
    extra = np.array([r['extra_listed'] for r in meta], dtype=bool)
    src = np.array([r['source_class'] for r in meta])
    fam = np.array([r['family'] for r in meta])
    puid = np.array([r['pair_uid'] for r in meta])

    # --dry-run works on a handful of pairs so the shapes are exercised in seconds
    sub = np.arange(len(m_uid))
    if a.dry:
        keep_p = set(sorted(set(puid.tolist()))[:a.dry_pairs])
        sub = np.flatnonzero(np.array([p in keep_p for p in puid]))
        fold, listed, label_pair = fold[sub], listed[sub], label_pair[sub]
        in_pool, extra, src, fam, puid = in_pool[sub], extra[sub], src[sub], fam[sub], puid[sub]
        m_uid = [m_uid[i] for i in sub]
        have = set(m_uid)
        scope57 = [u for u in scope57 if u in have]
        subset = [u for u in subset if u in have]
        print(f'DRY RUN: {len(keep_p)} pairs, {len(m_uid)} rows, scope57 {len(scope57)}, '
              f'subset {len(subset)}', flush=True)

    if os.path.exists(a.emb) and os.path.exists(a.uids):
        E = np.load(a.emb)
        e_uid = read_list(a.uids)
        assert len(e_uid) == E.shape[0], 'emb_uids.txt does not match the .npy row count'
        pos = {u: i for i, u in enumerate(e_uid)}
        missing = [u for u in m_uid if u not in pos]
        assert not missing, f'{len(missing)} universe rows have no embedding ({missing[:3]})'
        X = E[np.array([pos[u] for u in m_uid])].astype(np.float32)
        del E
        synth = False
    else:
        assert a.dry, f'{a.emb} / {a.uids} not found (only --dry-run may synthesise them)'
        X = np.random.default_rng(268).standard_normal((len(m_uid), 1024), dtype=np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        synth = True
        print('DRY RUN: emb_universe.npy absent, using RANDOM UNIT VECTORS', flush=True)

    uidx = {u: i for i, u in enumerate(m_uid)}
    for nm, lst in (('scope57', scope57), ('subset', subset)):
        miss = [u for u in lst if u not in uidx]
        assert not miss, f'{len(miss)} {nm} uids are not in the universe meta'
    i57 = np.array([uidx[u] for u in scope57], dtype=np.int64)
    isub = np.array([uidx[u] for u in subset], dtype=np.int64)

    summary = {'tag': a.tag, 'DRY_RUN': bool(a.dry), 'synthetic_embeddings': bool(synth),
               'NOTE': 'nothing in gpu/ has ever run on a GPU; this file records one run of '
                       'fit_heads.py only',
               'emb': {'path': a.emb, 'n_rows': int(X.shape[0]), 'dim': int(X.shape[1])},
               'universe': {'n_rows': len(m_uid), 'n_pairs': int(len(set(puid.tolist()))),
                            'n_in_pool': int(in_pool.sum()), 'n_extra_listed': int(extra.sum()),
                            'n_listed': int((listed == 1).sum())},
               'heads': {'lr': {'C': LR_C}, 'lgb': LGB_KW, 'mlp': MLP_KW, 'seeds': seeds}}

    # ---------------------------------------------------------------- H1 evidence heads
    t = time.time()
    tr = (in_pool | extra) & np.isin(fold, TRAIN_FOLDS)
    assert tr.sum() > 0 and listed[tr].sum() > 0, 'no positive training row'
    assert not np.isin(fold[i57], TRAIN_FOLDS).any() or a.dry, \
        'a scope57 row sits in a training fold - the held-out read would be circular'
    out57, outsub = fit_bank(X[tr], listed[tr], [X[i57], X[isub]], seeds, fast=a.dry)
    timings['h1_heads_seconds'] = round(time.time() - t, 1)
    summary['h1_train'] = {'n_rows': int(tr.sum()), 'n_listed': int(listed[tr].sum()),
                           'folds': list(TRAIN_FOLDS)}

    for name, ii, o in (('scope57', i57, out57), ('subset', isub, outsub)):
        p = os.path.join(a.out, f'h1_{name}_scores.jsonl')
        with open(p, 'w', encoding='utf-8', newline='\n') as fh:
            for j, u in enumerate([m_uid[k] for k in ii]):
                fh.write(json.dumps({'uid': u, 'k': 1,
                                     'mean_conf': round(100.0 * float(o['p'][j]), 4),
                                     'p_lr': round(float(o['p_lr'][j]), 6),
                                     'p_lgb': round(float(o['p_lgb'][j]), 6),
                                     'p_mlp': round(float(o['p_mlp'][j]), 6)}) + '\n')
        y = (listed[ii] == 1).astype(int)
        blk = {'overall': metrics(y, o['p']),
               'by_head': {k: metrics(y, o[k]) for k in ('p_lr', 'p_lgb', 'p_mlp')},
               'by_source_class': {}, 'by_family': {},
               'conf': {'mean': round(float(100 * o['p'].mean()), 3),
                        'min': round(float(100 * o['p'].min()), 3),
                        'max': round(float(100 * o['p'].max()), 3)}}
        for c in sorted(set(src[ii].tolist())):
            m = src[ii] == c
            blk['by_source_class'][c] = metrics(y[m], o['p'][m])
        for f in sorted(set(fam[ii].tolist())):
            m = fam[ii] == f
            blk['by_family'][f] = metrics(y[m], o['p'][m])
        summary[f'h1_{name}'] = blk
        print(f'H1 {name}: {json.dumps(blk["overall"])}', flush=True)

    # ---------------------------------------------------------------- H1 OOF over the pool
    t = time.time()
    pool = np.flatnonzero(in_pool)
    p_oof = np.full(len(m_uid), np.nan)
    oof_parts = {k: np.full(len(m_uid), np.nan) for k in ('p_lr', 'p_lgb', 'p_mlp')}
    for f in range(N_FOLDS):
        te = pool[fold[pool] == f]
        trf = pool[fold[pool] != f]
        if len(te) == 0 or listed[trf].sum() == 0:
            continue
        o, = fit_bank(X[trf], listed[trf], [X[te]], seeds, fast=a.dry)
        p_oof[te] = o['p']
        for k in oof_parts:
            oof_parts[k][te] = o[k]
        print(f'  H1 OOF fold {f}: train {len(trf)} / test {len(te)}  '
              f'{time.time() - t:.0f}s', flush=True)
    timings['h1_oof_seconds'] = round(time.time() - t, 1)
    got = ~np.isnan(p_oof)
    assert got[pool].all() or a.dry, 'some in_pool row has no OOF probability'
    np.savez(os.path.join(a.out, 'h1_oof.npz'),
             uid=np.array([m_uid[i] for i in np.flatnonzero(got)]),
             p_oof=p_oof[got], fold=fold[got],
             **{k: v[got] for k, v in oof_parts.items()})
    summary['h1_oof'] = {'n': int(got.sum()),
                         'overall': metrics((listed[got] == 1).astype(int), p_oof[got])}
    print('H1 OOF:', json.dumps(summary['h1_oof']['overall']), flush=True)

    # ---------------------------------------------------------------- H2 pair features
    t = time.time()
    clean = got & in_pool & (label_pair == 0)
    q99 = float(np.quantile(p_oof[clean], .99))
    q999 = float(np.quantile(p_oof[clean], .999))
    summary['h2_thresholds'] = {'clean_q99': round(q99, 6), 'clean_q999': round(q999, 6),
                                'n_clean_pool_rows': int(clean.sum())}

    Z = np.full((len(m_uid), PCA_K), np.nan, dtype=np.float64)
    pca_info = {}
    for f in range(N_FOLDS):
        te = pool[fold[pool] == f]
        trf = pool[fold[pool] != f]
        if len(te) == 0 or len(trf) <= PCA_K:
            continue
        pca = PCA(n_components=PCA_K, random_state=seeds[0]).fit(X[trf])
        Z[te] = pca.transform(X[te])
        pca_info[str(f)] = {'n_fit_rows': int(len(trf)),
                            'explained_variance_ratio_sum': round(
                                float(pca.explained_variance_ratio_.sum()), 5)}
    summary['h2_pca'] = {'k': PCA_K, 'fitted_per_fold_on_other_folds': True, 'folds': pca_info}

    # per-hand PCA-32, saved uid by uid (not pre-pooled) so step268_pair_emb_read.py can re-pool
    # it -- together with h1_oof.npz's p_oof -- on every view the pair-proxy harness builds,
    # instead of only on the pair's fixed top-40 pool the way pair_emb_features.parquet is.
    zgot = in_pool & ~np.isnan(Z).any(axis=1)
    np.savez(os.path.join(a.out, 'emb_pca32.npz'),
             uid=np.array([m_uid[i] for i in np.flatnonzero(zgot)]),
             z=Z[zgot].astype(np.float16))
    summary['h2_pca']['n_saved_emb_pca32'] = int(zgot.sum())
    print(f'wrote emb_pca32.npz: {int(zgot.sum())} in_pool rows', flush=True)

    by_pair = {}
    for i in pool:
        by_pair.setdefault(puid[i], []).append(i)
    cols = (['pair_uid', 'fold'] + ['agg_' + n for n in AGG_NAMES]
            + [f'pca_mean_{k:02d}' for k in range(PCA_K)]
            + [f'pca_max_{k:02d}' for k in range(PCA_K)])
    rows = []
    for pu in sorted(by_pair):
        ii = np.array(by_pair[pu])
        fs = set(fold[ii].tolist())
        assert len(fs) == 1, f'pair {pu} straddles folds {sorted(fs)} - folds are by table'
        pv = p_oof[ii]
        pv = np.where(np.isnan(pv), 0.5, pv)
        zz = Z[ii]
        if np.isnan(zz).any():
            zz = np.nan_to_num(zz)
        rows.append([pu, int(fold[ii[0]])] + [float(x) for x in pair_aggs6(pv, q99, q999)]
                    + [float(x) for x in zz.mean(0)] + [float(x) for x in zz.max(0)])
    files = write_table(os.path.join(a.out, 'pair_emb_features'), cols, rows)
    timings['h2_seconds'] = round(time.time() - t, 1)
    summary['h2'] = {'n_pairs': len(rows), 'n_cols': len(cols) - 2, 'files': files,
                     'columns': cols}
    print(f'H2: {len(rows)} pairs x {len(cols) - 2} columns -> {files}', flush=True)

    # ---------------------------------------------------------------- H3 behaviour family
    t = time.time()
    posmask = in_pool & (label_pair == 1)
    pp, pf, PX = [], [], []
    for pu in sorted(set(puid[posmask].tolist())):
        ii = np.flatnonzero(posmask & (puid == pu))
        f = set(fam[ii].tolist())
        assert len(f) == 1, f'positive pair {pu} carries several families {sorted(f)}'
        fv = f.pop()
        if fv not in FAMS:
            continue
        pp.append(pu)
        pf.append(fv)
        PX.append(X[ii].mean(0))
    h3 = {'n_pairs': len(pp)}
    if len(pp) >= 2 * N_FOLDS and len(set(pf)) > 1:
        PX = np.asarray(PX, dtype=np.float64)
        yf = np.array([FAMS.index(x) for x in pf])
        pfold = np.array([int(fold[np.flatnonzero(puid == pu)[0]]) for pu in pp])
        P = np.full((len(pp), len(FAMS)), np.nan)
        for f in range(N_FOLDS):
            te = np.flatnonzero(pfold == f)
            trf = np.flatnonzero(pfold != f)
            if len(te) == 0 or len(set(yf[trf].tolist())) < 2:
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=LR_C, max_iter=2000)).fit(PX[trf], yf[trf])
            pr = m.predict_proba(PX[te])
            for j, cl in enumerate(m.classes_):
                P[te, cl] = pr[:, j]
        ok = ~np.isnan(P).any(1)
        pred = np.where(ok, np.nanargmax(np.nan_to_num(P, nan=-1), axis=1), -1)
        h3['accuracy'] = round(float((pred[ok] == yf[ok]).mean()), 5)
        h3['n_scored'] = int(ok.sum())
        h3['per_family'] = {FAMS[c]: {'n': int((yf[ok] == c).sum()),
                                      'recall': round(float(
                                          (pred[ok][yf[ok] == c] == c).mean()), 5)
                                      if (yf[ok] == c).any() else None}
                            for c in range(len(FAMS))}
        with open(os.path.join(a.out, 'h3_family_oof.jsonl'), 'w', encoding='utf-8',
                  newline='\n') as fh:
            for j, pu in enumerate(pp):
                if not ok[j]:
                    continue
                fh.write(json.dumps({'pair_uid': pu, 'fold': int(pfold[j]),
                                     'probs': {FAMS[c]: round(float(P[j, c]), 6)
                                               for c in range(len(FAMS))},
                                     'argmax': FAMS[int(pred[j])],
                                     'true_family': pf[j]}) + '\n')
    else:
        h3['note'] = 'too few positive pairs to cross-validate'
    timings['h3_seconds'] = round(time.time() - t, 1)
    summary['h3_family'] = h3
    print('H3:', json.dumps(h3.get('accuracy', h3)), flush=True)

    timings['total_seconds'] = round(time.time() - T0, 1)
    summary['timings'] = timings
    with open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=1)
    print('wrote', os.path.join(a.out, 'summary.json'), 'in', timings['total_seconds'], 's')


if __name__ == '__main__':
    main()
