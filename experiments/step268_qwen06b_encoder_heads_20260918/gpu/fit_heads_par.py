"""STEP268 B2 -- parallel-fold driver for fit_heads.py's H1 out-of-fold loop.

fit_heads.py runs, sequentially: the held-out bank (scope57 + subset), a 5-fold OOF loop
over in_pool rows (the slow part -- ~40 min/fold on the GPU box because the MLP/logistic
heads barely touch the 64 cores), then H2 (PCA-32 pooled embeddings + pair aggregate table)
and H3 (family logistic) and summary.json.  This file lets the 5 OOF folds run as 5
independent OS processes and adds a --merge step that stitches their output back into
EXACTLY the files fit_heads.py itself would have produced.

It changes nothing about the modelling: same data loading, same fold definition, same
fit_bank hyper-parameters and seeds, same H1/H2/H3 code (imported from fit_heads.py, or
copied verbatim where it had to change).  The only functional difference from fit_heads.py
is that LightGBM's n_jobs is now a CLI flag (--n-jobs) instead of the hard-coded -1, so five
concurrent fold processes do not all try to grab every core at once; fit_bank is copied here
with that one line changed, everything else in it is byte-for-byte identical to
fit_heads.fit_bank.  Thread limits for the underlying BLAS/OpenMP libraries are the
launcher's job (set OMP_NUM_THREADS etc. in the environment of each process); this script
does not touch them.

USAGE (five parallel folds + merge, from the gpu/ directory, real data)
  for f in 0 1 2 3 4; do
    OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12 \\
      python fit_heads_par.py --emb ../emb/emb_universe.npy --uids ../emb/emb_uids.txt \\
        --meta ../data/universe_meta.jsonl --scope57 ../data/heldout_scope57_uids.txt \\
        --subset ../data/heldout_subset_uids.txt --out ../heads/ --seeds 268,269,270 \\
        --fold $f --n-jobs 12 &
  done
  wait
  python fit_heads_par.py --emb ../emb/emb_universe.npy --uids ../emb/emb_uids.txt \\
      --meta ../data/universe_meta.jsonl --scope57 ../data/heldout_scope57_uids.txt \\
      --subset ../data/heldout_subset_uids.txt --out ../heads/ --seeds 268,269,270 \\
      --merge --skip-heldout

  python fit_heads_par.py --dry-run --fold 0 --out heads_dry/     (one fold, IO + shapes)
  python fit_heads_par.py --dry-run --merge --out heads_dry/      (needs oof_fold0..4.npz)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# fit_heads.py lives next to this file -- import it as a module so read_meta / read_list /
# pair_aggs6 / write_table / metrics / the FAMS & LGB_KW & MLP_KW constants are reused
# verbatim rather than re-typed (and so a change to those in fit_heads.py cannot silently
# drift out of step with this file).  fit_heads.py is never edited by this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_heads as fh  # noqa: E402

FAMS = fh.FAMS
TRAIN_FOLDS = fh.TRAIN_FOLDS
N_FOLDS = fh.N_FOLDS
PCA_K = fh.PCA_K
AGG_NAMES = fh.AGG_NAMES
LR_C = fh.LR_C
LGB_KW = fh.LGB_KW
MLP_KW = fh.MLP_KW


# --------------------------------------------------------------------------- the head bank
def fit_bank(Xtr, ytr, Xte_list, seeds, fast=False, n_jobs=-1):
    """Byte-for-byte fit_heads.fit_bank, except LightGBM's n_jobs is a parameter instead of
    the hard-coded -1 (so N parallel fold processes can each be told to use a share of the
    box's cores rather than all of them)."""
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
        m = LGBMClassifier(random_state=int(s), n_jobs=n_jobs, **lgb_kw).fit(Xtr, ytr)
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


# --------------------------------------------------------------------------- data loading
def load_data(a):
    """Exact mirror of fit_heads.main's loading section (meta parsing, dry-run pair subset,
    real-vs-synthetic embedding matrix, scope57/subset index arrays).  Every one of the N
    processes (5 fold workers + the merge step) calls this independently and gets identical
    arrays: the meta file, the uid ordering and (in --dry-run) the RNG seed (268) that draws
    the synthetic embeddings are all fixed inputs, so there is nothing to hand between
    processes except the OOF results themselves."""
    seeds = [int(x) for x in a.seeds.split(',') if x.strip()]
    meta = fh.read_meta(a.meta)
    m_uid = [r['uid'] for r in meta]
    scope57 = fh.read_list(a.scope57)
    subset = fh.read_list(a.subset)

    fold = np.array([r['fold'] for r in meta])
    listed = np.array([r['listed'] for r in meta])
    label_pair = np.array([r['label_pair'] for r in meta])
    in_pool = np.array([r['in_pool'] for r in meta], dtype=bool)
    extra = np.array([r['extra_listed'] for r in meta], dtype=bool)
    src = np.array([r['source_class'] for r in meta])
    fam = np.array([r['family'] for r in meta])
    puid = np.array([r['pair_uid'] for r in meta])

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
        e_uid = fh.read_list(a.uids)
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

    return dict(seeds=seeds, m_uid=m_uid, scope57=scope57, subset=subset, fold=fold,
                listed=listed, label_pair=label_pair, in_pool=in_pool, extra=extra, src=src,
                fam=fam, puid=puid, X=X, synth=synth, i57=i57, isub=isub, uidx=uidx)


# --------------------------------------------------------------------------- --fold F
def run_fold(a, d):
    """Compute ONLY OOF fold a.fold: same train/test row selection as fit_heads.py's OOF
    loop body, same fit_bank call (with --n-jobs), saved to {out}/oof_fold{F}.npz."""
    f = a.fold
    assert 0 <= f < N_FOLDS, f'--fold must be in [0, {N_FOLDS})'
    fold, in_pool, listed, X, m_uid = d['fold'], d['in_pool'], d['listed'], d['X'], d['m_uid']
    pool = np.flatnonzero(in_pool)
    te = pool[fold[pool] == f]
    trf = pool[fold[pool] != f]
    out_path = os.path.join(a.out, f'oof_fold{f}.npz')
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if len(te) == 0 or listed[trf].sum() == 0:
        # identical skip condition to fit_heads.py's "if len(te) == 0 or listed[trf].sum()
        # == 0: continue" -- this fold contributes no OOF rows (only possible in --dry-run
        # on a tiny pair subset).  Still write a (empty) file so --merge has all 5 to read.
        print(f'  H1 OOF fold {f}: SKIPPED (test {len(te)}, train positives '
              f'{int(listed[trf].sum()) if len(trf) else 0})', flush=True)
        np.savez(out_path, uid=np.array([], dtype=str),
                 p=np.array([]), p_lr=np.array([]), p_lgb=np.array([]), p_mlp=np.array([]),
                 fold=np.array([], dtype=np.int64), train_n=np.array([len(trf)]),
                 test_n=np.array([0]), seconds=np.array([time.time() - t0]))
        return
    o, = fit_bank(X[trf], listed[trf], [X[te]], d['seeds'], fast=a.dry, n_jobs=a.n_jobs)
    dt = time.time() - t0
    print(f'  H1 OOF fold {f}: train {len(trf)} / test {len(te)}  {dt:.0f}s', flush=True)
    np.savez(out_path,
             uid=np.array([m_uid[i] for i in te]), p=o['p'], p_lr=o['p_lr'], p_lgb=o['p_lgb'],
             p_mlp=o['p_mlp'], fold=fold[te], train_n=np.array([len(trf)]),
             test_n=np.array([len(te)]), seconds=np.array([dt]))


# --------------------------------------------------------------------------- H1 held-out
def fit_heldout(a, d):
    """Fit the held-out bank exactly as fit_heads.py's H1 section does, returning
    (out57, outsub, seconds).  Used for a plain (non-parallel) run and by --merge when
    --skip-heldout is NOT given."""
    fold, in_pool, extra, listed, X = d['fold'], d['in_pool'], d['extra'], d['listed'], d['X']
    i57, isub = d['i57'], d['isub']
    t = time.time()
    tr = (in_pool | extra) & np.isin(fold, TRAIN_FOLDS)
    assert tr.sum() > 0 and listed[tr].sum() > 0, 'no positive training row'
    assert not np.isin(fold[i57], TRAIN_FOLDS).any() or a.dry, \
        'a scope57 row sits in a training fold - the held-out read would be circular'
    out57, outsub = fit_bank(X[tr], listed[tr], [X[i57], X[isub]], d['seeds'], fast=a.dry,
                              n_jobs=a.n_jobs)
    return out57, outsub, round(time.time() - t, 1), int(tr.sum()), int(listed[tr].sum())


def write_h1_scores_file(a, d, name, ii, o):
    """Identical to fit_heads.py's h1_{name}_scores.jsonl writer."""
    m_uid = d['m_uid']
    path = os.path.join(a.out, f'h1_{name}_scores.jsonl')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh_:
        for j, u in enumerate([m_uid[k] for k in ii]):
            fh_.write(json.dumps({'uid': u, 'k': 1,
                                  'mean_conf': round(100.0 * float(o['p'][j]), 4),
                                  'p_lr': round(float(o['p_lr'][j]), 6),
                                  'p_lgb': round(float(o['p_lgb'][j]), 6),
                                  'p_mlp': round(float(o['p_mlp'][j]), 6)}) + '\n')
    return path


def load_h1_scores_file(a, d, name, uid_order):
    """--skip-heldout: rebuild an o-like dict {'p','p_lr','p_lgb','p_mlp'} (in uid_order,
    which is the same order fit_heads.py wrote the file in) from the existing
    {out}/h1_{name}_scores.jsonl.  mean_conf is 100*p rounded to 4dp, so p is recovered as
    mean_conf/100 (round-trips to within 1e-6, i.e. the file's own precision)."""
    path = os.path.join(a.out, f'h1_{name}_scores.jsonl')
    recs = {}
    with open(path, encoding='utf-8') as fh_:
        for ln in fh_:
            if ln.strip():
                r = json.loads(ln)
                recs[r['uid']] = r
    missing = [u for u in uid_order if u not in recs]
    assert not missing, f'{len(missing)} {name} uids missing from existing {path}'
    return {
        'p': np.array([recs[u]['mean_conf'] / 100.0 for u in uid_order]),
        'p_lr': np.array([recs[u]['p_lr'] for u in uid_order]),
        'p_lgb': np.array([recs[u]['p_lgb'] for u in uid_order]),
        'p_mlp': np.array([recs[u]['p_mlp'] for u in uid_order]),
    }


def h1_block(d, ii, o):
    """Identical to fit_heads.py's per-head-block metrics (overall / by_head /
    by_source_class / by_family / conf)."""
    listed, src, fam = d['listed'], d['src'], d['fam']
    y = (listed[ii] == 1).astype(int)
    blk = {'overall': fh.metrics(y, o['p']),
           'by_head': {k: fh.metrics(y, o[k]) for k in ('p_lr', 'p_lgb', 'p_mlp')},
           'by_source_class': {}, 'by_family': {},
           'conf': {'mean': round(float(100 * o['p'].mean()), 3),
                    'min': round(float(100 * o['p'].min()), 3),
                    'max': round(float(100 * o['p'].max()), 3)}}
    for c in sorted(set(src[ii].tolist())):
        m = src[ii] == c
        blk['by_source_class'][c] = fh.metrics(y[m], o['p'][m])
    for f in sorted(set(fam[ii].tolist())):
        m = fam[ii] == f
        blk['by_family'][f] = fh.metrics(y[m], o['p'][m])
    return blk


# --------------------------------------------------------------------------- H2 / H3 (copied
# verbatim from fit_heads.main; they only consume p_oof / X / the meta arrays, never
# fit_bank, so they are cheap enough to just re-run once, in the merge process)
def run_h2(a, d, p_oof):
    fold, in_pool, label_pair, puid, X = d['fold'], d['in_pool'], d['label_pair'], d['puid'], d['X']
    m_uid, seeds = d['m_uid'], d['seeds']
    pool = np.flatnonzero(in_pool)
    got = ~np.isnan(p_oof)
    t = time.time()
    clean = got & in_pool & (label_pair == 0)
    q99 = float(np.quantile(p_oof[clean], .99))
    q999 = float(np.quantile(p_oof[clean], .999))
    h2_thresholds = {'clean_q99': round(q99, 6), 'clean_q999': round(q999, 6),
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
    h2_pca = {'k': PCA_K, 'fitted_per_fold_on_other_folds': True, 'folds': pca_info}

    zgot = in_pool & ~np.isnan(Z).any(axis=1)
    np.savez(os.path.join(a.out, 'emb_pca32.npz'),
             uid=np.array([m_uid[i] for i in np.flatnonzero(zgot)]),
             z=Z[zgot].astype(np.float16))
    h2_pca['n_saved_emb_pca32'] = int(zgot.sum())
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
        rows.append([pu, int(fold[ii[0]])] + [float(x) for x in fh.pair_aggs6(pv, q99, q999)]
                    + [float(x) for x in zz.mean(0)] + [float(x) for x in zz.max(0)])
    files = fh.write_table(os.path.join(a.out, 'pair_emb_features'), cols, rows)
    h2_seconds = round(time.time() - t, 1)
    h2 = {'n_pairs': len(rows), 'n_cols': len(cols) - 2, 'files': files, 'columns': cols}
    print(f'H2: {len(rows)} pairs x {len(cols) - 2} columns -> {files}', flush=True)
    return h2_thresholds, h2_pca, h2, h2_seconds


def run_h3(a, d):
    in_pool, label_pair, puid, fam, fold, X = (d['in_pool'], d['label_pair'], d['puid'],
                                               d['fam'], d['fold'], d['X'])
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
                  newline='\n') as fh_:
            for j, pu in enumerate(pp):
                if not ok[j]:
                    continue
                fh_.write(json.dumps({'pair_uid': pu, 'fold': int(pfold[j]),
                                     'probs': {FAMS[c]: round(float(P[j, c]), 6)
                                               for c in range(len(FAMS))},
                                     'argmax': FAMS[int(pred[j])],
                                     'true_family': pf[j]}) + '\n')
    else:
        h3['note'] = 'too few positive pairs to cross-validate'
    print('H3:', json.dumps(h3.get('accuracy', h3)), flush=True)
    return h3, round(time.time() - t, 1)


# --------------------------------------------------------------------------- summary skeleton
def base_summary(a, d):
    m_uid, X, puid, in_pool, extra, listed = (d['m_uid'], d['X'], d['puid'], d['in_pool'],
                                              d['extra'], d['listed'])
    return {'tag': a.tag, 'DRY_RUN': bool(a.dry), 'synthetic_embeddings': bool(d['synth']),
            'NOTE': 'nothing in gpu/ has ever run on a GPU; this file records one run of '
                    'fit_heads_par.py only',
            'emb': {'path': a.emb, 'n_rows': int(X.shape[0]), 'dim': int(X.shape[1])},
            'universe': {'n_rows': len(m_uid), 'n_pairs': int(len(set(puid.tolist()))),
                        'n_in_pool': int(in_pool.sum()), 'n_extra_listed': int(extra.sum()),
                        'n_listed': int((listed == 1).sum())},
            'heads': {'lr': {'C': LR_C}, 'lgb': LGB_KW, 'mlp': MLP_KW, 'seeds': d['seeds']}}


def finish_and_write(a, d, summary, timings, out57, outsub, p_oof, oof_parts, T0):
    """Shared tail: H1 held-out blocks + h1_oof.npz + H2 + H3 + summary.json.  Used by both
    the plain (non-parallel) run and --merge."""
    m_uid, fold, in_pool, listed = d['m_uid'], d['fold'], d['in_pool'], d['listed']
    pool = np.flatnonzero(in_pool)

    for name, ii, o in (('scope57', d['i57'], out57), ('subset', d['isub'], outsub)):
        blk = h1_block(d, ii, o)
        summary[f'h1_{name}'] = blk
        print(f'H1 {name}: {json.dumps(blk["overall"])}', flush=True)

    got = ~np.isnan(p_oof)
    assert got[pool].all() or a.dry, 'some in_pool row has no OOF probability'
    np.savez(os.path.join(a.out, 'h1_oof.npz'),
             uid=np.array([m_uid[i] for i in np.flatnonzero(got)]),
             p_oof=p_oof[got], fold=fold[got],
             **{k: v[got] for k, v in oof_parts.items()})
    summary['h1_oof'] = {'n': int(got.sum()),
                         'overall': fh.metrics((listed[got] == 1).astype(int), p_oof[got])}
    print('H1 OOF:', json.dumps(summary['h1_oof']['overall']), flush=True)

    h2_thresholds, h2_pca, h2, h2_seconds = run_h2(a, d, p_oof)
    summary['h2_thresholds'] = h2_thresholds
    summary['h2_pca'] = h2_pca
    summary['h2'] = h2
    timings['h2_seconds'] = h2_seconds

    h3, h3_seconds = run_h3(a, d)
    summary['h3_family'] = h3
    timings['h3_seconds'] = h3_seconds

    timings['total_seconds'] = round(time.time() - T0, 1)
    summary['timings'] = timings
    with open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8') as fh_:
        json.dump(summary, fh_, indent=1)
    print('wrote', os.path.join(a.out, 'summary.json'), 'in', timings['total_seconds'], 's')


# --------------------------------------------------------------------------- --merge
def run_merge(a, d):
    T0 = time.time()
    summary = base_summary(a, d)
    timings = {'parallel_folds': True}

    fold, in_pool, extra, listed = d['fold'], d['in_pool'], d['extra'], d['listed']
    tr = (in_pool | extra) & np.isin(fold, TRAIN_FOLDS)
    summary['h1_train'] = {'n_rows': int(tr.sum()), 'n_listed': int(listed[tr].sum()),
                           'folds': list(TRAIN_FOLDS)}

    if a.skip_heldout:
        out57 = load_h1_scores_file(a, d, 'scope57', d['scope57'])
        outsub = load_h1_scores_file(a, d, 'subset', d['subset'])
        print('--skip-heldout: h1_scope57/h1_subset rebuilt from existing score files',
              flush=True)
    else:
        out57, outsub, secs, _n_rows, _n_listed = fit_heldout(a, d)
        timings['h1_heads_seconds'] = secs  # h1_train above already matches (same tr formula)
        write_h1_scores_file(a, d, 'scope57', d['i57'], out57)
        write_h1_scores_file(a, d, 'subset', d['isub'], outsub)

    m_uid, uidx = d['m_uid'], d['uidx']
    p_oof = np.full(len(m_uid), np.nan)
    oof_parts = {k: np.full(len(m_uid), np.nan) for k in ('p_lr', 'p_lgb', 'p_mlp')}
    fold_seconds = {}
    fold_rows = {}
    for f in range(N_FOLDS):
        path = os.path.join(a.out, f'oof_fold{f}.npz')
        assert os.path.exists(path), f'missing {path} - run --fold {f} first'
        z = np.load(path, allow_pickle=True)
        uids_f = z['uid']
        if uids_f.size == 0:
            fold_rows[str(f)] = 0
            continue
        idx = np.array([uidx[u] for u in uids_f])
        # cross-check against the fold array in meta -- catches a stale/mismatched npz
        assert np.array_equal(fold[idx], z['fold']), f'oof_fold{f}.npz fold mismatch'
        p_oof[idx] = z['p']
        for k in oof_parts:
            oof_parts[k][idx] = z[k]
        fold_rows[str(f)] = int(uids_f.size)
        if 'seconds' in z.files:
            fold_seconds[str(f)] = float(z['seconds'][0])
    if fold_seconds:
        timings['h1_oof_fold_seconds'] = {k: round(v, 1) for k, v in fold_seconds.items()}
        timings['h1_oof_seconds'] = round(sum(fold_seconds.values()), 1)
    timings['h1_oof_fold_rows'] = fold_rows

    finish_and_write(a, d, summary, timings, out57, outsub, p_oof, oof_parts, T0)


# --------------------------------------------------------------------------- plain (non-split)
def run_plain(a, d):
    """Neither --fold nor --merge: the whole fit_heads.py pipeline in one process (all 5 OOF
    folds run sequentially here), just with --n-jobs threaded through fit_bank.  Not needed
    for the parallel launch, but keeps this script a drop-in replacement for fit_heads.py."""
    T0 = time.time()
    summary = base_summary(a, d)
    timings = {}

    out57, outsub, secs, n_rows, n_listed = fit_heldout(a, d)
    timings['h1_heads_seconds'] = secs
    summary['h1_train'] = {'n_rows': n_rows, 'n_listed': n_listed, 'folds': list(TRAIN_FOLDS)}
    write_h1_scores_file(a, d, 'scope57', d['i57'], out57)
    write_h1_scores_file(a, d, 'subset', d['isub'], outsub)

    fold, in_pool, listed, X, m_uid = d['fold'], d['in_pool'], d['listed'], d['X'], d['m_uid']
    pool = np.flatnonzero(in_pool)
    p_oof = np.full(len(m_uid), np.nan)
    oof_parts = {k: np.full(len(m_uid), np.nan) for k in ('p_lr', 'p_lgb', 'p_mlp')}
    t = time.time()
    for f in range(N_FOLDS):
        te = pool[fold[pool] == f]
        trf = pool[fold[pool] != f]
        if len(te) == 0 or listed[trf].sum() == 0:
            continue
        o, = fit_bank(X[trf], listed[trf], [X[te]], d['seeds'], fast=a.dry, n_jobs=a.n_jobs)
        p_oof[te] = o['p']
        for k in oof_parts:
            oof_parts[k][te] = o[k]
        print(f'  H1 OOF fold {f}: train {len(trf)} / test {len(te)}  '
              f'{time.time() - t:.0f}s', flush=True)
    timings['h1_oof_seconds'] = round(time.time() - t, 1)

    finish_and_write(a, d, summary, timings, out57, outsub, p_oof, oof_parts, T0)


# --------------------------------------------------------------------------- main
def build_argparser():
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
    ap.add_argument('--fold', type=int, default=None,
                     help='compute only OOF fold F (0..N_FOLDS-1), write oof_fold{F}.npz, '
                          'and exit -- for running the 5 folds as separate processes')
    ap.add_argument('--n-jobs', type=int, default=-1, dest='n_jobs',
                     help='n_jobs passed to LightGBM (fit_heads.py hard-codes -1); set this '
                          'to a share of the box\'s cores when running several folds at once')
    ap.add_argument('--merge', action='store_true',
                     help='read the 5 oof_fold*.npz files this script wrote and produce the '
                          'same h1_oof.npz / emb_pca32.npz / pair_emb_features.* / '
                          'h3_family_oof.jsonl / summary.json fit_heads.py would have')
    ap.add_argument('--skip-heldout', action='store_true', dest='skip_heldout',
                     help='(only with --merge) do not refit the held-out bank; rebuild the '
                          'h1_scope57/h1_subset summary blocks from the existing '
                          '{out}/h1_scope57_scores.jsonl and h1_subset_scores.jsonl instead')
    return ap


def main(argv=None) -> None:
    ap = build_argparser()
    a = ap.parse_args(argv)
    assert not (a.fold is not None and a.merge), '--fold and --merge are mutually exclusive'
    assert a.skip_heldout is False or a.merge, '--skip-heldout only makes sense with --merge'
    os.makedirs(a.out, exist_ok=True)

    d = load_data(a)

    if a.fold is not None:
        run_fold(a, d)
    elif a.merge:
        run_merge(a, d)
    else:
        run_plain(a, d)


if __name__ == '__main__':
    main()
