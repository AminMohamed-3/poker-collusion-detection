"""STEP194b: decoded audit measurement of the causal fold features, at the LP original-pool level.

Phase A measured stage-1 quality only: out-of-fold network AUC on the development frame rose from
0.8944652667 to 0.8996402697, +0.005175, inside the registered band. AUC is not the competition's
unit, so this step takes the same change through the production chain the .90821 file is built on
and reads the DECODED per-query AP@5 on the training-only 162-query audit.

The chain replicated here is step165b's at u = 0 (the original pool, no view averaging):
  X -> joint network, 250 fits (5 outer x 10 seeds + 5 x 4 inner x 10 seeds)
    -> 7 aligned outputs EA (inner, for training) and EB (outer, for scoring)
    -> per-family LightGBM stage 2 on [S* columns, EA]
    -> typed decoder (T1 off for CI)
    -> per-query AP@5

The new columns enter BOTH stages: `struct` feeds the network through S165.matrices, and `shared`
feeds stage 2 through colsf. That is deliberate and is what a production build would do.

Registration outputs_step194_20260916/registration194.json (amendment_2 records this level).
Audit only. No evaluation pair, no held fold, never uploads.

Usage: python step194b_C_causal_fold_audit.py REGSHA
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
from evidence_aug import step165_C_e2e_audit as S165

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step194_20260916'
REG = OUT / 'registration194.json'
FEAT = OUT / 'causal_fold_features.parquet'
NEW = ['pf_others_fold_after_pair_agg', 'pf_others_fold_before_pair_agg',
       'pf_pair_aggresses_preflop', 'pf_others_call_after_pair_agg']
K, FAMS, CI = S165.K, S165.FAMS, S165.CI


def lp_score(D, tag):
    """One arm: 250 network fits, stage-2 LP, original-pool score vector."""
    t0 = time.perf_counter()
    X, C = S165.matrices(D)
    d, Q = D['d'], D['Q']
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'],
                 d['relevance'].to_numpy(), pairs, tables)
    jobs = []
    for k, (tr, te) in enumerate(outer):
        for s in S165.SEEDS_INNER_OUTER:
            jobs.append((('outer', k), 'joint', s, rowmask(tr), rowmask(te)))
        for j, (itr, ite) in enumerate(GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])):
            for s in S165.SEEDS_INNER:
                jobs.append((('inner', k, j), 'joint', s, rowmask(tr[itr]), rowmask(tr[ite])))
    print(f'[{tag}] X {X.shape}  {len(jobs)} network fits', flush=True)
    with Pool(14, initializer=S165.pool_init, initargs=init_args) as pool:
        out = pool.map(S165.fit_job, jobs, chunksize=1)
    print(f'[{tag}] network refits done {round(time.perf_counter()-t0)}s', flush=True)

    nq = len(Q)
    AF_te, AF_in = {k: [] for k in range(5)}, {k: {} for k in range(5)}
    for jtag, m, s, rows, tq, p, S, trace, sec in out:
        if jtag[0] == 'outer':
            AF_te[jtag[1]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
        else:
            AF_in[jtag[1]].setdefault(jtag[2], []).append((tq, S165.aligned_features(D, rows, tq, S, p)))
    qpairs = Q['pair_id'].to_numpy()
    folds_q, EA, EB = [], {}, np.full((nq, len(S165.NN_COLS)), np.nan)
    for k, (tr, te) in enumerate(outer):
        trq = np.flatnonzero(np.isin(qpairs, upairs[tr]))
        teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
        ea = np.full((nq, len(S165.NN_COLS)), np.nan)
        for j, lst in AF_in[k].items():
            ea[lst[0][0]] = np.mean([a for _, a in lst], axis=0)
        eb = np.full((nq, len(S165.NN_COLS)), np.nan)
        eb[AF_te[k][0][0]] = np.mean([a for _, a in AF_te[k]], axis=0)
        assert np.isfinite(ea[trq]).all() and np.isfinite(eb[teq]).all()
        folds_q.append((trq, teq))
        EA[k] = ea[trq]
        EB[teq] = eb[teq]
    assert np.isfinite(EB).all()

    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    colsf = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    ML = {}
    for f in FAMS:
        z = Q.select(colsf[f]).to_numpy().astype(float)
        for k, (tr, te) in enumerate(folds_q):
            a = tr[isf[f][tr]]
            ML[(f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(
                np.column_stack([z[a], EA[k][isf[f][tr]]]), y[a])
    p = D['SS'].copy()
    for f in FAMS:
        z = Q.select(colsf[f]).to_numpy().astype(float)
        for k, (tr, te) in enumerate(folds_q):
            b = te[isf[f][te]]
            if not len(b):
                continue
            p[D['QIX'][b]] = ML[(f, k)].predict_proba(np.column_stack([z[b], EB[b]]))[:, 1] + 10
    print(f'[{tag}] LP scored {round(time.perf_counter()-t0)}s', flush=True)
    return p


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert reg['new_columns'] == NEW
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    t0 = time.perf_counter()

    feat = pl.read_parquet(FEAT).select(K + NEW)
    scores = {}
    for arm in ('control', 'with_new'):
        D = S165.load()
        d0 = D['d'].select(K)
        q0 = D['Q'].select(K)
        cov = d0.join(feat, on=K, how='inner').height
        assert cov == d0.height, f'feature parquet misses {d0.height - cov} audit rows'
        if arm == 'with_new':
            D['d'] = D['d'].join(feat, on=K, how='left', validate='1:1', maintain_order='left')
            D['Q'] = D['Q'].join(feat, on=K, how='left', validate='1:1', maintain_order='left')
            assert D['d'].select(K).equals(d0) and D['Q'].select(K).equals(q0), 'join reordered rows'
            assert D['d'].select(NEW).null_count().to_numpy().sum() == 0
            D['struct'] = D['struct'] + NEW
            D['shared'] = D['shared'] + NEW
        scores[arm] = lp_score(D, arm)
        if arm == 'control':
            Dref = D

    d = Dref['d']
    dec = Dref['dec']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    M = {}
    for arm, s in scores.items():
        _, m = summarize(key.with_columns(pl.Series('score', dec(s))), truths, fams)
        M[arm] = np.array([m[q] for q in qids])
    delta = M['with_new'] - M['control']
    hv = np.array([half(ptab[q]) for q in qids])
    R = lambda x: round(float(x), 6)
    res = {'registration_sha256': reg_sha, 'level': 'LP original pool (u=0), 162-query audit',
           'n_queries': len(qids),
           'map5_control': R(M['control'].mean()), 'map5_with_new': R(M['with_new'].mean()),
           'decoded_delta': R(delta.mean()),
           'half0_delta': R(delta[hv == 0].mean()), 'half1_delta': R(delta[hv == 1].mean()),
           'half0_n': int((hv == 0).sum()), 'half1_n': int((hv == 1).sum()),
           'positive_on_both_halves': bool(delta[hv == 0].mean() > 0 and delta[hv == 1].mean() > 0),
           'queries_improved': int((delta > 0).sum()), 'queries_worsened': int((delta < 0).sum()),
           'per_family': {}, 'seconds': round(time.perf_counter() - t0, 1)}
    for f in FAMS:
        sel = np.array([fams[q] == f for q in qids])
        res['per_family'][f] = {'n': int(sel.sum()), 'delta': R(delta[sel].mean())}
    res['G_equals_delta_here'] = ('no configuration is SELECTED in this step -- there are two arms '
                                 'fixed in advance -- so there is no selection shrinkage and the '
                                 'half floor is the only gate that applies')
    res['bar'] = {'corrected_build_bar_decoded': 0.0073,
                  'clears_bar': bool(delta.mean() >= 0.0073)}
    (OUT / 'results194b_audit.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
