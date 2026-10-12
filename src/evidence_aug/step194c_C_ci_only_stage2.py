"""STEP194c: the causal fold columns in COORDINATED ISOLATION's stage-2 only.

STEP194b was a NO-GO on its own terms (decoded +0.000453, halves +0.008736 / -0.009156, half floor
FAILED). Its per-family split was CI +0.035736, SP -0.000638, DT -0.019907 -- the feature helps the
family it was induced from and hurts another. registration194c registers this second look, shrinks
the estimate because CI was the best of three, and discloses the multiplicity.

The joint network is untouched here, so DT and SP are unchanged BY CONSTRUCTION and their decoded
deltas must be exactly 0.0. That is the gate. Both arms share ONE network fit.

Audit only. No evaluation pair, no held fold, never uploads.

Usage: python step194c_C_ci_only_stage2.py REGSHA
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
REG = OUT / 'registration194c.json'
FEAT = OUT / 'causal_fold_features.parquet'
NEW = ['pf_others_fold_after_pair_agg', 'pf_others_fold_before_pair_agg',
       'pf_pair_aggresses_preflop', 'pf_others_call_after_pair_agg']
K, FAMS, CI = S165.K, S165.FAMS, S165.CI


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    t0 = time.perf_counter()

    D = S165.load()
    feat = pl.read_parquet(FEAT).select(K + NEW)
    q0 = D['Q'].select(K)
    D['Q'] = D['Q'].join(feat, on=K, how='left', validate='1:1', maintain_order='left')
    assert D['Q'].select(K).equals(q0), 'join reordered Q'
    assert D['Q'].select(NEW).null_count().to_numpy().sum() == 0, 'unmatched audit rows'
    # the NETWORK is untouched: struct/shared are NOT extended, so X is the incumbent matrix
    X, C = S165.matrices(D)
    assert X.shape[1] == 453, f'network input changed: {X.shape}'

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
    print(f'X {X.shape}  {len(jobs)} network fits (shared by both arms)', flush=True)
    with Pool(14, initializer=S165.pool_init, initargs=init_args) as pool:
        out = pool.map(S165.fit_job, jobs, chunksize=1)
    print(f'network refits done {round(time.perf_counter()-t0)}s', flush=True)

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
        folds_q.append((trq, teq))
        EA[k] = ea[trq]
        EB[teq] = eb[teq]
    assert np.isfinite(EB).all()

    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    base = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}

    def run(colsf):
        p = D['SS'].copy()
        for f in FAMS:
            z = Q.select(colsf[f]).to_numpy().astype(float)
            for k, (tr, te) in enumerate(folds_q):
                a = tr[isf[f][tr]]
                m = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(
                    np.column_stack([z[a], EA[k][isf[f][tr]]]), y[a])
                b = te[isf[f][te]]
                if len(b):
                    p[D['QIX'][b]] = m.predict_proba(np.column_stack([z[b], EB[b]]))[:, 1] + 10
        return p

    ci_cols = dict(base)
    ci_cols[CI] = base[CI] + NEW
    scores = {'control': run(base), 'CI_STAGE2_ONLY': run(ci_cols)}
    print(f'both arms scored {round(time.perf_counter()-t0)}s', flush=True)

    dec = D['dec']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    M = {}
    for arm, s in scores.items():
        _, m = summarize(key.with_columns(pl.Series('score', dec(s))), truths, fams)
        M[arm] = np.array([m[q] for q in qids])
    delta = M['CI_STAGE2_ONLY'] - M['control']
    hv = np.array([half(ptabq[q]) for q in qids])
    R = lambda x: round(float(x), 6)

    res = {'registration_sha256': reg_sha, 'level': 'LP original pool (u=0), 162-query audit',
           'is_a_second_look': True, 'n_queries': len(qids),
           'map5_control': R(M['control'].mean()),
           'map5_CI_stage2_only': R(M['CI_STAGE2_ONLY'].mean()),
           'decoded_delta': R(delta.mean()),
           'half0_delta': R(delta[hv == 0].mean()), 'half1_delta': R(delta[hv == 1].mean()),
           'per_family': {}, 'gates': {}, 'seconds': round(time.perf_counter() - t0, 1)}
    for f in FAMS:
        sel = np.array([fams[q] == f for q in qids])
        fd = delta[sel]
        res['per_family'][f] = {
            'n': int(sel.sum()), 'delta': R(fd.mean()),
            'half0': R(fd[hv[sel] == 0].mean()) if (hv[sel] == 0).any() else None,
            'half1': R(fd[hv[sel] == 1].mean()) if (hv[sel] == 1).any() else None}
    untouched = {f: float(np.abs(delta[[fams[q] == f for q in qids]]).max()) for f in FAMS if f != CI}
    res['gates']['DT_SP_exactly_zero'] = {'max_abs': {k: R(v) for k, v in untouched.items()},
                                          'passed': all(v == 0.0 for v in untouched.values())}
    cih = res['per_family'][CI]
    res['gates']['CI_positive_on_both_halves'] = bool(
        cih['half0'] is not None and cih['half1'] is not None and cih['half0'] > 0 and cih['half1'] > 0)
    res['gates']['control_reproduces_194b'] = {
        'step194b_control_map5': 0.69, 'here': R(M['control'].mean()),
        'tolerance': 0.002, 'passed': bool(abs(M['control'].mean() - 0.69) <= 0.002)}
    res['bar'] = {'corrected_build_bar_decoded': 0.0073, 'clears_bar': bool(delta.mean() >= 0.0073)}
    (OUT / 'results194c_ci_only.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
