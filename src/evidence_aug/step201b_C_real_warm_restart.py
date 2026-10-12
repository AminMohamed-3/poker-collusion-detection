"""STEP201b: the arm that separates STAGING from AUGMENTATION.

STEP201 asks whether pretraining on augmented views then finetuning on real data beats one phase on
real data. Its control C shuffles the augmented labels -- which makes C's phase 1 a CORRUPTED task,
not an unaugmented one. Arms A and D pretrain on a frame whose top-20 rows and LABELS are identical
to the real ones (only chrono_frac, log n_hands and the 136 context_ columns differ). So A > C only
establishes that a sane warm-up beats a corrupted one. It says nothing about the augmentation.

E_REAL_WARMUP_THEN_REAL is two joint phases on the REAL batch. Everything that differs between E and
STEP201's arm A is the augmentation and nothing else.

    A - E   the value of the augmented DATA, staging held constant
    E - ctl the value of STAGING alone, data held constant

Registered in registration201b BEFORE results201.json was read, including that A - E OVERRIDES A - C.

Audit only. No evaluation pair, no held fold, never uploads.

Usage: python step201b_C_real_warm_restart.py REGSHA
"""
import os
import sys
sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
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
from evidence_aug import step201_C_augmentation_curriculum as S201

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step201_20260917'
REG = OUT / 'registration201b.json'
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
ARMS = ['control', 'E_REAL_WARMUP_THEN_REAL']


def fit_job3(job):
    """STEP201's fit_job2 restricted to the real bundle: one phase, or two phases on the SAME data."""
    from evidence_aug import step165_e2e_mlp as M
    tag, arm, seed, train_pairs, pred_pairs = job
    keys = ['W1', 'b1', 'W2', 'b2', 'U', 'c', 'V1', 'e1', 'v2', 'e2']
    R = S201.GA['real']
    Btr, Bv, Xs, Cs = S201._split(R, seed, M, train_pairs)
    P = M.init(Btr.X.shape[1], Btr.C.shape[1], seed, Btr.X.dtype)
    trace = M.adam_phase(P, Btr, 'joint', keys, Bv)
    if arm == 'E_REAL_WARMUP_THEN_REAL':
        # a second phase on identical data: fresh Adam moments, fresh patience counter. A warm
        # restart. This is the whole arm.
        trace = trace + M.adam_phase(P, Btr, 'joint', keys, Bv)
    pmask = np.isin(R['pid'], pred_pairs)
    rows = np.flatnonzero(pmask)
    tq = np.flatnonzero(pmask[R['QIX']])
    Bte = S201._batch(M, R, pmask, Xs, Cs)
    p, S = M.predict(P, Bte)
    return tag, arm, seed, rows, tq, p.astype(np.float64), S.astype(np.float64), \
        [float(x) for x in trace]


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    assert not (OUT / 'results201b.json').exists(), 'preserve a completed read'
    # results201.json is the VOID float64-bundle read (it failed its control gate at 0.683308).
    # A and C are carried from the corrected float32 rerun, whose control reproduces 0.690 exactly.
    r201 = json.loads((OUT / 'results201_rerun_float32.json').read_text(encoding='utf-8'))
    assert r201['gates']['control_reproduces_0690']['passed'], 'carrying numbers from a voided read'
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    D = S165.load()
    d, Q = D['d'], D['Q']
    Xr, Cr = S165.matrices(D)
    S201.save_bundle('real', (Xr, Cr, D['QIX'], D['fam'], d['relevance'].to_numpy(),
                              d['pair_id'].to_numpy(), d['table_id'].to_numpy()))
    del Xr, Cr

    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    jobs = []
    for arm in ARMS:
        for k_, (tr, te) in enumerate(outer):
            for s in S165.SEEDS_INNER_OUTER:
                jobs.append((('outer', k_), arm, s, upairs[tr], upairs[te]))
            for j, (itr, ite) in enumerate(
                    GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])):
                for s in S165.SEEDS_INNER:
                    jobs.append((('inner', k_, j), arm, s, upairs[tr[itr]], upairs[tr[ite]]))
    print('%d fits over %d arms' % (len(jobs), len(ARMS)), flush=True)
    with Pool(14, initializer=S201.pool_init2, initargs=({'real': 'real'},)) as pool:
        out = pool.map(fit_job3, jobs, chunksize=1)
    print('all fits done %ds' % round(time.perf_counter() - t0), flush=True)

    ac = D['ac']
    nq = len(Q)
    y = Q['relevance'].to_numpy().astype(int)
    qpairs = Q['pair_id'].to_numpy()
    colsf = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols']
             for f in FAMS}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    Zf = {f: Q.select(colsf[f]).to_numpy().astype(float) for f in FAMS}
    scores, traces = {}, {}
    for arm in ARMS:
        AF_te, AF_in = {k_: [] for k_ in range(5)}, {k_: {} for k_ in range(5)}
        tr_all = []
        for tag, a_, s, rows, tq, p, S, trace in out:
            if a_ != arm:
                continue
            tr_all.append(trace)
            if tag[0] == 'outer':
                AF_te[tag[1]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
            else:
                AF_in[tag[1]].setdefault(tag[2], []).append(
                    (tq, S165.aligned_features(D, rows, tq, S, p)))
        folds_q, EA, EB = [], {}, np.full((nq, len(S165.NN_COLS)), np.nan)
        for k_, (tr, te) in enumerate(outer):
            trq = np.flatnonzero(np.isin(qpairs, upairs[tr]))
            teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
            ea = np.full((nq, len(S165.NN_COLS)), np.nan)
            for j, lst in AF_in[k_].items():
                ea[lst[0][0]] = np.mean([x for _, x in lst], axis=0)
            eb = np.full((nq, len(S165.NN_COLS)), np.nan)
            eb[AF_te[k_][0][0]] = np.mean([x for _, x in AF_te[k_]], axis=0)
            folds_q.append((trq, teq))
            EA[k_] = ea[trq]
            EB[teq] = eb[teq]
        assert np.isfinite(EB).all(), arm + ': EB incomplete'
        p = D['SS'].copy()
        for f in FAMS:
            z = Zf[f]
            for k_, (tr, te) in enumerate(folds_q):
                aix = tr[isf[f][tr]]
                m = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(
                    np.column_stack([z[aix], EA[k_][isf[f][tr]]]), y[aix])
                b = te[isf[f][te]]
                if len(b):
                    p[D['QIX'][b]] = m.predict_proba(np.column_stack([z[b], EB[b]]))[:, 1] + 10
        scores[arm] = p
        L = max(len(u) for u in tr_all)
        traces[arm] = [round(float(x), 4) for x in
                       np.mean(np.array([t for t in tr_all if len(t) == L]), axis=0)]
        print('  %-26s scored  %ds' % (arm, round(time.perf_counter() - t0)), flush=True)

    dec = D['dec']
    truths = {k_[0]: set(g['hand_id'])
              for k_, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    kf = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabq[q]) for q in qids])
    R_ = lambda x: round(float(x), 6)
    MM = {}
    for arm, s in scores.items():
        _, mm = summarize(kf.with_columns(pl.Series('score', dec(s))), truths, fams)
        MM[arm] = np.array([mm[q] for q in qids])

    delta = MM['E_REAL_WARMUP_THEN_REAL'] - MM['control']
    E_ = R_(delta.mean())
    A_ = r201['arms']['A_WARMUP_AUG_THEN_REAL']['decoded_delta']
    C_ = r201['arms']['C_SHUFFLED_AUG_WARMUP']['decoded_delta']
    res = {'registration_sha256': reg_sha,
           'level': 'LP original pool (u=0), 162-query audit',
           'map5_control': R_(MM['control'].mean()),
           'phase_traces_mean': traces,
           'E_REAL_WARMUP_THEN_REAL': {
               'map5': R_(MM['E_REAL_WARMUP_THEN_REAL'].mean()), 'decoded_delta': E_,
               'half0': R_(delta[hv == 0].mean()), 'half1': R_(delta[hv == 1].mean()),
               'positive_on_both_halves': bool(delta[hv == 0].mean() > 0
                                               and delta[hv == 1].mean() > 0),
               'per_family': {f: R_(delta[[fams[q] == f for q in qids]].mean()) for f in FAMS}},
           'carried_from_step201': {'A_decoded_delta': A_, 'C_decoded_delta': C_,
                                    'control_map5': r201['map5_control']},
           'THE_SUBTRACTION': {
               'A_minus_E': R_(A_ - E_),
               'E_minus_control': E_,
               'augmentation_carries_something': bool(A_ - E_ > 0.004),
               'reading': ('A - E is the value of the augmented DATA with staging held constant, '
                           'and it OVERRIDES A - C by the terms of registration201b.')},
           'gates': {
               'control_reproduces_0690': {
                   'here': R_(MM['control'].mean()), 'tolerance': 0.002,
                   'passed': bool(abs(MM['control'].mean() - 0.690) <= 0.002)},
               'control_matches_step201_control': {
                   'here': R_(MM['control'].mean()), 'step201': r201['map5_control'],
                   'tolerance': 0.001,
                   'passed': bool(abs(MM['control'].mean() - r201['map5_control']) <= 0.001)},
               'phase2_actually_trained': {
                   'E_phase2_best_epoch_mean': traces['E_REAL_WARMUP_THEN_REAL'][5]
                   if len(traces['E_REAL_WARMUP_THEN_REAL']) >= 6 else None,
                   'passed': bool(len(traces['E_REAL_WARMUP_THEN_REAL']) >= 6
                                  and traces['E_REAL_WARMUP_THEN_REAL'][5] > 0)}}}
    res['verdict'] = ('AUGMENTATION_CARRIES_SOMETHING' if res['THE_SUBTRACTION'][
        'augmentation_carries_something'] else
        'IT_WAS_THE_SCHEDULE' if abs(E_) > 0.004 else 'BOTH_NOTHING')
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    (OUT / 'results201b.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
