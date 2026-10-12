"""STEP202: the null distribution of the 162-query decoded MAP@5 under numerical luck alone.

STEP201 stored its bundle as float64 where the incumbent standardizes float32 -- a numerically
IRRELEVANT difference, float64 being strictly more precise -- and the CONTROL pipeline moved by
-0.0067 decoded on identical data, folds, seeds and code. 250 averaged network fits did not wash it
out. That is almost exactly the +0.0073 build bar. This step measures how much of any audit delta is
basin selection rather than signal.

    UNPAIRED_seed_blocks  6 full 250-fit control runs on shifted seed blocks. Block 0 IS the
                          incumbent and must reproduce 0.690.
    PAIRED_stage2_seed    one fixed block, stage 2 refitted at 8 LightGBM random_states.
    PAIRED_dtype          the float64 accident itself, re-expressed as a paired draw.

Paired and unpaired nulls are different quantities and both are reported. Every read this session was
constructed PAIRED (arm and control share seeds), so the paired null is the one that governs them.

Audit only. No evaluation pair, no held fold, never uploads, produces no file.

Usage: python step202_C_noise_floor.py REGSHA
"""
import os
import sys
sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import itertools
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
OUT = CV / 'outputs_step202_20260917'
REG = OUT / 'registration202.json'
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
NBLOCK = 6
STAGE2_SEEDS = [11, 12, 13, 14, 15, 16, 17, 18]


def fit_job4(job):
    """one control fit against a named bundle. No curriculum, no augmentation, one joint phase."""
    from evidence_aug import step165_e2e_mlp as M
    tag, bundle, seed, train_pairs, pred_pairs = job
    keys = ['W1', 'b1', 'W2', 'b2', 'U', 'c', 'V1', 'e1', 'v2', 'e2']
    G = S201.GA[bundle]
    Btr, Bv, Xs, Cs = S201._split(G, seed, M, train_pairs)
    P = M.init(Btr.X.shape[1], Btr.C.shape[1], seed, Btr.X.dtype)
    M.adam_phase(P, Btr, 'joint', keys, Bv)
    pmask = np.isin(G['pid'], pred_pairs)
    rows = np.flatnonzero(pmask)
    tq = np.flatnonzero(pmask[G['QIX']])
    p, S = M.predict(P, S201._batch(M, G, pmask, Xs, Cs))
    return tag, rows, tq, p.astype(np.float64), S.astype(np.float64)


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    assert not (OUT / 'results202.json').exists(), 'preserve a completed read'
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    D = S165.load()
    d, Q = D['d'], D['Q']
    X, C = S165.matrices(D)
    meta = (D['QIX'], D['fam'], d['relevance'].to_numpy(),
            d['pair_id'].to_numpy(), d['table_id'].to_numpy())
    S201.save_bundle('nf32', (X, C) + meta, dtype=np.float32)
    S201.save_bundle('nf64', (X, C) + meta, dtype=np.float64)
    del X, C

    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))

    # block 0 is the incumbent's own seed block; 1..5 shift both ranges by 1000 each
    blocks = {}
    for b in range(NBLOCK):
        blocks['B%d_f32' % b] = ('nf32', [s + 1000 * b for s in S165.SEEDS_INNER_OUTER],
                                 [s + 1000 * b for s in S165.SEEDS_INNER])
    blocks['B0_f64'] = ('nf64', list(S165.SEEDS_INNER_OUTER), list(S165.SEEDS_INNER))
    assert blocks['B0_f32'][1] == list(S165.SEEDS_INNER_OUTER), 'block 0 is not the incumbent block'

    jobs = []
    for name, (bundle, so, si) in blocks.items():
        for k_, (tr, te) in enumerate(outer):
            for s in so:
                jobs.append(((name, 'outer', k_), bundle, s, upairs[tr], upairs[te]))
            for j, (itr, ite) in enumerate(
                    GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])):
                for s in si:
                    jobs.append(((name, 'inner', k_, j), bundle, s,
                                 upairs[tr[itr]], upairs[tr[ite]]))
    print('%d fits over %d blocks' % (len(jobs), len(blocks)), flush=True)
    with Pool(14, initializer=S201.pool_init2, initargs=({'nf32': 'nf32', 'nf64': 'nf64'},)) as pool:
        out = pool.map(fit_job4, jobs, chunksize=1)
    print('all fits done %ds' % round(time.perf_counter() - t0), flush=True)

    nq = len(Q)
    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    qpairs = Q['pair_id'].to_numpy()
    colsf = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols']
             for f in FAMS}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    Zf = {f: Q.select(colsf[f]).to_numpy().astype(float) for f in FAMS}

    def chain(name, rs):
        AF_te, AF_in = {k_: [] for k_ in range(5)}, {k_: {} for k_ in range(5)}
        for tag, rows, tq, p, S in out:
            if tag[0] != name:
                continue
            if tag[1] == 'outer':
                AF_te[tag[2]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
            else:
                AF_in[tag[2]].setdefault(tag[3], []).append(
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
        assert np.isfinite(EB).all(), name + ': EB incomplete'
        p = D['SS'].copy()
        for f in FAMS:
            z = Zf[f]
            for k_, (tr, te) in enumerate(folds_q):
                aix = tr[isf[f][tr]]
                m = lgb.LGBMClassifier(**ac.CLF, random_state=rs).fit(
                    np.column_stack([z[aix], EA[k_][isf[f][tr]]]), y[aix])
                b = te[isf[f][te]]
                if len(b):
                    p[D['QIX'][b]] = m.predict_proba(np.column_stack([z[b], EB[b]]))[:, 1] + 10
        return p

    dec = D['dec']
    truths = {k_[0]: set(g['hand_id'])
              for k_, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    kf = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    R = lambda x: round(float(x), 6)

    def mapvec(p):
        _, mm = summarize(kf.with_columns(pl.Series('score', dec(p))), truths, fams)
        return np.array([mm[q] for q in qids])

    per_q = {name: mapvec(chain(name, 11)) for name in blocks}
    for name in blocks:
        print('  %-8s map5 %.6f  %ds' % (name, per_q[name].mean(),
                                         round(time.perf_counter() - t0)), flush=True)
    s2 = {rs: mapvec(chain('B0_f32', rs)) for rs in STAGE2_SEEDS}
    print('  stage-2 seeds done %ds' % round(time.perf_counter() - t0), flush=True)

    bnames = ['B%d_f32' % b for b in range(NBLOCK)]
    bm = np.array([per_q[n].mean() for n in bnames])
    pairdiff = np.array([bm[i] - bm[j] for i, j in itertools.combinations(range(NBLOCK), 2)])
    s2m = np.array([s2[rs].mean() for rs in STAGE2_SEEDS])
    s2paired = np.array([(s2[rs] - s2[11]).mean() for rs in STAGE2_SEEDS[1:]])

    res = {'registration_sha256': reg_sha, 'level': 'LP original pool (u=0), 162-query audit',
           'n_queries': len(qids), 'n_network_fits': len(jobs),
           'block_map5': {n: R(per_q[n].mean()) for n in blocks},
           'UNPAIRED': {
               'blocks': [R(x) for x in bm], 'mean': R(bm.mean()),
               'sd': R(bm.std(ddof=1)), 'min': R(bm.min()), 'max': R(bm.max()),
               'range': R(bm.max() - bm.min()),
               'sd_relative_error_with_n_blocks': R(1 / np.sqrt(2 * (NBLOCK - 1))),
               'pairwise_diff_absmax': R(np.abs(pairdiff).max()),
               'pairwise_diff_sd': R(pairdiff.std(ddof=1))},
           'PAIRED_stage2_seed': {
               'map5': [R(x) for x in s2m], 'sd': R(s2m.std(ddof=1)),
               'paired_delta_vs_rs11_absmax': R(np.abs(s2paired).max()),
               'paired_delta_sd': R(s2paired.std(ddof=1))},
           'PAIRED_dtype': {
               'f32': R(per_q['B0_f32'].mean()), 'f64': R(per_q['B0_f64'].mean()),
               'delta': R(per_q['B0_f64'].mean() - per_q['B0_f32'].mean()),
               'note': 'same seeds, same folds, same data; only the standardisation dtype differs'}}
    res['gates'] = {'block0_is_the_incumbent': {
        'here': R(bm[0]), 'expected': 0.690, 'tolerance': 0.001,
        'passed': bool(abs(bm[0] - 0.690) <= 0.001)}}
    sd = bm.std(ddof=1)
    res['HOW_TO_READ_A_DELTA'] = {
        'unpaired_sd': R(sd),
        'bar_in_unpaired_sd': R(0.0073 / sd) if sd > 0 else None,
        'paired_stage2_sd': R(s2paired.std(ddof=1)),
        'caution': ('the UNPAIRED sd applies only when an arm is compared against a control from a '
                    'DIFFERENT fitting run. Every read this session was paired -- arm and control '
                    'share seeds inside one run -- so the paired null governs them, and the dtype '
                    'draw is the only paired network-level observation available here.')}
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results202.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
