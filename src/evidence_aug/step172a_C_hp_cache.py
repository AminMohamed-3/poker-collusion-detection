"""Step 172a (Claude): feature cache for the STEP172 stage-2 hyperparameter search (user request 2026-09-15: hyperparameter search for
all components). Nothing is selected here. It rebuilds, bit for bit, the inputs every stage-2 configuration needs on the training-only
162-query audit:
  * the 7 aligned stage-1 outputs of the STEP165 joint network (production LP columns) on three table partitions:
    P0 = STEP165/165b (sklearn GroupKFold(5) over pairs by table, inner GroupKFold(4)); P1651 / P1652 = STEP165c (tables shuffled and dealt,
    inner seed 10*partition+fold). Test rows = mean of 10 outer seeds, training rows = inner 4-fold OOF x 10 seeds (as STEP165-165c);
  * the original top-20 pool plus the same 40 label-free pool views as v1 (20 thinned 1241/2241, 20 padded 1242/2242, u = .5), as one
    float64 array (41 x 3240 x union columns) so a configuration scores views without rebuilding frames.
Gates (all exact, max abs diff 0.0): S*g and LP with ac.CLF reproduce audit_scores165.parquet (P0) and audit_scores165c.parquet
(P1651/P1652); v1LP on P0 computed from the cache reproduces audit_scores165b.parquet. Outputs experiments/hp_search_20260915/cache/.
Usage: run REGSHA"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '1'
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug import step165b_C_lp_v1_level as S165B
from evidence_aug import step165c_C_lp_partition_replicate as S165C

CV = POKER / 'Claude Version'
E165 = CV / 'experiments/e2e_evidence_C_20260915'
EXP = CV / 'experiments/hp_search_20260915'
CACHE = EXP / 'cache'
REG = EXP / 'registration172.json'
K, FAMS, CI, NN = S165.K, S165.FAMS, S165.CI, S165.NN_COLS
PARTS = ['P0', 'P1651', 'P1652']


def partition(name, upairs, ptab):
    if name == 'P0':
        outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
        inner = [list(GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])) for tr, _ in outer]
    else:
        ps = int(name[1:])
        outer = S165C.table_folds(ptab, ps)
        inner = [S165C.inner_folds(ptab[tr], ps * 10 + k) for k, (tr, _) in enumerate(outer)]
    return outer, inner


def union_columns(D):
    U = D['shared'] + D['rcols'] + D['cs_cols'] + D['pf_cols']
    assert len(set(U)) == len(U)
    idx = {f: np.array([U.index(c) for c in D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols']]) for f in FAMS}
    return U, idx


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[2]
    assert S165.sha(REG) == reg_sha
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert S165.sha(Path(__file__)) == reg['code']['cache_runner_sha256']
    CACHE.mkdir(parents=True, exist_ok=True)
    assert not (CACHE / 'receipt172a.json').exists(), 'preserve a completed cache'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    D = S165.load()
    X, C = S165.matrices(D)
    d, Q = D['d'], D['Q']
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True); ptab = tables[pfirst]
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'], d['relevance'].to_numpy(), pairs, tables)
    P = {n: partition(n, upairs, ptab) for n in PARTS}
    jobs = []
    for n, (outer, inner) in P.items():
        for k, (tr, te) in enumerate(outer):
            for s in S165.SEEDS_INNER_OUTER:
                jobs.append((('outer', n, k), 'joint', s, rowmask(tr), rowmask(te)))
            for j, (itr, ite) in enumerate(inner[k]):
                for s in S165.SEEDS_INNER:
                    jobs.append((('inner', n, k, j), 'joint', s, rowmask(tr[itr]), rowmask(tr[ite])))
    print('network jobs', len(jobs), flush=True)
    with Pool(12, initializer=S165.pool_init, initargs=init_args) as pool:
        out = pool.map(S165.fit_job, jobs, chunksize=1)
    print('network fits done', round(time.perf_counter() - t0), 's', flush=True)
    nq = len(Q); qpairs = Q['pair_id'].to_numpy()
    s165 = pl.read_parquet(E165 / 'audit_scores165.parquet').sort(K); s165c = pl.read_parquet(E165 / 'audit_scores165c.parquet').sort(K)
    s165b = pl.read_parquet(E165 / 'audit_scores165b.parquet').sort(K)
    for s_ in (s165, s165b, s165c):
        assert s_.select(K).equals(d.select(K))
    ref = {'P0': (s165['S_star_g'].to_numpy(), s165['LP'].to_numpy())}
    for ps in (1651, 1652):
        ref[f'P{ps}'] = (s165c[f'S_star_g_{ps}'].to_numpy(), s165c[f'LP_{ps}'].to_numpy())
    gates, meta = {}, {'partitions': {}}
    for n, (outer, inner) in P.items():
        AF_te = {k: [] for k in range(5)}; AF_in = {k: {} for k in range(5)}
        for tag, m, s, rows, tq, p, S, trace, sec in out:
            if tag[1] != n:
                continue
            if tag[0] == 'outer':
                AF_te[tag[2]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
            else:
                AF_in[tag[2]].setdefault(tag[3], []).append((tq, S165.aligned_features(D, rows, tq, S, p)))
        folds_q, extra, EA, EB = [], {}, np.full((5, nq, len(NN)), np.nan), np.full((nq, len(NN)), np.nan)
        for k, (tr, te) in enumerate(outer):
            trq = np.flatnonzero(np.isin(qpairs, upairs[tr])); teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
            ea = np.full((nq, len(NN)), np.nan)
            for j, lst in AF_in[k].items():
                assert len(lst) == len(S165.SEEDS_INNER)
                ea[lst[0][0]] = np.mean([a for _, a in lst], axis=0)
            eb = np.full((nq, len(NN)), np.nan)
            assert len(AF_te[k]) == len(S165.SEEDS_INNER_OUTER)
            eb[AF_te[k][0][0]] = np.mean([a for _, a in AF_te[k]], axis=0)
            assert np.isfinite(ea[trq]).all() and np.isfinite(eb[teq]).all()
            folds_q.append((trq, teq)); extra[k] = (ea[trq], eb[teq]); EA[k] = ea; EB[teq] = eb[teq]
        assert np.isfinite(EB).all()
        SG = S165.full_scores(D, S165.lgb_stage2(D, folds_q)); LP = S165.full_scores(D, S165.lgb_stage2(D, folds_q, extra))
        gates[n] = {'S_star_g': float(np.abs(SG - ref[n][0]).max()), 'LP': float(np.abs(LP - ref[n][1]).max())}
        print(n, 'replay', gates[n], flush=True)
        assert gates[n]['S_star_g'] == 0.0 and gates[n]['LP'] == 0.0
        np.savez(CACHE / f'nn_{n}.npz', EA=EA, EB=EB, **{f'trq{k}': a for k, (a, _) in enumerate(folds_q)}, **{f'teq{k}': b for k, (_, b) in enumerate(folds_q)})
        meta['partitions'][n] = {'sha256': S165.sha(CACHE / f'nn_{n}.npz')}
    # view cache
    U, idx = union_columns(D)
    views, pads = S165B.build_views(D)
    V = np.zeros((41, nq, len(U)), dtype=np.float64)
    V[0] = Q.select(U).to_numpy().astype(float)
    i = 1
    for maker, seeds in ((views, (1241, 2241)), (pads, (1242, 2242))):
        for s in seeds:
            for v in maker(.5, 10, s):
                V[i] = v.select(U).to_numpy().astype(float); i += 1
        print('views', i - 1, round(time.perf_counter() - t0), 's', flush=True)
    assert i == 41
    np.save(CACHE / 'views41.npy', V)
    meta.update({'views41_sha256': S165.sha(CACHE / 'views41.npy'), 'union_columns': U, 'family_column_index': {f: idx[f].tolist() for f in FAMS},
                 'view_order': 'index 0 original pool; 1-10 thin 1241; 11-20 thin 2241; 21-30 pad 1242; 31-40 pad 2242'})
    # v1LP gate on P0 through the cache path used by step172b
    from evidence_aug import step172b_C_hp_search as S172B
    y = Q['relevance'].to_numpy().astype(int); fam = Q['behavior_family'].to_numpy()
    nz = np.load(CACHE / 'nn_P0.npz')
    full = S172B.v1_full(dict(S172B.INCUMBENT), V, idx, nz, y, fam, D['SS'], D['QIX'])
    gates['P0_v1LP_from_cache'] = float(np.abs(full - s165b['v1LP'].to_numpy()).max())
    print('v1LP replay', gates['P0_v1LP_from_cache'], flush=True)
    assert gates['P0_v1LP_from_cache'] == 0.0
    (CACHE / 'meta172a.json').write_text(json.dumps(meta), encoding='utf-8')
    assert check_protected() == prot
    rec = {'registration_sha256': reg_sha, 'gates': gates, 'network_fits': len(out), 'meta_sha256': S165.sha(CACHE / 'meta172a.json'),
           'wall_seconds': round(time.perf_counter() - t0, 1)}
    (CACHE / 'receipt172a.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print('DONE', json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
