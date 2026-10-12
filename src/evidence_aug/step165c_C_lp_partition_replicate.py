"""Step 165c (Claude): partition replicate of LP before any production build. STEP165/165b measured LP (S* columns + 7 aligned stage-1
network outputs in the per-family LightGBM stage 2) against S*g on ONE table partition (sklearn GroupKFold(5) over pairs by table), which
is itself -.011 (S* level) below S*'s own folds; partition noise is about +-.01-.02. This step repeats the STEP165 LP / S*g construction
exactly (same network module and config, joint mode, 10 outer seeds for test outputs, 4 inner folds x 10 seeds for training-row outputs)
on two new random table partitions (seeds 1651, 1652; tables shuffled then dealt to 5 folds). S* level, no views, training-only 162-query
audit, production typed decoder with T1 off for CI. Never uploads. Usage: run REGSHA"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '1'
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
from multiprocessing import Pool
from evidence_aug import step165_C_e2e_audit as S165

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/e2e_evidence_C_20260915'
REG = EXP / 'registration165c.json'
PART_SEEDS = [1651, 1652]
NJ = len(S165.NN_COLS)


def table_folds(ptab, seed, k=5):
    """Pair-level folds: unique tables shuffled by seed and dealt round-robin to k folds."""
    ut = np.unique(ptab)
    perm = np.random.default_rng(seed).permutation(len(ut))
    fold_of = dict(zip(ut[perm], np.arange(len(ut)) % k))
    f = np.array([fold_of[t] for t in ptab])
    return [(np.flatnonzero(f != j), np.flatnonzero(f == j)) for j in range(k)]


def inner_folds(ptab_tr, seed, k=4):
    return table_folds(ptab_tr, seed, k)


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[2]
    assert S165.sha(REG) == reg_sha
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert S165.sha(Path(__file__)) == reg['code']['runner_sha256']
    assert S165.sha(CV / 'src/evidence_aug/step165_e2e_mlp.py') == reg['code']['module_sha256']
    assert S165.sha(CV / 'src/evidence_aug/step165_C_e2e_audit.py') == reg['code']['helpers_sha256']
    dest = EXP / 'results165c.json'
    assert not dest.exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    D = S165.load()
    X, C = S165.matrices(D)
    d = D['d']
    pairs = d['pair_id'].to_numpy(); tables = d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'], d['relevance'].to_numpy(), pairs, tables)
    parts = {ps: table_folds(ptab, ps) for ps in PART_SEEDS}
    jobs = []
    for ps, outer in parts.items():
        for k, (tr, te) in enumerate(outer):
            for s in S165.SEEDS_INNER_OUTER:
                jobs.append((('outer', ps, k), 'joint', s, rowmask(tr), rowmask(te)))
            for j, (itr, ite) in enumerate(inner_folds(ptab[tr], ps * 10 + k)):
                for s in S165.SEEDS_INNER:
                    jobs.append((('inner', ps, k, j), 'joint', s, rowmask(tr[itr]), rowmask(tr[ite])))
    print('jobs', len(jobs), flush=True)
    with Pool(12, initializer=S165.pool_init, initargs=init_args) as pool:
        out = pool.map(S165.fit_job, jobs, chunksize=1)
    nq = len(D['Q']); qpairs = D['Q']['pair_id'].to_numpy()
    ac, dec = D['ac'], D['dec']
    R = lambda x: round(float(x), 6)

    def cmp(A, B):
        c = ac.evaluate(d, {'incumbent': A, 'arm': B})['comparisons']['arm']
        return {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']],
                'fam': {kk[:2]: R(x) for kk, x in c['by_family'].items()}, 'halves': {kk: R(x) for kk, x in c['halves'].items()}}
    res = {'registration_sha256': reg_sha, 'partitions': {}}
    scores = d.select(S165.K)
    for ps, outer in parts.items():
        AF_te = {k: [] for k in range(5)}; AF_in = {k: {} for k in range(5)}
        for tag, m, s, rows, tq, p, S, trace, sec in out:
            if tag[1] != ps:
                continue
            if tag[0] == 'outer':
                AF_te[tag[2]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
            else:
                AF_in[tag[2]].setdefault(tag[3], []).append((tq, S165.aligned_features(D, rows, tq, S, p)))
        folds_q, extra = [], {}
        for k, (tr, te) in enumerate(outer):
            trq = np.flatnonzero(np.isin(qpairs, upairs[tr])); teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
            ea = np.full((nq, NJ), np.nan)
            for j, lst in AF_in[k].items():
                assert len(lst) == len(S165.SEEDS_INNER)
                ea[lst[0][0]] = np.mean([a for _, a in lst], axis=0)
            eb = np.full((nq, NJ), np.nan)
            assert len(AF_te[k]) == len(S165.SEEDS_INNER_OUTER)
            eb[AF_te[k][0][0]] = np.mean([a for _, a in AF_te[k]], axis=0)
            assert np.isfinite(ea[trq]).all() and np.isfinite(eb[teq]).all()
            folds_q.append((trq, teq)); extra[k] = (ea[trq], eb[teq])
        SG = S165.full_scores(D, S165.lgb_stage2(D, folds_q))
        LP = S165.full_scores(D, S165.lgb_stage2(D, folds_q, extra))
        DSG, DLP = dec(SG), dec(LP)
        res['partitions'][str(ps)] = {'decoded_LP_vs_S_star_g': cmp(DSG, DLP), 'raw_LP_vs_S_star_g': cmp(SG, LP), 'decoded_S_star_g_vs_S_star': cmp(dec(D['SS']), DSG),
                                      'decoded_map5': {'S_star_g': R(ac.evaluate(d, {'incumbent': DSG})['sets']['incumbent']['map5']), 'LP': R(ac.evaluate(d, {'incumbent': DLP})['sets']['incumbent']['map5'])}}
        scores = scores.with_columns(pl.Series(f'S_star_g_{ps}', SG), pl.Series(f'LP_{ps}', LP))
        print(ps, json.dumps(res['partitions'][str(ps)]['decoded_LP_vs_S_star_g']), flush=True)
    new = [res['partitions'][str(ps)]['decoded_LP_vs_S_star_g']['mean'] for ps in PART_SEEDS]
    res['primary_mean_new_partitions'] = R(np.mean(new))
    res['pooled_with_step165'] = R(np.mean(new + [reg['step165_partition_increment']]))
    res['proceed_to_production'] = bool(np.mean(new) > 0)
    scores.write_parquet(EXP / 'audit_scores165c.parquet')
    res['scores_sha256'] = S165.sha(EXP / 'audit_scores165c.parquet')
    assert check_protected() == prot
    res['wall_seconds'] = round(time.perf_counter() - t0, 1)
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('PRIMARY', res['primary_mean_new_partitions'], 'pooled', res['pooled_with_step165'], 'proceed', res['proceed_to_production'], 'wall', res['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
