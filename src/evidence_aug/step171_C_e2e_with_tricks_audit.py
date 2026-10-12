"""Step 171 (Claude): the pure end-to-end evidence network of STEP165 with two production tricks it lacked (user request 2026-09-15:
"Yes try it as well"): (1) the LightGBM stage-1 ranker score as an input (oof and its within-pair rank oof_rank, in the trunk and in the
stage-2 head), and (2) test-time view averaging over the same 40 label-free pool views as v1 (20 thinned: seeds 1241/2241; 20 padded:
1242/2242; u = .5; step-124 recipe with every pool field recomputed: chrono_rank, chrono_frac, n_hands, oof_rank, 136 context_ fields).
Same network module and CONFIG as STEP165, same GroupKFold(5) table partition, 20 seeds per fold, joint (end-to-end) and two-phase modes.
Primary: decode(NE1v) vs decode(v1g) (STEP165b: S* columns with views on the same folds), ER1. Training-only 162-query audit; no held
folds, no evaluation pairs. Never uploads. Usage: run REGSHA"""
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
from sklearn.model_selection import GroupKFold
from evidence_aug import step165_C_e2e_audit as S165

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/e2e_evidence_C_20260915'
REG = EXP / 'registration171.json'
K, FAMS = S165.K, S165.FAMS
SEEDS = list(S165.SEEDS_OUTER)
MODES = ('joint', 'two_stage')


def xplus(frame, D):
    """S165.matrices X (rcols, struct, chrono_frac, log n_hands, family one-hot) + oof, oof_rank."""
    fam = frame['behavior_family'].to_numpy()
    X = np.column_stack([frame.select(D['rcols'] + D['struct']).to_numpy().astype(np.float64), frame['chrono_frac'].to_numpy(), np.log(frame['n_hands'].to_numpy())] +
                        [(fam == f).astype(float) for f in FAMS] + [frame['oof'].to_numpy(), frame['oof_rank'].to_numpy()])
    assert np.isfinite(X).all()
    return X


def cplus(top, D):
    C = np.column_stack([top.select(D['cs_cols'] + D['pf_cols']).to_numpy().astype(np.float64), top['chrono_frac'].to_numpy(), top['oof'].to_numpy(), top['oof_rank'].to_numpy()])
    assert np.isfinite(C).all()
    return C


def fit_job(job):
    """S165.fit_job with the extended matrices; also returns parameters and the training standardisation statistics."""
    from evidence_aug import step165_e2e_mlp as M
    k, mode, seed, train_mask, pred_mask = job
    G = S165.G
    rows_tr = np.flatnonzero(train_mask)
    tabs = np.unique(G['tab'][rows_tr])
    vt = np.random.default_rng([seed, 7]).choice(tabs, size=max(1, int(round(len(tabs) * M.CONFIG['val_share_tables']))), replace=False)
    val_mask = train_mask & np.isin(G['tab'], vt)
    fit_mask = train_mask & ~val_mask
    crow = np.flatnonzero(train_mask[G['QIX']])
    xs = (G['X'][rows_tr].mean(0), np.where(G['X'][rows_tr].std(0) > 1e-9, G['X'][rows_tr].std(0), 1.))
    cs = (G['C'][crow].mean(0), np.where(G['C'][crow].std(0) > 1e-9, G['C'][crow].std(0), 1.))
    Xs = S165.standardize(G['X'], rows_tr); Cs = S165.standardize(G['C'], crow)
    assert np.array_equal(Xs, np.clip((G['X'] - xs[0]) / xs[1], -6, 6)) and np.array_equal(Cs, np.clip((G['C'] - cs[0]) / cs[1], -6, 6))
    Btr, _, _ = S165.make_batch(M, G, Xs, Cs, fit_mask)
    Bv, _, _ = S165.make_batch(M, G, Xs, Cs, val_mask)
    P, trace = M.train(Btr, mode, seed, Bv)
    Bte, rows, tq = S165.make_batch(M, G, Xs, Cs, pred_mask)
    p, _ = M.predict(P, Bte)
    return k, mode, seed, P, xs, cs, tq, p.astype(np.float64), [float(x) for x in trace]


def view_frames(D, kind, seed, u, m):
    """Full-hand pool views (all rows), one per copy, sorted by (pair_id, chrono_rank); qpos = position in Q for original top-20 rows, -1 otherwise."""
    from evidence_aug.frame import context
    d, Q = D['d'], D['Q']
    ctxc = [c for c in D['rcols'] if c.startswith('context_')]; src = [c[8:] for c in ctxc]
    assert len(ctxc) == 136 and all(s in d.columns for s in src)
    hand = [c for c in D['rcols'] if not c.startswith('context_')]
    qpos = np.full(d.height, -1); qpos[D['QIX']] = np.arange(len(D['QIX']))
    POOL = d.select(list(dict.fromkeys(K + ['behavior_family', 'chrono_rank', 'oof'] + src + hand + D['struct']))).with_columns(pl.Series('qpos', qpos))
    POOL = POOL.with_columns((pl.col('qpos') >= 0).alias('top0'))
    cmax = float(POOL['chrono_rank'].max()) + 1
    top0 = POOL['top0'].to_numpy()
    for c_ in range(m):
        r = np.random.default_rng([seed, c_]).random(POOL.height)
        pick = (r < u) & ~top0
        if kind == 'thin':
            a = POOL.filter(pl.Series(~pick))
        else:
            a = pl.concat([POOL, POOL.filter(pl.Series(pick)).with_columns((pl.col('chrono_rank') + cmax * 10).alias('chrono_rank'), pl.lit(False).alias('top0'), pl.lit(-1).cast(POOL.schema['qpos']).alias('qpos'))])
        a = a.with_columns((pl.col('chrono_rank').rank('ordinal').over('pair_id') - 1).cast(pl.Float64).alias('chrono_rank'),
                           pl.len().over('pair_id').cast(pl.Float64).alias('n_hands'),
                           pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
        a = a.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac'))
        a = context(a, src).sort('pair_id', 'chrono_rank')
        t = a.filter(pl.col('top0'))
        t = t.join(Q.select(K + D['cs_cols'] + D['pf_cols']), on=K, how='left', validate='1:1', maintain_order='left')
        assert t['qpos'].n_unique() == len(D['QIX']) == t.height
        yield a, t


def predict_frame(a, t, D, fits, test_pairs):
    """Stage-2 probability for the original top-20 rows of test pairs in one view frame, per fit (returns qpos, matrix fits x rows)."""
    from evidence_aug import step165_e2e_mlp as M
    am = np.isin(a['pair_id'].to_numpy(), test_pairs)
    a2 = a.filter(pl.Series(am)); t2 = t.filter(pl.Series(np.isin(t['pair_id'].to_numpy(), test_pairs)))
    X = xplus(a2, D).astype(np.float32); C = cplus(t2, D).astype(np.float32)
    pid = a2['pair_id'].to_numpy()
    T = np.flatnonzero(a2['top0'].to_numpy())
    assert np.array_equal(pid[T], t2['pair_id'].to_numpy())
    famidx = np.array([FAMS.index(f) for f in a2['behavior_family'].to_numpy()])
    out = []
    for (P, xs, cs) in fits:
        B = M.Batch(np.clip((X - xs[0]) / xs[1], -6, 6), famidx, np.zeros(len(pid)), pid, T, np.clip((C - cs[0]) / cs[1], -6, 6))
        out.append(M.predict(P, B)[0].astype(np.float64))
    return t2['qpos'].to_numpy(), np.array(out)


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[2]
    assert S165.sha(REG) == reg_sha
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert S165.sha(Path(__file__)) == reg['code']['runner_sha256']
    for n, v in reg['inputs'].items():
        assert S165.sha(POKER / v['path']) == v['sha256'], n
    dest = EXP / 'results171.json'
    assert not dest.exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    D = S165.load()
    d, Q = D['d'], D['Q']
    X0, C0 = S165.matrices(D)
    X, C = xplus(d, D), cplus(Q, D)
    assert np.array_equal(X[:, :X0.shape[1]], X0) and np.array_equal(C[:, :C0.shape[1]], C0)
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True); ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'], d['relevance'].to_numpy(), pairs, tables)
    jobs = [(k, m, s, rowmask(tr), rowmask(te)) for k, (tr, te) in enumerate(outer) for m in MODES for s in SEEDS]
    print('data', X.shape, C.shape, 'jobs', len(jobs), flush=True)
    with Pool(10, initializer=S165.pool_init, initargs=init_args) as pool:
        out = pool.map(fit_job, jobs, chunksize=1)
    print('fits done', round(time.perf_counter() - t0), 's', flush=True)
    nq = len(Q)
    P0 = {m: np.zeros(nq) for m in MODES}; FITS = {(k, m): [] for k in range(5) for m in MODES}
    for k, m, s, P, xs, cs, tq, p, trace in out:
        P0[m][tq] += p / len(SEEDS); FITS[(k, m)].append((P, xs, cs))
    # gate 1: main-process recomputation on the original pool equals the worker predictions (float32 arithmetic)
    thin0 = next(view_frames(D, 'thin', 0, 0., 1))
    a0, t0f = thin0
    Xa = xplus(a0.sort(K), D)
    gate_u0_X = float(np.abs(Xa - X).max())
    print('u=0 view X max abs diff', gate_u0_X, flush=True)
    assert gate_u0_X < 1e-9
    rep = {m: np.zeros(nq) for m in MODES}
    for k, (tr, te) in enumerate(outer):
        for m in MODES:
            qp, pr = predict_frame(a0, t0f, D, FITS[(k, m)], upairs[te])
            rep[m][qp] += pr.mean(0)
    gate_replay = {m: float(np.abs(rep[m] - P0[m]).max()) for m in MODES}
    print('original-pool replay through the view path', gate_replay, flush=True)
    assert max(gate_replay.values()) < 1e-5
    acc = {m: P0[m].copy() for m in MODES}; nv = 0
    for kind, seeds in (('thin', (1241, 2241)), ('pad', (1242, 2242))):
        for sd in seeds:
            for a, t in view_frames(D, kind, sd, .5, 10):
                for k, (tr, te) in enumerate(outer):
                    for m in MODES:
                        qp, pr = predict_frame(a, t, D, FITS[(k, m)], upairs[te])
                        acc[m][qp] += pr.mean(0)
                nv += 1
            print('views', nv, round(time.perf_counter() - t0), 's', flush=True)
    assert nv == 40
    PV = {m: acc[m] / (nv + 1) for m in MODES}
    s165 = pl.read_parquet(EXP / 'audit_scores165.parquet').sort(K); s165b = pl.read_parquet(EXP / 'audit_scores165b.parquet').sort(K)
    assert s165.select(K).equals(d.select(K)) and s165b.select(K).equals(d.select(K))
    S = {'NE1_0': S165.full_scores(D, P0['joint']), 'N21_0': S165.full_scores(D, P0['two_stage']),
         'NE1_v': S165.full_scores(D, PV['joint']), 'N21_v': S165.full_scores(D, PV['two_stage']),
         'NE': s165['NE'].to_numpy(), 'N2': s165['N2'].to_numpy(), 'v1g': s165b['v1g'].to_numpy(), 'v1': D['V1'], 'S_star': D['SS']}
    S['blend_v1g_NE1_v'] = S165.full_scores(D, .5 * (S['v1g'][D['QIX']] - 10) + .5 * PV['joint'])
    ac, dec = D['ac'], D['dec']
    DEC = {n: dec(v) for n, v in S.items()}
    R = lambda x: round(float(x), 6)

    def cmp(A, B):
        c = ac.evaluate(d, {'incumbent': A, 'arm': B})['comparisons']['arm']
        return {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']],
                'fam': {kk[:2]: R(x) for kk, x in c['by_family'].items()}, 'halves': {kk: R(x) for kk, x in c['halves'].items()}}
    comp = {'primary_NE1_v_vs_v1g': cmp(DEC['v1g'], DEC['NE1_v']),
            'N21_v_vs_v1g': cmp(DEC['v1g'], DEC['N21_v']),
            'stage1_input_effect_NE1_0_vs_NE': cmp(DEC['NE'], DEC['NE1_0']), 'stage1_input_effect_N21_0_vs_N2': cmp(DEC['N2'], DEC['N21_0']),
            'view_effect_NE1_v_vs_NE1_0': cmp(DEC['NE1_0'], DEC['NE1_v']), 'view_effect_N21_v_vs_N21_0': cmp(DEC['N21_0'], DEC['N21_v']),
            'end_to_end_effect_NE1_v_vs_N21_v': cmp(DEC['N21_v'], DEC['NE1_v']), 'blend_vs_v1g': cmp(DEC['v1g'], DEC['blend_v1g_NE1_v'])}
    rule = reg['rule']; c = comp['primary_NE1_v_vs_v1g']
    passes = c['mean'] > 0 and c['ci95'][0] > rule['ci_lower_floor'] and min(c['fam'].values()) >= rule['family_floor']
    maps = {n: R(ac.evaluate(d, {'incumbent': v})['sets']['incumbent']['map5']) for n, v in DEC.items()}
    d.select(K).with_columns(*[pl.Series(n, S[n]) for n in ('NE1_0', 'N21_0', 'NE1_v', 'N21_v', 'blend_v1g_NE1_v')]).write_parquet(EXP / 'audit_scores171.parquet')
    assert check_protected() == prot
    tr_ = {m: np.array([o[8] for o in out if o[1] == m]) for m in MODES}
    res = {'registration_sha256': reg_sha, 'gates': {'u0_view_X_max_abs_diff': gate_u0_X, 'view_path_replay_max_abs_diff': gate_replay, 'views': nv},
           'decoded_map5': maps, 'comparisons': comp, 'ER1_pass': bool(passes),
           'mean_best_epoch': {m: R(v[:, -1].mean()) for m, v in tr_.items()}, 'fits': len(out),
           'scores_sha256': S165.sha(EXP / 'audit_scores171.parquet'), 'held_reads': 11, 'wall_seconds': round(time.perf_counter() - t0, 1)}
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    for n, v in comp.items():
        print(n, json.dumps(v), flush=True)
    print('decoded MAP', json.dumps(maps), 'ER1', passes, 'wall', res['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
