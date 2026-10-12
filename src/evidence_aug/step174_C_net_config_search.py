"""Step 174 (Claude): H2 of the user's hyperparameter search. The LP stage-1 network (step165_e2e_mlp, joint mode) under alternative
settings, plus the user's original 'more features out of stage 1' in a seed-averageable form. Each arm refits the STEP165-165c network
(test rows: mean of 10 outer seeds; training rows: inner 4-fold OOF x 10 seeds) on the three cached table partitions of step172a and is
read as LP at the v1 level (41-view mean) with one fixed stage 2 (registration: 'stage2_params'), decoded with the production typed decoder.
Arms (registration 'arms'): N00 = the cached incumbent network (no refit); F = incumbent network + 4 permutation-invariant medium-layer
features per top-20 hand (cosine of h to the pair's top-5 stage-1 mean h; pair-normalised distance of h to the pair mean; stage-2 logit
centred on the pair's top-20; own stage-1 score z within pair), gate: its 7 standard columns equal the cache exactly; N01-N10 = single
setting changes; N11 = 20 seeds (cache seeds + 10 new).
Honest selection as step172b: per-query AP@5 averaged over partitions, cross-half G, final = best on all 162 queries, rule G > 0 and the
final arm passes ER1 against N00 and is not N00. Training-only 162-query audit. Outputs experiments/hp_search_20260915/net174/.
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

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/hp_search_20260915'
CACHE = EXP / 'cache'
OUT = EXP / 'net174'
REG = EXP / 'registration174.json'
PARTS = ['P0', 'P1651', 'P1652']
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
NEW_OUTER = [16520 + i for i in range(10)]
NEW_INNER = [16620 + i for i in range(10)]
ARMS = {
    'F_medium_features': {'config': {}, 'mode': 'joint', 'extra': True},
    'N01_wide_128_32_64': {'config': {'h1': 128, 'medium': 32, 'h2': 64}},
    'N02_narrow_32_8_16': {'config': {'h1': 32, 'medium': 8, 'h2': 16}},
    'N03_wd_1e-1': {'config': {'wd': 1e-1}},
    'N04_wd_1e-3': {'config': {'wd': 1e-3}},
    'N05_lr_2e-3': {'config': {'lr': 2e-3}},
    'N06_lam_list2_0': {'config': {'lam_list2': 0.}},
    'N07_lam_list2_2': {'config': {'lam_list2': 2.}},
    'N08_w_stage1_3': {'config': {'w_stage1': 3.}},
    'N09_two_stage': {'config': {}, 'mode': 'two_stage'},
    'N10_patience80_epochs800': {'config': {'patience': 80, 'epochs_max': 800}},
    'N11_seeds20': {'config': {}, 'seeds': 'new'},
}

G = {}


def pool_init(Xa, Ca, QIX, fam, rel, pid, tab, qpid):
    G.update(X=Xa, C=Ca, QIX=QIX, fam=fam, tab=tab, rel=rel, pid=pid, qpid=qpid)


def medium_features(rows, tq, cache, famidx):
    """Permutation-invariant summaries of the medium layer for top-20 rows tq (indices into Q) of the predicted batch."""
    pid = G['pid'][rows]
    h, S, o = cache['h'].astype(np.float64), cache['S'].astype(np.float64), cache['o'].astype(np.float64)
    own = S[np.arange(len(rows)), famidx]
    _, first = np.unique(pid, return_index=True)
    gid = np.repeat(np.arange(len(first)), np.diff(np.r_[first, len(pid)]))
    cnt = np.diff(np.r_[first, len(pid)]).astype(float)
    hbar = np.add.reduceat(h, first, axis=0) / cnt[:, None]
    # pair top-5 by own stage-1 score
    order = np.lexsort((-own, gid))
    rank = np.empty(len(own), int); rank[order] = np.arange(len(own)) - np.repeat(first, np.diff(np.r_[first, len(pid)]))
    t5 = rank < 5
    h5 = np.add.reduceat(h * t5[:, None], first, axis=0) / np.maximum(np.add.reduceat(t5.astype(float), first), 1)[:, None]
    nrm = lambda a: np.sqrt((a * a).sum(1)) + 1e-9
    cos = (h * h5[gid]).sum(1) / (nrm(h) * nrm(h5[gid]))
    dist = nrm(h - hbar[gid]); dist = dist / (np.add.reduceat(dist, first) / cnt)[gid]
    mu = (np.add.reduceat(own, first) / cnt)[gid]; sd = np.sqrt(np.add.reduceat((own - mu) ** 2, first) / cnt)[gid] + 1e-9
    z = (own - mu) / sd
    pos = np.full(len(G['pid']), -1); pos[rows] = np.arange(len(rows))
    sub = pos[G['QIX'][tq]]
    tp = G['qpid'][tq]
    _, tf = np.unique(tp, return_index=True)
    tc = np.diff(np.r_[tf, len(tp)]).astype(float)
    tg = np.repeat(np.arange(len(tf)), np.diff(np.r_[tf, len(tp)]))
    oc = o - (np.add.reduceat(o, tf) / tc)[tg]
    return np.column_stack([cos[sub], dist[sub], oc, z[sub]])


def fit_job(job):
    """step165 fit_job under an arm's network settings; returns aligned features (7, or 11 with medium features) for the predicted top-20."""
    from evidence_aug import step165_e2e_mlp as M
    from evidence_aug import step165_C_e2e_audit as S165
    tag, arm, seed, train_mask, pred_mask = job
    spec = ARMS.get(arm, {'config': {}})
    base = {'h1': 64, 'medium': 16, 'h2': 32, 'epochs_max': 400, 'patience': 40, 'lr': 1e-3, 'wd': 1e-2, 'lam_list2': .5, 'w_stage1': 1., 'clip': 6., 'val_share_tables': .2}
    M.CONFIG.clear(); M.CONFIG.update(base); M.CONFIG.update(spec['config'])
    S165.G.update(G); S165.G['d'] = pl.DataFrame({'relevance': G['rel'], 'pair_id': G['pid']})
    mode = spec.get('mode', 'joint')
    rows_tr = np.flatnonzero(train_mask)
    tabs = np.unique(G['tab'][rows_tr])
    vt = np.random.default_rng([seed, 7]).choice(tabs, size=max(1, int(round(len(tabs) * M.CONFIG['val_share_tables']))), replace=False)
    val_mask = train_mask & np.isin(G['tab'], vt)
    fit_mask = train_mask & ~val_mask
    Dd = {'d': S165.G['d'], 'fam': G['fam'], 'QIX': G['QIX'], 'Q': pl.DataFrame({'pair_id': G['qpid']})}
    Xs = S165.standardize(G['X'], rows_tr)
    Cs = S165.standardize(G['C'], np.flatnonzero(train_mask[G['QIX']]))
    Btr, _, _ = S165.make_batch(M, Dd, Xs, Cs, fit_mask)
    Bv, _, _ = S165.make_batch(M, Dd, Xs, Cs, val_mask)
    P, trace = M.train(Btr, mode, seed, Bv)
    Bte, rows, tq = S165.make_batch(M, Dd, Xs, Cs, pred_mask)
    cache = M.forward(P, Bte, with_stage2=True)
    p, S = 1 / (1 + np.exp(-cache['o'])), cache['S']
    af = S165.aligned_features(Dd, rows, tq, S.astype(np.float64), p.astype(np.float64))
    if spec.get('extra'):
        famidx = np.array([S165.FAMS.index(f) for f in G['fam'][rows]])
        af = np.column_stack([af, medium_features(rows, tq, cache, famidx)])
    return tag, arm, seed, tq, af


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[2]
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug import step172a_C_hp_cache as S172A
    from evidence_aug import step172b_C_hp_search as S172B
    assert S165.sha(REG) == reg_sha
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert S165.sha(Path(__file__)) == reg['code']['runner_sha256']
    assert json.loads(json.dumps(ARMS)) == reg['arms']
    stage2 = reg['stage2_params']
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / 'results174.json'
    assert not dest.exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    from multiprocessing import Pool
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step65b_family_stage2 import boot
    from evidence_aug.step5_evaluate import half
    D = S165.load()
    X, C = S165.matrices(D)
    d, Q = D['d'], D['Q']
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True); ptab = tables[pfirst]
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    qpairs = Q['pair_id'].to_numpy(); nq = len(Q)
    meta = json.loads((CACHE / 'meta172a.json').read_text(encoding='utf-8'))
    V = np.load(CACHE / 'views41.npy', mmap_mode='r'); assert S165.sha(CACHE / 'views41.npy') == meta['views41_sha256']
    idx = {f: np.array(v) for f, v in meta['family_column_index'].items()}
    cache_nz = {n: dict(np.load(CACHE / f'nn_{n}.npz')) for n in PARTS}
    for n in PARTS:
        assert S165.sha(CACHE / f'nn_{n}.npz') == meta['partitions'][n]['sha256']
    y = Q['relevance'].to_numpy().astype(int); fam = Q['behavior_family'].to_numpy()
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'], d['relevance'].to_numpy(), pairs, tables, qpairs)
    Pn = {n: S172A.partition(n, upairs, ptab) for n in PARTS}
    feats = {}  # arm -> partition -> nz dict
    with Pool(12, initializer=pool_init, initargs=init_args) as pool:
        for arm, spec in ARMS.items():
            outer_seeds = NEW_OUTER if spec.get('seeds') == 'new' else S165.SEEDS_INNER_OUTER
            inner_seeds = NEW_INNER if spec.get('seeds') == 'new' else S165.SEEDS_INNER
            jobs = []
            for n, (outer, inner) in Pn.items():
                for k, (tr, te) in enumerate(outer):
                    for s in outer_seeds:
                        jobs.append((('outer', n, k), arm, s, rowmask(tr), rowmask(te)))
                    for j, (itr, ite) in enumerate(inner[k]):
                        for s in inner_seeds:
                            jobs.append((('inner', n, k, j), arm, s, rowmask(tr[itr]), rowmask(tr[ite])))
            res = pool.map(fit_job, jobs, chunksize=1)
            feats[arm] = {}
            for n, (outer, inner) in Pn.items():
                J = res[0][4].shape[1]
                EA, EB = np.full((5, nq, J), np.nan), np.full((nq, J), np.nan)
                for k, (tr, te) in enumerate(outer):
                    trq = np.flatnonzero(np.isin(qpairs, upairs[tr])); teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
                    acc_in = {}
                    for tag, a_, s, tq, af in res:
                        if tag[1] == n and tag[0] == 'inner' and tag[2] == k:
                            acc_in.setdefault(tag[3], []).append((tq, af))
                    ea = np.full((nq, J), np.nan)
                    for j, lst in acc_in.items():
                        assert len(lst) == len(inner_seeds)
                        ea[lst[0][0]] = np.mean([a for _, a in lst], axis=0)
                    lst = [(tq, af) for tag, a_, s, tq, af in res if tag[1] == n and tag[0] == 'outer' and tag[2] == k]
                    assert len(lst) == len(outer_seeds)
                    EB[lst[0][0]] = np.mean([a for _, a in lst], axis=0)
                    EA[k] = ea
                    assert np.isfinite(ea[trq]).all() and np.isfinite(EB[teq]).all()
                nz = {'EA': EA, 'EB': EB, **{kk: v for kk, v in cache_nz[n].items() if kk.startswith(('trq', 'teq'))}}
                if spec.get('extra'):
                    assert np.array_equal(np.isnan(EA[:, :, :7]), np.isnan(cache_nz[n]['EA']))
                    gate = max(float(np.nanmax(np.abs(EA[:, :, :7] - cache_nz[n]['EA']))), float(np.abs(EB[:, :7] - cache_nz[n]['EB']).max()))
                    print(arm, n, 'standard-column replay vs cache', gate, flush=True)
                    assert gate == 0.0
                if spec.get('seeds') == 'new':
                    nz['EA'] = (EA + cache_nz[n]['EA']) / 2; nz['EB'] = (EB + cache_nz[n]['EB']) / 2
                feats[arm][n] = nz
                np.savez(OUT / f'nn_{arm}_{n}.npz', **nz)
            print('arm', arm, 'fits', len(jobs), round(time.perf_counter() - t0), 's', flush=True)
            del res
    feats = {'N00_incumbent': cache_nz, **feats}
    ids = list(feats)
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabd = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptabd.values()))
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    Mx = np.zeros((len(ids), len(PARTS), len(qids)))
    s165b = pl.read_parquet(CV / 'experiments/e2e_evidence_C_20260915/audit_scores165b.parquet').sort(S165.K)
    for i, arm in enumerate(ids):
        for j, n in enumerate(PARTS):
            s = S172B.v1_full(dict(stage2), np.asarray(V), idx, feats[arm][n], y, fam, D['SS'], D['QIX'])
            if arm == 'N00_incumbent' and n == 'P0' and stage2 == S172B.INCUMBENT:
                assert float(np.abs(s - s165b['v1LP'].to_numpy()).max()) == 0.0
            _, m = summarize(key.with_columns(pl.Series('score', D['dec'](s))), truths, fams)
            Mx[i, j] = [m[q] for q in qids]
        print('scored', arm, round(time.perf_counter() - t0), 's', flush=True)
    np.save(OUT / 'maps174.npy', Mx)
    R = lambda x: round(float(x), 6)
    A = Mx.mean(1); inc = 0
    hv = np.array([half(ptabd[q]) for q in qids])
    held = {}
    for h in (0, 1):
        on, off = hv == h, hv != h
        sh = int(np.argmax(A[:, on].mean(1)))
        held[h] = {'config': ids[sh], 'held_half_delta': R((A[sh, off] - A[inc, off]).mean()), 'held_half_n': int(off.sum()), 'selection_half_delta': R((A[sh, on] - A[inc, on]).mean())}
    Gh = (held[0]['held_half_delta'] * held[0]['held_half_n'] + held[1]['held_half_delta'] * held[1]['held_half_n']) / len(qids)
    fin = int(np.argmax(A.mean(1)))
    a_map = {q: A[inc, t] for t, q in enumerate(qids)}; b_map = {q: A[fin, t] for t, q in enumerate(qids)}
    c = boot(a_map, b_map, ptabd, pools)
    c['by_family'] = {f[:2]: R(np.mean([b_map[q] - a_map[q] for q in qids if fams[q] == f])) for f in FAMS}
    er1 = c['mean_delta'] > 0 and c['ci95'][0] > -.010 and min(c['by_family'].values()) >= -.020
    proceed = bool(Gh > 0 and er1 and fin != inc)
    table = [{'arm': ids[i], 'mean_map5': R(A[i].mean()), 'delta': R(A[i].mean() - A[inc].mean()),
              'per_partition_delta': [R((Mx[i, j] - Mx[inc, j]).mean()) for j in range(len(PARTS))],
              'half0_delta': R((A[i, hv == 0] - A[inc, hv == 0]).mean()), 'half1_delta': R((A[i, hv == 1] - A[inc, hv == 1]).mean()),
              'fam': {f[:2]: R(np.mean([A[i, t] - A[inc, t] for t, q in enumerate(qids) if fams[q] == f])) for f in FAMS}} for i in np.argsort(-A.mean(1))]
    out = {'registration_sha256': reg_sha, 'stage2_params': stage2, 'incumbent_map5': R(A[inc].mean()), 'cross_half': held, 'G_honest': R(Gh),
           'final_arm': ids[fin], 'final_vs_incumbent': {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']], 'fam': c['by_family']},
           'ER1_final': bool(er1), 'proceed': proceed, 'ranking': table, 'maps_sha256': S165.sha(OUT / 'maps174.npy'), 'wall_seconds': round(time.perf_counter() - t0, 1)}
    assert check_protected() == prot
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('G', out['G_honest'], 'final', out['final_arm'], json.dumps(out['final_vs_incumbent']), 'PROCEED', proceed, flush=True)


if __name__ == '__main__':
    main()
