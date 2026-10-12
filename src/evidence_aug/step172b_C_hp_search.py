"""Step 172b (Claude): registered hyperparameter search of the production evidence stage 2 (per-family LightGBM on the LP columns =
S* columns + 7 aligned stage-1 network outputs) at the v1 level (original pool + 40 label-free views), training-only 162-query audit.
Configurations: the incumbent (ac.CLF), 5 anchors and 90 random draws from a fixed space (generate_configs, seed 1720); the list is
embedded in the registration and re-generated here. Stochastic configurations (colsample or subsample < 1) are 5-seed bags (11-15).
One configuration is used for all three families. Every configuration is fitted on the cached folds of the three table partitions
(step172a) and scored as v1 (41-view probability mean), decoded with the production typed decoder (T1 off for CI).
Honest selection (registered): per query, AP@5 is averaged over partitions; halves = step5 half(table_id). The configuration best on
half h is scored on the other half against the incumbent; G = query-weighted mean of the two held-half deltas. Final = best on all
162 queries. Rule: G > 0 and the final configuration passes ER1 against the incumbent (mean > 0, pool-bootstrap CI lower > -.010,
every family >= -.020) and differs from the incumbent. Outputs experiments/hp_search_20260915/. Usage: run REGSHA | configs"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '1'
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/hp_search_20260915'
CACHE = EXP / 'cache'
REG = EXP / 'registration172.json'
PARTS = ['P0', 'P1651', 'P1652']
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
INCUMBENT = dict(n_estimators=150, num_leaves=7, max_depth=3, min_child_samples=40, learning_rate=.05, n_jobs=1, verbosity=-1)  # ac.CLF with n_jobs 1 (smoke: identical models to n_jobs 2)
BAG_SEEDS = [11, 12, 13, 14, 15]
SPACE = {'n_estimators': [60, 100, 150, 250, 400, 700], 'learning_rate': [.01, .02, .035, .05, .08, .12], 'num_leaves': [3, 4, 5, 7, 11, 15, 23, 31],
         'max_depth': [2, 3, 4, 5, 6, -1], 'min_child_samples': [5, 10, 20, 40, 60, 100], 'colsample_bytree': [.15, .3, .5, .7, 1.0],
         'subsample': [.5, .7, .85, 1.0], 'reg_lambda': [0., .5, 2., 5., 15.], 'reg_alpha': [0., 0., .1, 1.], 'min_split_gain': [0., 0., .02]}


def generate_configs():
    base = dict(INCUMBENT)
    cfgs = [('C00_incumbent', base),
            ('A01_ff50_bag', dict(base, colsample_bytree=.5)),
            ('A02_ff50_ss80_bag', dict(base, colsample_bytree=.5, subsample=.8, subsample_freq=1)),
            ('A03_300t_lr025', dict(base, n_estimators=300, learning_rate=.025)),
            ('A04_clf2_l15_d5_300t', dict(base, num_leaves=15, max_depth=5, n_estimators=300, min_child_samples=30)),
            ('A05_l3_d2_300t', dict(base, num_leaves=3, max_depth=2, n_estimators=300)),
            ('A06_l2reg5_ff30_bag', dict(base, reg_lambda=5., colsample_bytree=.3))]
    rng = np.random.default_rng(1720)
    while len(cfgs) < 97:
        c = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        c = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in c.items()}
        if not 2. <= c['n_estimators'] * c['learning_rate'] <= 40.:
            continue
        if c['subsample'] < 1.:
            c['subsample_freq'] = 1
        cfgs.append((f'R{len(cfgs):02d}', dict(c, n_jobs=1, verbosity=-1)))
    return cfgs


def stochastic(params):
    return params.get('colsample_bytree', 1.) < 1. or params.get('subsample', 1.) < 1.


def v1_full(params, V, idx, nz, y, fam, SS, QIX):
    """v1-level score over all audit rows, replicating step165b's summation order exactly (thin views 1-20, pads 21-40, original)."""
    import lightgbm as lgb
    seeds = BAG_SEEDS if stochastic(params) else [11]
    nv, nq = V.shape[0], V.shape[1]
    EB = nz['EB']
    P = np.zeros((nv, nq))
    for f in FAMS:
        isf = fam == f; cols = idx[f]
        for k in range(5):
            tr, te = nz[f'trq{k}'], nz[f'teq{k}']
            a, b = tr[isf[tr]], te[isf[te]]
            Xa = np.column_stack([V[0][:, cols][a], nz['EA'][k][a]])
            Zb = np.concatenate([np.column_stack([V[v][:, cols][b], EB[b]]) for v in range(nv)])
            preds = [lgb.LGBMClassifier(**params, random_state=s).fit(Xa, y[a]).predict_proba(Zb)[:, 1] for s in seeds]
            pr = preds[0] if len(preds) == 1 else np.mean(preds, axis=0)
            P[:, b] = pr.reshape(nv, len(b))
    def full(v):
        s = SS.copy(); s[QIX] = P[v] + 10
        return s
    thin, pad = np.zeros(len(SS)), np.zeros(len(SS))
    for v in range(1, 21):
        thin += full(v)
    for v in range(21, 41):
        pad += full(v)
    return (thin + pad + full(0)) / 41


W = {}


def worker_init():
    import polars as pl
    meta = json.loads((CACHE / 'meta172a.json').read_text(encoding='utf-8'))
    from evidence_aug import step165_C_e2e_audit as S165
    import audit_common as ac  # noqa: F401
    W['V'] = np.load(CACHE / 'views41.npy', mmap_mode='r')
    W['idx'] = {f: np.array(v) for f, v in meta['family_column_index'].items()}
    W['nz'] = {n: dict(np.load(CACHE / f'nn_{n}.npz')) for n in PARTS}
    lab = pl.read_parquet(CACHE / 'labels172.parquet')
    W['y'], W['fam'] = lab['relevance'].to_numpy().astype(int), lab['behavior_family'].to_numpy()
    aux = np.load(CACHE / 'aux172.npz'); W['SS'], W['QIX'] = aux['SS'], aux['QIX']


def job(args):
    cid, params, part = args
    t0 = time.perf_counter()
    V = np.asarray(W['V'])
    s = v1_full(params, V, W['idx'], W['nz'][part], W['y'], W['fam'], W['SS'], W['QIX'])
    return cid, part, s, round(time.perf_counter() - t0, 1)


def main():
    t0 = time.perf_counter()
    if sys.argv[1] == 'configs':
        print(json.dumps(generate_configs()))
        return
    reg_sha = sys.argv[2]
    from evidence_aug import step165_C_e2e_audit as S165
    assert S165.sha(REG) == reg_sha
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert S165.sha(Path(__file__)) == reg['code']['search_runner_sha256']
    rec172a = json.loads((CACHE / 'receipt172a.json').read_text(encoding='utf-8'))
    assert rec172a['registration_sha256'] == reg_sha and S165.sha(CACHE / 'meta172a.json') == rec172a['meta_sha256']
    cfgs = generate_configs()
    assert json.loads(json.dumps(cfgs)) == reg['configs']
    dest = EXP / 'results172.json'
    assert not dest.exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    import polars as pl
    from multiprocessing import Pool
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step65b_family_stage2 import boot
    from evidence_aug.step5_evaluate import half
    D = S165.load()
    d, Q = D['d'], D['Q']
    meta = json.loads((CACHE / 'meta172a.json').read_text(encoding='utf-8'))
    assert S165.sha(CACHE / 'views41.npy') == meta['views41_sha256']
    for n in PARTS:
        assert S165.sha(CACHE / f'nn_{n}.npz') == meta['partitions'][n]['sha256']
    Q.select('relevance', 'behavior_family').write_parquet(CACHE / 'labels172.parquet')
    np.savez(CACHE / 'aux172.npz', SS=D['SS'], QIX=D['QIX'])
    jobs = [(cid, p, part) for cid, p in cfgs for part in PARTS]
    # heavy configurations first so the pool finishes evenly
    cost = lambda p: p['n_estimators'] * (5 if stochastic(p) else 1) * (p['num_leaves'] if p['max_depth'] == -1 else min(p['num_leaves'], 2 ** p['max_depth']))
    jobs.sort(key=lambda j: -cost(j[1]))
    print('jobs', len(jobs), flush=True)
    S = {}
    with Pool(12, initializer=worker_init) as pool:
        for i, (cid, part, s, sec) in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            S[(cid, part)] = s
            if i % 24 == 0:
                print('fitted', i + 1, 'of', len(jobs), 'last', cid, part, sec, 's', round(time.perf_counter() - t0), 's', flush=True)
    # gate: incumbent on P0 through this path equals step165b v1LP (n_jobs as registered)
    s165b = pl.read_parquet(CV / 'experiments/e2e_evidence_C_20260915/audit_scores165b.parquet').sort(S165.K)
    gate = float(np.abs(S[('C00_incumbent', 'P0')] - s165b['v1LP'].to_numpy()).max())
    print('incumbent P0 v1LP replay', gate, flush=True)
    assert gate == 0.0
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    ids = [cid for cid, _ in cfgs]
    M = np.zeros((len(ids), len(PARTS), len(qids)))
    for i, cid in enumerate(ids):
        for j, part in enumerate(PARTS):
            _, m = summarize(key.with_columns(pl.Series('score', D['dec'](S[(cid, part)]))), truths, fams)
            M[i, j] = [m[q] for q in qids]
        if i % 16 == 0:
            print('decoded', i + 1, 'of', len(ids), round(time.perf_counter() - t0), 's', flush=True)
    np.save(EXP / 'maps172.npy', M)
    R = lambda x: round(float(x), 6)
    A = M.mean(1)  # configs x queries
    inc = ids.index('C00_incumbent')
    hv = np.array([half(ptab[q]) for q in qids])
    sel, held = {}, {}
    for h in (0, 1):
        on, off = hv == h, hv != h
        sh = int(np.argmax(A[:, on].mean(1)))
        sel[h] = ids[sh]
        held[h] = {'selected_on_half': h, 'config': ids[sh], 'held_half_delta': R((A[sh, off] - A[inc, off]).mean()), 'held_half_n': int(off.sum()),
                   'selection_half_delta': R((A[sh, on] - A[inc, on]).mean())}
    G = (held[0]['held_half_delta'] * held[0]['held_half_n'] + held[1]['held_half_delta'] * held[1]['held_half_n']) / len(qids)
    fin = int(np.argmax(A.mean(1)))
    a_map = {q: A[inc, t] for t, q in enumerate(qids)}; b_map = {q: A[fin, t] for t, q in enumerate(qids)}
    c = boot(a_map, b_map, ptab, pools)
    c['by_family'] = {f[:2]: R(np.mean([b_map[q] - a_map[q] for q in qids if fams[q] == f])) for f in FAMS}
    c['per_partition'] = {part: R((M[fin, j] - M[inc, j]).mean()) for j, part in enumerate(PARTS)}
    er1 = c['mean_delta'] > 0 and c['ci95'][0] > -.010 and min(c['by_family'].values()) >= -.020
    proceed = bool(G > 0 and er1 and fin != inc)
    order = np.argsort(-A.mean(1))
    table = [{'config': ids[i], 'mean_map5': R(A[i].mean()), 'delta': R(A[i].mean() - A[inc].mean()),
              'per_partition_delta': [R((M[i, j] - M[inc, j]).mean()) for j in range(len(PARTS))],
              'half0_delta': R((A[i, hv == 0] - A[inc, hv == 0]).mean()), 'half1_delta': R((A[i, hv == 1] - A[inc, hv == 1]).mean())} for i in order]
    from scipy.stats import spearmanr
    res = {'registration_sha256': reg_sha, 'gate_incumbent_P0_v1LP_replay': gate, 'incumbent_map5': R(A[inc].mean()),
           'incumbent_map5_per_partition': [R(M[inc, j].mean()) for j in range(len(PARTS))],
           'cross_half': held, 'G_honest': R(G), 'final_config': ids[fin], 'final_params': dict(cfgs[fin][1]),
           'final_vs_incumbent': {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']],
                                  'fam': c['by_family'], 'per_partition': c['per_partition']},
           'ER1_final': bool(er1), 'proceed': proceed,
           'descriptive': {'half_rank_spearman': R(spearmanr(A[:, hv == 0].mean(1), A[:, hv == 1].mean(1))[0]),
                           'n_configs_above_incumbent': int((A.mean(1) > A[inc].mean()).sum()),
                           'partition_rank_spearman': [R(spearmanr(M[:, a].mean(1), M[:, b].mean(1))[0]) for a, b in ((0, 1), (0, 2), (1, 2))]},
           'ranking': table, 'maps_sha256': S165.sha(EXP / 'maps172.npy'), 'wall_seconds': round(time.perf_counter() - t0, 1)}
    assert check_protected() == prot
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('G', res['G_honest'], 'final', res['final_config'], json.dumps(res['final_vs_incumbent']), 'ER1', er1, 'PROCEED', proceed, flush=True)


if __name__ == '__main__':
    main()
