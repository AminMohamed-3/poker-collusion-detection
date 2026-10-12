"""Step 175 (Claude): H5 of the user's hyperparameter search, the typed decoder's input calibration. The production decoder
(step104 decode) reads the stage-2 probability p of each top-20 hand as P(listed) and orders hands by P(fewer than five listed hands
precede it in the listing order: T1 first, then chronology). It is exact only if p is calibrated; p is a 41-view probability mean
(LP at v1 level), which shrinks extremes. Arms: p' = sigmoid(a * logit(p) + b) before decoding, a in {.6, .8, 1, 1.25, 1.5, 2},
b in {-1, -.5, 0, .5, 1} (30 cells; a = 1, b = 0 is the incumbent and reproduces decode(v1LP) exactly).
Scores: incumbent LP v1 on the three step172a partitions (v1_full with ac.CLF; gate: P0 equals STEP165b v1LP exactly).
Honest selection as step172b (partition-averaged per-query AP@5, cross-half G, final = best cell on all 162 queries); rule G > 0 and
ER1 for the final cell against the incumbent and final != incumbent. Descriptive: top-20 log-loss and mean p by cell.
Output experiments/hp_search_20260915/calib175/results175.json. Usage: run REGSHA"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '2'
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/hp_search_20260915'
CACHE = EXP / 'cache'
OUT = EXP / 'calib175'
REG = EXP / 'registration175.json'
PARTS = ['P0', 'P1651', 'P1652']
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
A_GRID = [.6, .8, 1., 1.25, 1.5, 2.]
B_GRID = [-1., -.5, 0., .5, 1.]


def calibrate(full, QIX, a, b):
    """Apply p' = sigmoid(a logit(p) + b) to the top-20 rows (scores 10 + p); a = 1, b = 0 returns the input unchanged."""
    if a == 1. and b == 0.:
        return full
    s = full.copy()
    p = np.clip(full[QIX] - 10, 1e-6, 1 - 1e-6)
    s[QIX] = 10 + 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))
    return s


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[2]
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug import step172b_C_hp_search as S172B
    assert S165.sha(REG) == reg_sha
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert S165.sha(Path(__file__)) == reg['code']['runner_sha256'] and reg['grid'] == {'a': A_GRID, 'b': B_GRID}
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / 'results175.json'
    assert not dest.exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step65b_family_stage2 import boot
    from evidence_aug.step5_evaluate import half
    D = S165.load(); d, Q = D['d'], D['Q']
    meta = json.loads((CACHE / 'meta172a.json').read_text(encoding='utf-8'))
    assert S165.sha(CACHE / 'views41.npy') == meta['views41_sha256']
    V = np.load(CACHE / 'views41.npy', mmap_mode='r'); idx = {f: np.array(v) for f, v in meta['family_column_index'].items()}
    y = Q['relevance'].to_numpy().astype(int); fam = Q['behavior_family'].to_numpy()
    s165b = pl.read_parquet(CV / 'experiments/e2e_evidence_C_20260915/audit_scores165b.parquet').sort(S165.K)
    base = {}
    for n in PARTS:
        assert S165.sha(CACHE / f'nn_{n}.npz') == meta['partitions'][n]['sha256']
        base[n] = S172B.v1_full(dict(S172B.INCUMBENT), np.asarray(V), idx, dict(np.load(CACHE / f'nn_{n}.npz')), y, fam, D['SS'], D['QIX'])
    gate = float(np.abs(base['P0'] - s165b['v1LP'].to_numpy()).max()); assert gate == 0.0
    print('v1LP bases built, P0 replay', gate, round(time.perf_counter() - t0), 's', flush=True)
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    cells = [(a, b) for a in A_GRID for b in B_GRID]
    ids = [f'a{a}_b{b}' for a, b in cells]
    inc = ids.index('a1.0_b0.0')
    M = np.zeros((len(cells), len(PARTS), len(qids)))
    ll, mp = {}, {}
    yv = y
    for i, (a, b) in enumerate(cells):
        lls = []
        for j, n in enumerate(PARTS):
            s = calibrate(base[n], D['QIX'], a, b)
            _, m = summarize(key.with_columns(pl.Series('score', D['dec'](s))), truths, fams)
            M[i, j] = [m[q] for q in qids]
            pv = np.clip(s[D['QIX']] - 10, 1e-6, 1 - 1e-6)
            lls.append(float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))); mp[ids[i]] = round(float(pv.mean()), 4)
        ll[ids[i]] = round(float(np.mean(lls)), 5)
    np.save(OUT / 'maps175.npy', M)
    R = lambda x: round(float(x), 6)
    A = M.mean(1)
    hv = np.array([half(ptab[q]) for q in qids])
    held = {}
    for h in (0, 1):
        on, off = hv == h, hv != h
        sh = int(np.argmax(A[:, on].mean(1)))
        held[h] = {'cell': ids[sh], 'held_half_delta': R((A[sh, off] - A[inc, off]).mean()), 'held_half_n': int(off.sum()), 'selection_half_delta': R((A[sh, on] - A[inc, on]).mean())}
    G = (held[0]['held_half_delta'] * held[0]['held_half_n'] + held[1]['held_half_delta'] * held[1]['held_half_n']) / len(qids)
    fin = int(np.argmax(A.mean(1)))
    a_map = {q: A[inc, t] for t, q in enumerate(qids)}; b_map = {q: A[fin, t] for t, q in enumerate(qids)}
    c = boot(a_map, b_map, ptab, pools)
    c['by_family'] = {f[:2]: R(np.mean([b_map[q] - a_map[q] for q in qids if fams[q] == f])) for f in FAMS}
    er1 = c['mean_delta'] > 0 and c['ci95'][0] > -.010 and min(c['by_family'].values()) >= -.020
    proceed = bool(G > 0 and er1 and fin != inc)
    grid = {ids[i]: {'delta': R(A[i].mean() - A[inc].mean()), 'half0': R((A[i, hv == 0] - A[inc, hv == 0]).mean()), 'half1': R((A[i, hv == 1] - A[inc, hv == 1]).mean()),
                     'per_partition': [R((M[i, j] - M[inc, j]).mean()) for j in range(len(PARTS))], 'logloss_top20': ll[ids[i]], 'mean_p': mp[ids[i]]} for i in range(len(cells))}
    out = {'registration_sha256': reg_sha, 'gate_P0_v1LP_replay': gate, 'incumbent_map5': R(A[inc].mean()), 'cross_half': held, 'G_honest': R(G),
           'final_cell': ids[fin], 'final_vs_incumbent': {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']], 'fam': c['by_family']},
           'ER1_final': bool(er1), 'proceed': proceed, 'grid': grid, 'maps_sha256': S165.sha(OUT / 'maps175.npy'), 'wall_seconds': round(time.perf_counter() - t0, 1)}
    assert check_protected() == prot
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('G', out['G_honest'], 'final', out['final_cell'], json.dumps(out['final_vs_incumbent']), 'PROCEED', proceed, flush=True)


if __name__ == '__main__':
    main()
