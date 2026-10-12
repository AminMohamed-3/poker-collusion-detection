"""Step 178 (Claude): H5b, decomposition of the STEP175 decoder calibration. The typed decoder scores a top-20 hand as p * q, where q is
P(fewer than five listed hands precede it) computed from the probabilities of the hands before it (T1 first, then chronology). STEP175
shifted one probability for both roles (p' = sigmoid(logit p + .5)). Here the own factor and the counting probabilities get separate
shifts: score = sigmoid(logit p + b_own) * q(sigmoid(logit p + b_cnt)). Grid b_own in {0, .25, .5, .75, 1} x b_cnt in {0, .25, .5, .75,
1, 1.5} (30 cells); incumbent (.5, .5) = STEP175 production; (0, 0) = the pre-STEP175 decoder. Scores: LP v1 on the three step172a
partitions (gate: P0 equals STEP165b v1LP); gates: decode2(s, s) equals step104 decode(s), and cell (.5, .5) equals STEP175's a1.0_b0.5
per-query maps. Honest selection as STEP175 (cross-half G vs incumbent, final = best on all 162 queries, rule G > 0 and ER1 and final !=
incumbent). Output experiments/hp_search_20260915/own_count178/results178.json. Usage: run REGSHA"""
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
OUT = EXP / 'own_count178'
REG = EXP / 'registration178.json'
PARTS = ['P0', 'P1651', 'P1652']
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
B_OWN = [0., .25, .5, .75, 1.]
B_CNT = [0., .25, .5, .75, 1., 1.5]
INC = 'own0.5_cnt0.5'


def shift(full, QIX, b):
    from evidence_aug.step175_C_decoder_calibration import calibrate
    return calibrate(full, QIX, 1., b)


def decode2(fr, t1, s_own, s_cnt):
    """step104 decode with separate own and counting probabilities; decode2(s, s) == decode(s)."""
    from evidence_aug.step104_C_typed_decoder_build import less_than_five
    out = s_own.copy()
    pid, cr = fr['pair_id'].to_numpy(), fr['chrono_rank'].to_numpy()
    idx = np.flatnonzero(s_own > 5)
    assert np.array_equal(idx, np.flatnonzero(s_cnt > 5))
    _, first = np.unique(pid[idx], return_index=True)
    for g in np.split(idx, first[1:]):
        p = np.clip(s_own[g] - 10, 0, 1); pc = np.clip(s_cnt[g] - 10, 0, 1)
        order = np.lexsort((cr[g], ~t1[g]))
        q = np.empty(len(g)); q[order] = less_than_five(pc[order])
        out[g] = 10 + p * q + 1e-9 * p
    return out


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[2]
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug import step172b_C_hp_search as S172B
    from evidence_aug.step104_C_typed_decoder_build import chronology, align
    assert S165.sha(REG) == reg_sha
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert S165.sha(Path(__file__)) == reg['code']['runner_sha256'] and reg['grid'] == {'b_own': B_OWN, 'b_cnt': B_CNT}
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / 'results178.json'
    assert not dest.exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step65b_family_stage2 import boot
    from evidence_aug.step5_evaluate import half
    D = S165.load(); d, Q = D['d'], D['Q']
    v = pl.read_parquet(CV / 'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(S165.K)
    fr = chronology(d.select(S165.K + ['table_id']).with_columns(pl.Series('score', v['S_star'].to_numpy())))
    t1 = align(fr, pl.read_parquet(CV / 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
    typed = t1 & (D['fam'] != S165.CI)
    meta = json.loads((CACHE / 'meta172a.json').read_text(encoding='utf-8'))
    assert S165.sha(CACHE / 'views41.npy') == meta['views41_sha256']
    V = np.load(CACHE / 'views41.npy', mmap_mode='r'); idx = {f: np.array(x) for f, x in meta['family_column_index'].items()}
    y = Q['relevance'].to_numpy().astype(int); fam = Q['behavior_family'].to_numpy()
    s165b = pl.read_parquet(CV / 'experiments/e2e_evidence_C_20260915/audit_scores165b.parquet').sort(S165.K)
    base = {}
    for n in PARTS:
        assert S165.sha(CACHE / f'nn_{n}.npz') == meta['partitions'][n]['sha256']
        base[n] = S172B.v1_full(dict(S172B.INCUMBENT), np.asarray(V), idx, dict(np.load(CACHE / f'nn_{n}.npz')), y, fam, D['SS'], D['QIX'])
    gate_v1 = float(np.abs(base['P0'] - s165b['v1LP'].to_numpy()).max()); assert gate_v1 == 0.0
    s5 = shift(base['P0'], D['QIX'], .5)
    gate_dec = float(np.abs(decode2(fr, typed, s5, s5) - D['dec'](s5)).max()); assert gate_dec == 0.0
    print('bases built; gates v1', gate_v1, 'decode2', gate_dec, round(time.perf_counter() - t0), 's', flush=True)
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    cells = [(a, b) for a in B_OWN for b in B_CNT]
    ids = [f'own{a}_cnt{b}' for a, b in cells]
    inc = ids.index(INC)
    M = np.zeros((len(cells), len(PARTS), len(qids)))
    shifted = {n: {b: shift(base[n], D['QIX'], b) for b in sorted(set(B_OWN) | set(B_CNT))} for n in PARTS}
    for i, (a, b) in enumerate(cells):
        for j, n in enumerate(PARTS):
            s = decode2(fr, typed, shifted[n][a], shifted[n][b])
            _, m = summarize(key.with_columns(pl.Series('score', s)), truths, fams)
            M[i, j] = [m[q] for q in qids]
    M175 = np.load(EXP / 'calib175/maps175.npy')
    ids175 = [f'a{a}_b{b}' for a in [.6, .8, 1., 1.25, 1.5, 2.] for b in [-1., -.5, 0., .5, 1.]]
    gate175 = float(np.abs(M[inc] - M175[ids175.index('a1.0_b0.5')]).max()); assert gate175 == 0.0
    gate175_old = float(np.abs(M[ids.index('own0.0_cnt0.0')] - M175[ids175.index('a1.0_b0.0')]).max()); assert gate175_old == 0.0
    np.save(OUT / 'maps178.npy', M)
    R = lambda x: round(float(x), 6)
    A = M.mean(1)
    hv = np.array([half(ptab[q]) for q in qids])
    held = {}
    for h in (0, 1):
        on, off = hv == h, hv != h
        sc = A[:, on].mean(1)
        sh = max(range(len(ids)), key=lambda i: (sc[i], i == inc, -i))
        held[h] = {'cell': ids[sh], 'held_half_delta': R((A[sh, off] - A[inc, off]).mean()), 'held_half_n': int(off.sum()), 'selection_half_delta': R((A[sh, on] - A[inc, on]).mean())}
    G = (held[0]['held_half_delta'] * held[0]['held_half_n'] + held[1]['held_half_delta'] * held[1]['held_half_n']) / len(qids)
    tot = A.mean(1)
    fin = max(range(len(ids)), key=lambda i: (tot[i], i == inc, -i))
    a_map = {q: A[inc, t] for t, q in enumerate(qids)}; b_map = {q: A[fin, t] for t, q in enumerate(qids)}
    c = boot(a_map, b_map, ptab, pools)
    c['by_family'] = {f[:2]: R(np.mean([b_map[q] - a_map[q] for q in qids if fams[q] == f])) for f in FAMS}
    er1 = c['mean_delta'] > 0 and c['ci95'][0] > -.010 and min(c['by_family'].values()) >= -.020
    proceed = bool(G > 0 and er1 and fin != inc)
    grid = {ids[i]: {'delta': R(A[i].mean() - A[inc].mean()), 'half0': R((A[i, hv == 0] - A[inc, hv == 0]).mean()), 'half1': R((A[i, hv == 1] - A[inc, hv == 1]).mean()),
                     'per_partition': [R((M[i, j] - M[inc, j]).mean()) for j in range(len(PARTS))]} for i in range(len(cells))}
    out = {'registration_sha256': reg_sha, 'gates': {'P0_v1LP_replay': gate_v1, 'decode2_equals_decode': gate_dec, 'inc_equals_175_a1_b0.5': gate175, 'own0_cnt0_equals_175_a1_b0': gate175_old},
           'incumbent_map5': R(A[inc].mean()), 'cross_half': held, 'G_honest': R(G), 'final_cell': ids[fin],
           'final_vs_incumbent': {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']], 'fam': c['by_family']},
           'ER1_final': bool(er1), 'proceed': proceed, 'grid': grid, 'maps_sha256': S165.sha(OUT / 'maps178.npy'), 'wall_seconds': round(time.perf_counter() - t0, 1)}
    assert check_protected() == prot
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('G', out['G_honest'], 'final', out['final_cell'], json.dumps(out['final_vs_incumbent']), 'PROCEED', proceed, flush=True)


if __name__ == '__main__':
    main()
