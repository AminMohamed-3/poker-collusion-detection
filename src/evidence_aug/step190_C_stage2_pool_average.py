"""STEP190: average the whole 97-configuration stage-2 pool instead of selecting from it.

STEP172b selected among these same 97 configurations and failed honestly (G -0.000462).
STEP185 showed that the stage-1 pool STEP174 failed to select from becomes a gain the moment
it is averaged. This is the identical operation on the identical failure, one axis over.

STEP172b already ran these exact 291 fits but cached only the decoded per-query MAPs
(maps172.npy), which is why it could select and could not average. This run caches the SCORE
VECTORS, and gates every configuration against maps172.npy so the replay is provably exact.

Stage 1 is INC and is frozen: no network is fitted, and in production the whole pool is scored
inside ONE view-construction pass.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
CV = POKER / 'Claude Version'
EXP = CV / 'experiments/hp_search_20260915'
CACHE = EXP / 'cache'
OUT = CV / 'experiments/net_ensemble_20260916'
REG = OUT / 'registration190.json'
GRID = OUT / 'grid190.npz'
PARTS = ['P0', 'P1651', 'P1652']
NONBAG_ANCHORS = ['A03_300t_lr025', 'A04_clf2_l15_d5_300t', 'A05_l3_d2_300t']
BAGS4 = ['C00_incumbent', 'A01_ff50_bag', 'A02_ff50_ss80_bag', 'A06_l2reg5_ff30_bag']
CUM_N = [1, 2, 4, 8, 16, 32, 64, 97]
W = {}


def worker_init():
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug import step172b_C_hp_search as S172B
    meta = json.loads((CACHE / 'meta172a.json').read_text(encoding='utf-8'))
    D = S165.load()
    W['V'] = np.load(CACHE / 'views41.npy', mmap_mode='r')
    W['idx'] = {f: np.array(v) for f, v in meta['family_column_index'].items()}
    W['y'] = D['Q']['relevance'].to_numpy().astype(int)
    W['fam'] = D['Q']['behavior_family'].to_numpy()
    W['SS'], W['QIX'] = D['SS'], D['QIX']
    W['cfg'] = dict(S172B.generate_configs())
    W['NZ'] = {n: dict(np.load(CACHE / f'nn_{n}.npz')) for n in PARTS}


def cell_job(job):
    from evidence_aug import step172b_C_hp_search as S172B
    c, n = job
    s = S172B.v1_full(dict(W['cfg'][c]), np.asarray(W['V']), W['idx'], W['NZ'][n], W['y'], W['fam'], W['SS'], W['QIX'])
    return c, n, s


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[1]
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug import step172b_C_hp_search as S172B
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    assert S165.sha(REG) == reg_sha, 'registration hash mismatch'
    dest = OUT / 'results190.json'
    assert not dest.exists(), 'preserve a completed read'
    prot = check_protected()

    # generate_configs() returns [(name, params), ...]; maps172.npy rows are in exactly this order
    # (step172b: ids = [cid for cid, _ in cfgs]) and its PARTS order is identical, so the gate lines up.
    CFGS = [c for c, _ in S172B.generate_configs()]
    assert len(CFGS) == 97 and CFGS[0] == 'C00_incumbent', CFGS[:3]
    assert S172B.PARTS == PARTS, 'partition order must match maps172.npy'
    jobs = [(c, n) for c in CFGS for n in PARTS]

    SV = {}
    if GRID.exists():
        z = np.load(GRID)
        for k in z.files:
            c, n = k.rsplit('~', 1)
            SV[(c, n)] = z[k]
        print(f'loaded cached grid {len(SV)} cells', flush=True)
    if len(SV) < len(jobs):
        from multiprocessing import Pool
        with Pool(5, initializer=worker_init) as pool:
            done = 0
            for c, n, s in pool.imap_unordered(cell_job, jobs, chunksize=1):
                SV[(c, n)] = s
                done += 1
                if done % 24 == 0:
                    print(f'  {done}/{len(jobs)} cells  {round(time.perf_counter() - t0)}s', flush=True)
        np.savez(GRID, **{f'{c}~{n}': v for (c, n), v in SV.items()})
        print(f'grid cached to {GRID}  {round(time.perf_counter() - t0)}s', flush=True)

    D = S165.load()
    d = D['d']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabd = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabd[q]) for q in qids])
    R = lambda x: round(float(x), 6)

    ref = pl.read_parquet(CV / 'experiments/e2e_evidence_C_20260915/audit_scores165b.parquet').sort(S165.K)
    gate_inc = float(np.abs(SV[('C00_incumbent', 'P0')] - ref['v1LP'].to_numpy()).max())
    print(f'GATE incumbent cell vs step165b v1LP: {gate_inc}', flush=True)
    assert gate_inc == 0.0

    def per_query(s):
        _, mm = summarize(key.with_columns(pl.Series('score', D['dec'](s))), truths, fams)
        return np.array([mm[q] for q in qids])

    # ---- strong gate: replay every STEP172b decoded MAP exactly ----
    M = np.load(EXP / 'maps172.npy')
    assert M.shape == (97, 3, len(qids)), M.shape
    worst, nbad = 0.0, 0
    for i, c in enumerate(CFGS):
        for j, n in enumerate(PARTS):
            e = float(np.abs(per_query(SV[(c, n)]) - M[i, j]).max())
            worst = max(worst, e)
            nbad += e != 0.0
        if i % 24 == 0:
            print(f'  gate {i}/97 worst {worst}  {round(time.perf_counter() - t0)}s', flush=True)
    print(f'GATE vs maps172.npy: worst abs {worst}, cells differing {nbad}/291', flush=True)
    assert worst == 0.0, 'STEP172b replay is not exact'

    CACHED = {}

    def evaluate(cs):
        ck = tuple(sorted(cs))
        if ck in CACHED:
            return CACHED[ck]
        acc = np.zeros(len(qids))
        for n in PARTS:
            acc += per_query(np.mean([SV[(c, n)] for c in cs], axis=0))
        CACHED[ck] = acc / len(PARTS)
        return CACHED[ck]

    A0 = evaluate(['C00_incumbent'])
    print(f'incumbent mean AP@5 {A0.mean():.6f}', flush=True)

    table = {}

    def add(name, cs, tag='pre-specified'):
        A = evaluate(cs)
        table[name] = {'n_configs': len(cs), 'status': tag, 'mean_map5': R(A.mean()), 'delta': R(A.mean() - A0.mean()),
                       'half0_delta': R((A - A0)[hv == 0].mean()), 'half1_delta': R((A - A0)[hv == 1].mean())}
        print(f'{name:16s} n={len(cs):3d} {tag:20s} delta {A.mean() - A0.mean():+.6f} '
              f'h0 {(A - A0)[hv == 0].mean():+.6f} h1 {(A - A0)[hv == 1].mean():+.6f}  '
              f'{round(time.perf_counter() - t0)}s', flush=True)

    for N in CUM_N:
        add(f'POOL_{N}', CFGS[:N])
    add('RANDOM90', [c for c in CFGS if c.startswith('R')])
    add('ANCHORS_BAGS4', BAGS4)
    add('BAGGED94', [c for c in CFGS if c not in NONBAG_ANCHORS], 'POST-HOC-INFORMED')

    # ---- honest cross-half over the pre-specified arms only ----
    pre = [k for k, v in table.items() if v['status'] == 'pre-specified']
    held = {}
    for h in (0, 1):
        on = hv == h
        cells = {'POOL_%d' % N: CFGS[:N] for N in CUM_N}
        cells['RANDOM90'] = [c for c in CFGS if c.startswith('R')]
        cells['ANCHORS_BAGS4'] = BAGS4
        best = max(pre, key=lambda k: (evaluate(cells[k]) - A0)[on].mean())
        A = evaluate(cells[best])
        held[f'h{h}'] = {'picked_on_half': best, 'selection_half_delta': R((A - A0)[on].mean()),
                         'held_half_delta': R((A - A0)[~on].mean()), 'held_half_n': int((~on).sum())}
        print(f'cross-half h{h}: picked {best} sel {(A - A0)[on].mean():+.6f} held {(A - A0)[~on].mean():+.6f}', flush=True)
    G = float(np.average([held['h0']['held_half_delta'], held['h1']['held_half_delta']],
                         weights=[held['h0']['held_half_n'], held['h1']['held_half_n']]))
    print(f'G = {G:+.6f}', flush=True)

    # ---- registered build rule ----
    b4 = table['ANCHORS_BAGS4']
    ok = {k: (v['half0_delta'] > b4['half0_delta'] and v['half1_delta'] > b4['half1_delta'] and v['delta'] >= 0.0073)
          for k, v in table.items() if k != 'ANCHORS_BAGS4'}
    winners = [k for k, v in ok.items() if v]
    print('arms that beat ANCHORS_BAGS4 on both halves AND clear +.0073: ' + json.dumps(winners), flush=True)

    res = {'step': 190, 'registration_sha256': reg_sha, 'wall_seconds': round(time.perf_counter() - t0, 1),
           'gates': {'incumbent_vs_v1LP': gate_inc, 'maps172_replay_worst_abs': worst,
                     'maps172_cells_differing': int(nbad), 'protected': prot},
           'incumbent_mean_map5': R(A0.mean()), 'table': table, 'cross_half': held, 'G': R(G),
           'build_rule_winners': winners, 'reference_ANCHORS_BAGS4': b4}
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'wrote {dest}  {round(time.perf_counter() - t0)}s', flush=True)


if __name__ == '__main__':
    main()
