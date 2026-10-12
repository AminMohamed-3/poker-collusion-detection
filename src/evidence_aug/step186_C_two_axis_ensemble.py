"""Step 186 (Claude): ensemble the evidence pipeline on BOTH cached axes at once. STEP172b searched 97 stage-2 LightGBM configurations and
failed to pick one (G -.000462); STEP174 searched 12 stage-1 network architectures and failed to pick one (G -.000460); STEP185 then showed
the stage-1 pool turns into a gain the moment it is averaged instead of selected from (G +.002769, final +.00736, CI [+.0025, +.0130]).
This scores the full factorial of 11 stage-1 members x 7 stage-2 anchor configurations x 3 partitions = 231 v1 score vectors, so that any
stage-1 subset crossed with any stage-2 subset can be averaged for free afterwards.
The seven stage-2 configurations are STEP172b's ANCHORS, hand-designed for diversity before any result was seen; the 90 random draws are
deliberately not used as a pool to select from. Gate: the (INC, C00_incumbent) cell must reproduce step165b's v1LP exactly, and the
stage-2 = C00 column must reproduce STEP185's SCORE_mean numbers.
Selection is honest exactly as STEP172b/174/185: per-query AP@5 averaged over partitions, halves = step5 half(table_id), best on one half
scored on the other, G = query-weighted mean of the held-half deltas. Greedy runs per half and never sees the held half. The build bar is
unchanged at +.02 decoded (OBS 19.133). Training-only 162-query audit, no uploads, no production effect.
Usage: step186_C_two_axis_ensemble.py REGSHA"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/hp_search_20260915'
CACHE, NET = EXP / 'cache', EXP / 'net174'
OUT = CV / 'experiments/net_ensemble_20260916'
REG = OUT / 'registration186.json'
PARTS = ['P0', 'P1651', 'P1652']
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
INC = 'INC'
ARMS = ['N01_wide_128_32_64', 'N02_narrow_32_8_16', 'N03_wd_1e-1', 'N04_wd_1e-3', 'N05_lr_2e-3', 'N06_lam_list2_0',
        'N07_lam_list2_2', 'N08_w_stage1_3', 'N09_two_stage', 'N11_seeds20']
S1 = [INC] + ARMS
S2 = ['C00_incumbent', 'A01_ff50_bag', 'A02_ff50_ss80_bag', 'A03_300t_lr025', 'A04_clf2_l15_d5_300t', 'A05_l3_d2_300t', 'A06_l2reg5_ff30_bag']
GREEDY_MAX = 5
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
    W['NZ'] = {}
    for n in PARTS:
        W['NZ'][(INC, n)] = dict(np.load(CACHE / f'nn_{n}.npz'))
        for a in ARMS:
            W['NZ'][(a, n)] = dict(np.load(NET / f'nn_{a}_{n}.npz'))


def cell_job(job):
    from evidence_aug import step172b_C_hp_search as S172B
    m, c, n = job
    s = S172B.v1_full(dict(W['cfg'][c]), np.asarray(W['V']), W['idx'], W['NZ'][(m, n)], W['y'], W['fam'], W['SS'], W['QIX'])
    return m, c, n, s


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[1]
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug import step172b_C_hp_search as S172B
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step65b_family_stage2 import boot
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    assert S165.sha(REG) == reg_sha, 'registration hash mismatch'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert reg['stage1_members'] == S1 and reg['stage2_configs']['set'] == S2
    dest = OUT / 'results186.json'
    assert not dest.exists(), 'preserve a completed read'
    prot = check_protected()
    cfgs = dict(S172B.generate_configs())
    for c in S2:
        assert c in cfgs, c

    D = S165.load()
    d = D['d']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabd = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptabd.values()))
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabd[q]) for q in qids])

    jobs = [(m, c, n) for m in S1 for c in S2 for n in PARTS]
    print(f'{len(jobs)} cells', flush=True)
    SV = {}
    GRID = OUT / 'grid186.npz'
    if GRID.exists():
        z = np.load(GRID)
        for k in z.files:
            m, c, n = k.split('~')
            SV[(m, c, n)] = z[k]
        print('loaded cached grid', len(SV), 'cells', flush=True)
    from multiprocessing import Pool
    if len(SV) < len(jobs):
      with Pool(5, initializer=worker_init) as pool:
        done = 0
        for m, c, n, s in pool.imap_unordered(cell_job, jobs, chunksize=1):
            SV[(m, c, n)] = s
            done += 1
            if done % 21 == 0:
                print(f'  {done}/{len(jobs)} cells  {round(time.perf_counter() - t0)}s', flush=True)
      np.savez(GRID, **{f'{m}~{c}~{n}': v for (m, c, n), v in SV.items()})
      print('grid cached to', GRID, flush=True)

    ref = pl.read_parquet(CV / 'experiments/e2e_evidence_C_20260915/audit_scores165b.parquet').sort(S165.K)
    gate_inc = float(np.abs(SV[(INC, 'C00_incumbent', 'P0')] - ref['v1LP'].to_numpy()).max())
    print('gate incumbent cell vs step165b v1LP', gate_inc, flush=True)

    CACHED = {}

    def evaluate(cells):
        ck = tuple(sorted(cells))
        if ck in CACHED:
            return CACHED[ck]
        acc = np.zeros(len(qids))
        for n in PARTS:
            s = np.mean([SV[(m, c, n)] for m, c in cells], axis=0)
            _, mm = summarize(key.with_columns(pl.Series('score', D['dec'](s))), truths, fams)
            acc += np.array([mm[q] for q in qids])
        CACHED[ck] = acc / len(PARTS)
        return CACHED[ck]

    R = lambda x: round(float(x), 6)
    A0 = evaluate([(INC, 'C00_incumbent')])
    print(f'incumbent cell mean AP@5 {A0.mean():.6f}', flush=True)

    # ---- gate: the C00 column must reproduce STEP185's SCORE_mean numbers ----
    r185 = json.loads((OUT / 'results185.json').read_text(encoding='utf-8'))
    g185 = {}
    for k, v in r185['table'].items():
        if v['combiner'] == 'SCORE_mean':
            A = evaluate([(m, 'C00_incumbent') for m in v['members']])
            g185[k] = R(A.mean() - A0.mean() - v['delta'])
    print('gate vs STEP185 SCORE_mean (should all be 0.0):', json.dumps(g185), flush=True)

    # ---- marginals ----
    table = {}

    def add(name, cells):
        A = evaluate(cells)
        table[name] = {'n_cells': len(cells), 'mean_map5': R(A.mean()), 'delta': R(A.mean() - A0.mean()),
                       'half0_delta': R((A - A0)[hv == 0].mean()), 'half1_delta': R((A - A0)[hv == 1].mean()),
                       'cells': [f'{m}|{c}' for m, c in cells] if len(cells) <= 12 else f'{len(cells)} cells'}
        print(f'{name:28s} n={len(cells):3d} delta {A.mean() - A0.mean():+.6f} '
              f'h0 {(A - A0)[hv == 0].mean():+.6f} h1 {(A - A0)[hv == 1].mean():+.6f}  {round(time.perf_counter() - t0)}s', flush=True)

    for c in S2:
        add(f'S2only_{c}', [(INC, c)])
    add('S2_ALL7_at_INC', [(INC, c) for c in S2])
    add('S1_ALL11_at_C00', [(m, 'C00_incumbent') for m in S1])
    add('FULL77', [(m, c) for m in S1 for c in S2])
    add('S1_LOSS_x_S2_ALL7', [(m, c) for m in [INC, 'N06_lam_list2_0', 'N07_lam_list2_2', 'N08_w_stage1_3'] for c in S2])
    add('S1_ALL11_x_S2_BAGS', [(m, c) for m in S1 for c in ['C00_incumbent', 'A01_ff50_bag', 'A02_ff50_ss80_bag', 'A06_l2reg5_ff30_bag']])
    add('S1_185winner_x_S2_ALL7', [(m, c) for m in [INC, 'N06_lam_list2_0', 'N11_seeds20'] for c in S2])

    # ---- greedy over the 77 cells, per half, held half never seen ----
    allcells = [(m, c) for m in S1 for c in S2]
    greedy = {}
    for h in (0, 1):
        on = hv == h
        cur, best = [(INC, 'C00_incumbent')], evaluate([(INC, 'C00_incumbent')])[on].mean()
        path = [{'add': 'INC|C00_incumbent', 'sel_mean': R(best)}]
        for _ in range(GREEDY_MAX):
            cands = [(evaluate(cur + [x])[on].mean(), x) for x in allcells if x not in cur]
            v, x = max(cands)
            if v <= best + 1e-9:
                break
            cur, best = cur + [x], v
            path.append({'add': f'{x[0]}|{x[1]}', 'sel_mean': R(best)})
        A = evaluate(cur)
        greedy[f'h{h}'] = {'cells': [f'{m}|{c}' for m, c in cur], 'path': path,
                           'selection_half_delta': R((A - A0)[on].mean()), 'held_half_delta': R((A - A0)[~on].mean()), 'held_half_n': int((~on).sum())}
        print(f'greedy h{h}: {greedy[f"h{h}"]["cells"]} sel {(A - A0)[on].mean():+.6f} held {(A - A0)[~on].mean():+.6f}  {round(time.perf_counter() - t0)}s', flush=True)

    # ---- honest cross-half over named + greedy ----
    pool_c = {k: [tuple(x.split('|')) for x in v['cells']] if isinstance(v['cells'], list) else None for k, v in table.items()}
    pool_c = {k: v for k, v in pool_c.items() if v}
    pool_c['FULL77'] = allcells
    pool_c['S1_LOSS_x_S2_ALL7'] = [(m, c) for m in [INC, 'N06_lam_list2_0', 'N07_lam_list2_2', 'N08_w_stage1_3'] for c in S2]
    pool_c['S1_ALL11_x_S2_BAGS'] = [(m, c) for m in S1 for c in ['C00_incumbent', 'A01_ff50_bag', 'A02_ff50_ss80_bag', 'A06_l2reg5_ff30_bag']]
    pool_c['S1_ALL11_at_C00'] = [(m, 'C00_incumbent') for m in S1]
    for h in (0, 1):
        pool_c[f'GREEDY_h{h}'] = [tuple(x.split('|')) for x in greedy[f'h{h}']['cells']]
    held = {}
    for h in (0, 1):
        on = hv == h
        sel = max(pool_c, key=lambda k: evaluate(pool_c[k])[on].mean())
        A = evaluate(pool_c[sel])
        held[h] = {'config': sel, 'n_cells': len(pool_c[sel]), 'selection_half_delta': R((A - A0)[on].mean()),
                   'held_half_delta': R((A - A0)[~on].mean()), 'held_half_n': int((~on).sum())}
    Gh = (held[0]['held_half_delta'] * held[0]['held_half_n'] + held[1]['held_half_delta'] * held[1]['held_half_n']) / len(qids)

    named_only = {k: v for k, v in pool_c.items() if not k.startswith('GREEDY_')}
    held_n = {}
    for h in (0, 1):
        on = hv == h
        sel = max(named_only, key=lambda k: evaluate(named_only[k])[on].mean())
        A = evaluate(named_only[sel])
        held_n[h] = {'config': sel, 'held_half_delta': R((A - A0)[~on].mean()), 'held_half_n': int((~on).sum())}
    Gn = (held_n[0]['held_half_delta'] * held_n[0]['held_half_n'] + held_n[1]['held_half_delta'] * held_n[1]['held_half_n']) / len(qids)

    fin = max(pool_c, key=lambda k: evaluate(pool_c[k]).mean())
    Af = evaluate(pool_c[fin])
    a_map = {q: A0[t] for t, q in enumerate(qids)}
    b_map = {q: Af[t] for t, q in enumerate(qids)}
    c = boot(a_map, b_map, ptabd, pools)
    c['by_family'] = {f[:2]: R(np.mean([b_map[q] - a_map[q] for q in qids if fams[q] == f])) for f in FAMS}
    er1 = bool(c['mean_delta'] > 0 and c['ci95'][0] > -.010 and min(c['by_family'].values()) >= -.020)
    proceed = bool(Gh > 0 and er1 and pool_c[fin] != [(INC, 'C00_incumbent')])
    build = bool(proceed and c['mean_delta'] >= .02)

    out = {'registration_sha256': reg_sha, 'author': 'claude', 'stage1': S1, 'stage2': S2,
           'gates': {'incumbent_cell_vs_step165b_v1LP': gate_inc, 'C00_column_vs_step185_SCORE_mean': g185},
           'incumbent_mean_map5': R(A0.mean()),
           'table': dict(sorted(table.items(), key=lambda kv: -kv[1]['delta'])), 'greedy': greedy,
           'cross_half': held, 'G_honest': R(Gh),
           'G_registered_note': 'G_honest follows the registration exactly: the max is taken over a pool that INCLUDES the two greedy arms. A greedy arm is built to win its own selection half, so it is always picked there and then regresses, which makes this statistic partly a measure of greedy shrinkage rather than of ensembling.',
           'POSTHOC_G_named_only': R(Gn), 'POSTHOC_cross_half_named_only': held_n,
           'POSTHOC_note': 'NOT the registered rule -- computed after seeing that greedy shrank 91 percent (h0 sel +.0158 -> held +.0014). Restricts the selection pool to the pre-registered named compositions, which involve no post-hoc choice at all. Reported as a diagnostic, not as the pass criterion.', 'final_config': fin, 'final_n_cells': len(pool_c[fin]),
           'final_vs_incumbent': {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'by_family': c['by_family']},
           'ER1_final': er1, 'proceed': proceed,
           'build_bar': 'OBS 19.133: mean decoded delta >= +.02; not relaxed because STEP185 was positive',
           'build_justified': build, 'wall_seconds': round(time.perf_counter() - t0, 1)}
    assert check_protected() == prot
    assert gate_inc == 0.0, 'incumbent cell must reproduce step165b exactly'
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('FINAL', fin, 'G', R(Gh), 'delta', R(c['mean_delta']), 'ER1', er1, 'PROCEED', proceed, 'BUILD', build, flush=True)


if __name__ == '__main__':
    main()
