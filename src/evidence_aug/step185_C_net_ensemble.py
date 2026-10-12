"""Step 185 (Claude): ensemble the evidence stage-1 network across architectures, instead of picking one of them. Every public gain since
.90321 came from adding model diversity to an ensemble, but always on the PAIR tail (STEP157/158/163); the evidence network has never been
ensembled. STEP167 production is a single configuration and STEP174 searched 12 configurations in order to SELECT one -- which failed
(G -.00046) even though its best single arm read +.00593, because the arms disagree about which half of the tables they help.
Nothing is refitted: STEP174 cached every arm's aligned stage-1 features (net174/nn_<ARM>_<PART>.npz, 8076 s of fitting), so this only
combines and rescores. Members are the incumbent plus the eleven 7-column arms; F_medium_features is excluded (11 columns, not 7) and
N10_patience80_epochs800 is excluded as a bit-identical duplicate of the incumbent (patience 40 stopped before the 400-epoch cap) --
both exclusions are in the registration amendment, written before any composition was scored.
Three combiners: FEAT_mean (average the aligned EA/EB columns, one v1_full), SCORE_mean (average the member v1 score vectors),
SCORE_rankmean (average within-pair ranks and reassign the value multiset, so magnitudes are untouched and only the order changes).
Selection is honest exactly as STEP172b/174: per-query AP@5 averaged over the three partitions, halves = step5 half(table_id), the best
composition on one half is scored on the other, G = query-weighted mean of the two held-half deltas. Greedy forward selection is run
separately per half and never sees the held half. Training-only 162-query audit, no uploads, no production effect.
Usage: step185_C_net_ensemble.py REGSHA"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
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
REG = OUT / 'registration185.json'
PARTS = ['P0', 'P1651', 'P1652']
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
INC = 'INC'
ARMS = ['N01_wide_128_32_64', 'N02_narrow_32_8_16', 'N03_wd_1e-1', 'N04_wd_1e-3', 'N05_lr_2e-3', 'N06_lam_list2_0',
        'N07_lam_list2_2', 'N08_w_stage1_3', 'N09_two_stage', 'N11_seeds20']
MEMBERS = [INC] + ARMS
NAMED = {'INC_alone': [INC],
         'ALL11': MEMBERS,
         'WIDTH': [INC, 'N01_wide_128_32_64', 'N02_narrow_32_8_16'],
         'REGUL': [INC, 'N03_wd_1e-1', 'N04_wd_1e-3'],
         'LOSS': [INC, 'N06_lam_list2_0', 'N07_lam_list2_2', 'N08_w_stage1_3'],
         'SCHED': [INC, 'N11_seeds20'],
         'TOP3': [INC, 'N06_lam_list2_0', 'N01_wide_128_32_64', 'N05_lr_2e-3']}
GREEDY_MAX = 5


def rankmean(svecs, QIX, qpairs):
    """Average the within-pair ranks of the member score vectors, then reassign each pair's value multiset in that order.
    Magnitudes are identical to the plain mean; only the order inside a pair changes."""
    base = np.mean(svecs, axis=0)
    q = np.array([v[QIX] for v in svecs])                       # members x query rows
    rk = np.zeros(q.shape[1])
    _, first = np.unique(qpairs, return_index=True)
    first = np.sort(first)
    bounds = np.r_[first, len(qpairs)]
    for m in range(q.shape[0]):
        for i in range(len(first)):
            a, b = bounds[i], bounds[i + 1]
            seg = q[m, a:b]
            order = np.argsort(-seg, kind='mergesort')
            r = np.empty(b - a, float); r[order] = np.arange(b - a, dtype=float)
            rk[a:b] += r
    out = base.copy()
    qb = base[QIX]
    newq = np.empty_like(qb)
    for i in range(len(first)):
        a, b = bounds[i], bounds[i + 1]
        order = np.lexsort((np.arange(b - a), rk[a:b]))         # ascending mean rank = best first, index breaks ties
        newq[a + order] = np.sort(qb[a:b])[::-1]
    out[QIX] = newq
    return out


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
    assert 'amendment_before_measurement' in reg
    dest = OUT / 'results185.json'
    assert not dest.exists(), 'preserve a completed read'
    prot = check_protected()

    D = S165.load()
    d, Q = D['d'], D['Q']
    meta = json.loads((CACHE / 'meta172a.json').read_text(encoding='utf-8'))
    V = np.asarray(np.load(CACHE / 'views41.npy', mmap_mode='r'))
    assert S165.sha(CACHE / 'views41.npy') == meta['views41_sha256']
    idx = {f: np.array(v) for f, v in meta['family_column_index'].items()}
    y, fam = Q['relevance'].to_numpy().astype(int), Q['behavior_family'].to_numpy()
    qpairs = Q['pair_id'].to_numpy()
    stage2 = dict(S172B.INCUMBENT)

    # ---- load members, gate shapes and index arrays, confirm the two exclusions ----
    NZ = {}
    gates = {'excluded': {}}
    for n in PARTS:
        base = dict(np.load(CACHE / f'nn_{n}.npz'))
        assert S165.sha(CACHE / f'nn_{n}.npz') == meta['partitions'][n]['sha256']
        NZ[(INC, n)] = base
        for a in ARMS:
            z = dict(np.load(NET / f'nn_{a}_{n}.npz'))
            assert z['EA'].shape == base['EA'].shape and z['EB'].shape == base['EB'].shape, (a, n)
            assert np.array_equal(np.isnan(z['EA']), np.isnan(base['EA'])) and np.isfinite(z['EB']).all() == np.isfinite(base['EB']).all()
            for k in range(5):
                assert np.array_equal(z[f'trq{k}'], base[f'trq{k}']) and np.array_equal(z[f'teq{k}'], base[f'teq{k}'])
            NZ[(a, n)] = z
    for a, why in (('N10_patience80_epochs800', 'bit-identical duplicate of the incumbent'), ('F_medium_features', '11 feature columns, not 7')):
        z = dict(np.load(NET / f'nn_{a}_P0.npz'))
        b = NZ[(INC, 'P0')]
        gates['excluded'][a] = {'reason': why, 'ncols': int(z['EA'].shape[2]),
                                'maxabs_vs_incumbent_EA': float(np.nanmax(np.abs(z['EA'][:, :, :7] - b['EA']))),
                                'maxabs_vs_incumbent_EB': float(np.abs(z['EB'][:, :7] - b['EB']).max())}
    assert gates['excluded']['N10_patience80_epochs800']['maxabs_vs_incumbent_EA'] == 0.0

    # ---- per-member v1 score vectors (36 calls), with the step165b gate on the incumbent ----
    S = {}
    ref = pl.read_parquet(CV / 'experiments/e2e_evidence_C_20260915/audit_scores165b.parquet').sort(S165.K)
    for m in MEMBERS:
        for n in PARTS:
            S[(m, n)] = S172B.v1_full(stage2, V, idx, NZ[(m, n)], y, fam, D['SS'], D['QIX'])
        print(f'scored member {m} {round(time.perf_counter() - t0)}s', flush=True)
    gates['incumbent_reproduces_step165b_v1LP'] = float(np.abs(S[(INC, 'P0')] - ref['v1LP'].to_numpy()).max())
    assert gates['incumbent_reproduces_step165b_v1LP'] == 0.0

    # ---- metric plumbing ----
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabd = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptabd.values()))
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabd[q]) for q in qids])

    def permap(s_by_part):
        """per-query AP@5, averaged over the three partitions."""
        acc = np.zeros(len(qids))
        for n in PARTS:
            _, m = summarize(key.with_columns(pl.Series('score', D['dec'](s_by_part[n]))), truths, fams)
            acc += np.array([m[q] for q in qids])
        return acc / len(PARTS)

    CACHED = {}

    def evaluate(members, combiner):
        ck = (combiner, tuple(sorted(members)))
        if ck in CACHED:
            return CACHED[ck]
        out = {}
        for n in PARTS:
            if combiner == 'FEAT_mean':
                if len(members) == 1:
                    nz = NZ[(members[0], n)]
                else:
                    b = NZ[(members[0], n)]
                    nz = {'EA': np.mean([NZ[(m, n)]['EA'] for m in members], axis=0),
                          'EB': np.mean([NZ[(m, n)]['EB'] for m in members], axis=0),
                          **{k: v for k, v in b.items() if k.startswith(('trq', 'teq'))}}
                out[n] = S172B.v1_full(stage2, V, idx, nz, y, fam, D['SS'], D['QIX'])
            else:
                vecs = [S[(m, n)] for m in members]
                out[n] = np.mean(vecs, axis=0) if combiner == 'SCORE_mean' else rankmean(vecs, D['QIX'], qpairs)
        CACHED[ck] = permap(out)
        return CACHED[ck]

    A0 = evaluate([INC], 'SCORE_mean')
    print(f'incumbent mean AP@5 {A0.mean():.6f}  {round(time.perf_counter() - t0)}s', flush=True)
    R = lambda x: round(float(x), 6)

    # ---- named compositions x three combiners ----
    table = {}
    for cname, mem in NAMED.items():
        for comb in ('FEAT_mean', 'SCORE_mean', 'SCORE_rankmean'):
            if len(mem) == 1 and comb != 'SCORE_mean':
                continue
            A = evaluate(mem, comb)
            table[f'{cname}|{comb}'] = {'members': mem, 'combiner': comb, 'n_members': len(mem),
                                        'mean_map5': R(A.mean()), 'delta': R(A.mean() - A0.mean()),
                                        'half0_delta': R((A - A0)[hv == 0].mean()), 'half1_delta': R((A - A0)[hv == 1].mean())}
            print(f'{cname:10s} {comb:14s} n={len(mem):2d} delta {A.mean() - A0.mean():+.6f} '
                  f'h0 {(A - A0)[hv == 0].mean():+.6f} h1 {(A - A0)[hv == 1].mean():+.6f}  {round(time.perf_counter() - t0)}s', flush=True)

    # ---- greedy forward selection, run separately on each half, scored on the other ----
    greedy = {}
    for h in (0, 1):
        on = hv == h
        for comb in ('FEAT_mean', 'SCORE_mean'):
            cur, best = [INC], evaluate([INC], 'SCORE_mean')[on].mean()
            path = [{'add': INC, 'sel_mean': R(best)}]
            for _ in range(GREEDY_MAX):
                cands = [(evaluate(cur + [a], comb)[on].mean(), a) for a in MEMBERS if a not in cur]
                v, a = max(cands)
                if v <= best + 1e-9:
                    break
                cur, best = cur + [a], v
                path.append({'add': a, 'sel_mean': R(best)})
            A = evaluate(cur, comb)
            greedy[f'h{h}|{comb}'] = {'members': cur, 'path': path, 'selection_half_delta': R((A - A0)[on].mean()),
                                      'held_half_delta': R((A - A0)[~on].mean()), 'held_half_n': int((~on).sum())}
            print(f'greedy h{h} {comb}: {cur} sel {(A - A0)[on].mean():+.6f} held {(A - A0)[~on].mean():+.6f}  {round(time.perf_counter() - t0)}s', flush=True)

    # ---- honest cross-half G over the whole candidate pool (named + greedy), then the all-162 final ----
    pool = {k: v['members'] for k, v in table.items()}
    combs = {k: v['combiner'] for k, v in table.items()}
    for k, v in greedy.items():
        pool[f'GREEDY_{k}'] = v['members']; combs[f'GREEDY_{k}'] = k.split('|')[1]
    held = {}
    for h in (0, 1):
        on = hv == h
        sel = max(pool, key=lambda k: evaluate(pool[k], combs[k])[on].mean())
        A = evaluate(pool[sel], combs[sel])
        held[h] = {'config': sel, 'members': pool[sel], 'combiner': combs[sel],
                   'selection_half_delta': R((A - A0)[on].mean()), 'held_half_delta': R((A - A0)[~on].mean()), 'held_half_n': int((~on).sum())}
    Gh = (held[0]['held_half_delta'] * held[0]['held_half_n'] + held[1]['held_half_delta'] * held[1]['held_half_n']) / len(qids)

    fin = max(pool, key=lambda k: evaluate(pool[k], combs[k]).mean())
    Af = evaluate(pool[fin], combs[fin])
    a_map = {q: A0[t] for t, q in enumerate(qids)}
    b_map = {q: Af[t] for t, q in enumerate(qids)}
    c = boot(a_map, b_map, ptabd, pools)
    c['by_family'] = {f[:2]: R(np.mean([b_map[q] - a_map[q] for q in qids if fams[q] == f])) for f in FAMS}
    er1 = bool(c['mean_delta'] > 0 and c['ci95'][0] > -.010 and min(c['by_family'].values()) >= -.020)
    proceed = bool(Gh > 0 and er1 and pool[fin] != [INC])
    build = bool(proceed and c['mean_delta'] >= .02)

    out = {'registration_sha256': reg_sha, 'author': 'claude', 'members': MEMBERS, 'gates': gates,
           'incumbent_mean_map5': R(A0.mean()), 'stage2_params': stage2,
           'table': dict(sorted(table.items(), key=lambda kv: -kv[1]['delta'])), 'greedy': greedy,
           'cross_half': held, 'G_honest': R(Gh), 'final_config': fin, 'final_members': pool[fin], 'final_combiner': combs[fin],
           'final_vs_incumbent': {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'by_family': c['by_family']},
           'ER1_final': er1, 'proceed': proceed,
           'build_bar': 'OBS 19.133: a build needs mean decoded delta >= +.02 so that even half of it clears the corrected public SE of .00065',
           'build_justified': build, 'wall_seconds': round(time.perf_counter() - t0, 1)}
    assert check_protected() == prot
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('FINAL', fin, 'G', R(Gh), 'delta', R(c['mean_delta']), 'ER1', er1, 'PROCEED', proceed, 'BUILD', build, flush=True)


if __name__ == '__main__':
    main()
