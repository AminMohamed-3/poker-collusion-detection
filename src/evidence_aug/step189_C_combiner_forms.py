"""STEP189: order-only combiner forms over the cached STEP186 grid.

STEP186 fixed the combiner to a plain mean and searched over WHICH cells to include.
This asks the orthogonal question with the block held fixed: does a different ORDER-ONLY
combiner beat the plain mean?

Every arm reassigns the plain mean's own per-query value multiset after reordering, so the
structural decoder sees an identical magnitude distribution and only the order differs.
That keeps the whole study inside the OBS 19.138 "model quality" transfer class (+0.90)
rather than the "calibration / loss shape" class (-0.72).

Non-query rows are copied verbatim from the mean and never touched: v1LP carries a sentinel
near -1,000,005.167, and the STEP186 float32 failure was exactly a sentinel being quantised.
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
OUT = CV / 'experiments/net_ensemble_20260916'
REG = OUT / 'registration189.json'
PARTS = ['P0', 'P1651', 'P1652']
INC = 'INC'
ARMS = ['N01_wide_128_32_64', 'N02_narrow_32_8_16', 'N03_wd_1e-1', 'N04_wd_1e-3', 'N05_lr_2e-3', 'N06_lam_list2_0',
        'N07_lam_list2_2', 'N08_w_stage1_3', 'N09_two_stage', 'N11_seeds20']
S1 = [INC] + ARMS
BAGS = ['C00_incumbent', 'A01_ff50_bag', 'A02_ff50_ss80_bag', 'A06_l2reg5_ff30_bag']
NONBAG = ['A03_300t_lr025', 'A04_clf2_l15_d5_300t', 'A05_l3_d2_300t']
S2 = BAGS[:1] + ['A01_ff50_bag', 'A02_ff50_ss80_bag'] + NONBAG + ['A06_l2reg5_ff30_bag']

BLOCKS = {
    'BAGS44': [(m, c) for m in S1 for c in BAGS],
    'FULL77': [(m, c) for m in S1 for c in S2],
    'C00_11': [(m, 'C00_incumbent') for m in S1],
    'NONBAG33': [(m, c) for m in S1 for c in NONBAG],
}
COMBINERS = ['MEAN', 'RANKMEAN', 'ZMEAN', 'TRIM1', 'MEDIAN']


def segments(qpairs):
    """Contiguous [a, b) row ranges, one per query pair, in the order the rows appear."""
    _, first = np.unique(qpairs, return_index=True)
    first = np.sort(first)
    return np.r_[first, len(qpairs)]


def reorder(base, QIX, bounds, key):
    """Give base[QIX] a new within-pair order: smallest key gets the largest value.

    The value multiset inside every pair is untouched, so this cannot be a calibration change.
    key = -base reproduces base bit-for-bit, which is gate IDENT.
    """
    out = base.copy()
    qb = base[QIX]
    newq = np.empty_like(qb)
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        order = np.lexsort((np.arange(b - a), key[a:b]))
        newq[a + order] = np.sort(qb[a:b])[::-1]
    out[QIX] = newq
    return out


def _within_pair_rank(seg):
    order = np.argsort(-seg, kind='mergesort')
    r = np.empty(len(seg), float)
    r[order] = np.arange(len(seg), dtype=float)
    return r


def combine(name, svecs, QIX, bounds):
    """svecs: list of cell score vectors. Returns one score vector."""
    base = np.mean(svecs, axis=0)
    if name == 'MEAN':
        return base
    q = np.array([v[QIX] for v in svecs])                     # cells x query rows
    if name == 'RANKMEAN':
        rk = np.zeros(q.shape[1])
        for m in range(q.shape[0]):
            for i in range(len(bounds) - 1):
                a, b = bounds[i], bounds[i + 1]
                rk[a:b] += _within_pair_rank(q[m, a:b])
        key = rk
    elif name == 'ZMEAN':
        z = np.empty_like(q)
        for m in range(q.shape[0]):
            for i in range(len(bounds) - 1):
                a, b = bounds[i], bounds[i + 1]
                seg = q[m, a:b]
                sd = seg.std()
                z[m, a:b] = (seg - seg.mean()) / sd if sd > 0 else 0.0
        key = -z.mean(axis=0)
    elif name == 'TRIM1':
        if q.shape[0] <= 2:
            raise ValueError('TRIM1 needs >2 cells')
        s = np.sort(q, axis=0)
        key = -s[1:-1].mean(axis=0)
    elif name == 'MEDIAN':
        key = -np.median(q, axis=0)
    else:
        raise ValueError(name)
    return reorder(base, QIX, bounds, key)


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[1]
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    assert S165.sha(REG) == reg_sha, 'registration hash mismatch'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert reg['n_arms'] == len(BLOCKS) * len(COMBINERS)
    dest = OUT / 'results189.json'
    assert not dest.exists(), 'preserve a completed read'
    prot = check_protected()

    z = np.load(OUT / 'grid186.npz')
    SV = {}
    for k in z.files:
        m, c, n = k.split('~')
        SV[(m, c, n)] = z[k]
    print(f'loaded {len(SV)} cached cells', flush=True)

    D = S165.load()
    d, Q = D['d'], D['Q']
    QIX = D['QIX']
    qpairs = Q['pair_id'].to_numpy()
    bounds = segments(qpairs)
    print(f'{len(bounds) - 1} query pairs over {len(qpairs)} query rows', flush=True)

    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabd = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabd[q]) for q in qids])
    R = lambda x: round(float(x), 6)

    ref = pl.read_parquet(CV / 'experiments/e2e_evidence_C_20260915/audit_scores165b.parquet').sort(S165.K)
    gate_inc = float(np.abs(SV[(INC, 'C00_incumbent', 'P0')] - ref['v1LP'].to_numpy()).max())
    print(f'GATE incumbent cell vs step165b v1LP: {gate_inc}', flush=True)
    assert gate_inc == 0.0

    CACHED = {}

    def evaluate(cells, comb):
        ck = (tuple(sorted(cells)), comb)
        if ck in CACHED:
            return CACHED[ck]
        acc = np.zeros(len(qids))
        for n in PARTS:
            svecs = [SV[(m, c, n)] for m, c in cells]
            s = combine(comb, svecs, QIX, bounds)
            if comb != 'MEAN':
                base = np.mean(svecs, axis=0)
                nq = np.ones(len(s), bool)
                nq[QIX] = False
                assert np.array_equal(s[nq], base[nq]), f'{comb} touched non-query rows'
            _, mm = summarize(key.with_columns(pl.Series('score', D['dec'](s))), truths, fams)
            acc += np.array([mm[q] for q in qids])
        CACHED[ck] = acc / len(PARTS)
        return CACHED[ck]

    A0 = evaluate([(INC, 'C00_incumbent')], 'MEAN')
    print(f'incumbent cell mean AP@5 {A0.mean():.6f}', flush=True)

    # ---- gate IDENT: the reorder machinery with key = -base must return base bit-for-bit ----
    tst = [SV[(m, c, 'P0')] for m, c in BLOCKS['BAGS44']]
    b = np.mean(tst, axis=0)
    ident = reorder(b, QIX, bounds, -b[QIX])
    gate_ident = float(np.abs(ident - b).max())
    print(f'GATE IDENT (reorder by -base == base): {gate_ident}', flush=True)
    assert gate_ident == 0.0

    # ---- gate: MEAN reproduces STEP186 ----
    r186 = json.loads((OUT / 'results186.json').read_text(encoding='utf-8'))
    g186 = {}
    for bn, s186 in [('BAGS44', 'S1_ALL11_x_S2_BAGS'), ('FULL77', 'FULL77'), ('C00_11', 'S1_ALL11_at_C00')]:
        A = evaluate(BLOCKS[bn], 'MEAN')
        g186[bn] = R(A.mean() - A0.mean() - r186['table'][s186]['delta'])
    print(f'GATE vs STEP186 deltas (all 0.0): {json.dumps(g186)}', flush=True)
    assert all(v == 0.0 for v in g186.values()), g186

    # ---- the 20 arms ----
    table = {}
    for bn, cells in BLOCKS.items():
        for comb in COMBINERS:
            A = evaluate(cells, comb)
            nm = f'{bn}|{comb}'
            # multiset gate: per-pair value multiset must equal MEAN's
            if comb != 'MEAN':
                sv = [SV[(m, c, 'P0')] for m, c in cells]
                mv = np.mean(sv, axis=0)
                cv = combine(comb, sv, QIX, bounds)
                ms = float(np.abs(np.sort(mv[QIX]) - np.sort(cv[QIX])).max())
                assert ms == 0.0, (nm, ms)
            table[nm] = {'block': bn, 'combiner': comb, 'n_cells': len(cells),
                         'mean_map5': R(A.mean()), 'delta': R(A.mean() - A0.mean()),
                         'half0_delta': R((A - A0)[hv == 0].mean()), 'half1_delta': R((A - A0)[hv == 1].mean())}
            print(f'{nm:20s} n={len(cells):3d} delta {A.mean() - A0.mean():+.6f} '
                  f'h0 {(A - A0)[hv == 0].mean():+.6f} h1 {(A - A0)[hv == 1].mean():+.6f}  '
                  f'{round(time.perf_counter() - t0)}s', flush=True)

    # ---- honest cross-half over all 20 arms ----
    held = {}
    for h in (0, 1):
        on = hv == h
        best = max(table, key=lambda k: (evaluate(BLOCKS[table[k]['block']], table[k]['combiner']) - A0)[on].mean())
        A = evaluate(BLOCKS[table[best]['block']], table[best]['combiner'])
        held[f'h{h}'] = {'picked_on_half': best, 'selection_half_delta': R((A - A0)[on].mean()),
                         'held_half_delta': R((A - A0)[~on].mean()), 'held_half_n': int((~on).sum())}
        print(f'cross-half h{h}: picked {best} sel {(A - A0)[on].mean():+.6f} '
              f'held {(A - A0)[~on].mean():+.6f}', flush=True)
    G = float(np.average([held['h0']['held_half_delta'], held['h1']['held_half_delta']],
                         weights=[held['h0']['held_half_n'], held['h1']['held_half_n']]))
    print(f'G (query-weighted mean of held-half deltas) = {G:+.6f}', flush=True)

    # ---- the registered build question: does any alternative beat MEAN on BOTH halves, per block? ----
    verdict = {}
    for bn in BLOCKS:
        mean_row = table[f'{bn}|MEAN']
        wins = [c for c in COMBINERS if c != 'MEAN'
                and table[f'{bn}|{c}']['half0_delta'] > mean_row['half0_delta']
                and table[f'{bn}|{c}']['half1_delta'] > mean_row['half1_delta']]
        verdict[bn] = {'mean_delta': mean_row['delta'], 'beats_mean_on_both_halves': wins}
    print('per-block verdict: ' + json.dumps(verdict), flush=True)
    build_comb = 'MEAN'
    bags_wins = verdict['BAGS44']['beats_mean_on_both_halves']
    if bags_wins:
        build_comb = max(bags_wins, key=lambda c: table[f'BAGS44|{c}']['delta'])
    print(f'COMBINER FOR THE BUILD: {build_comb}', flush=True)

    res = {'step': 189, 'registration_sha256': reg_sha, 'seconds': round(time.perf_counter() - t0, 1),
           'gates': {'incumbent_cell_vs_v1LP': gate_inc, 'ident_reorder': gate_ident, 'vs_step186': g186,
                     'protected': prot},
           'incumbent_cell_mean_map5': R(A0.mean()), 'table': table, 'cross_half': held, 'G': R(G),
           'verdict': verdict, 'combiner_for_the_build': build_comb}
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'wrote {dest}  {round(time.perf_counter() - t0)}s', flush=True)


if __name__ == '__main__':
    main()
