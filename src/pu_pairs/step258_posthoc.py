"""STEP258 POST HOC DIAGNOSTICS -- registered nowhere, cannot authorise any build, never uploaded.

G258 failed on both registered arms (results258.json).  These three cells answer ONE question the
failure raises: the registered secondary arm added the noT aggregates ALONGSIDE the all-hand ones and
the blend gave BOTH positive weights, so the spill-over discount the arm was meant to express never
appeared -- R1 pairs were promoted MORE, not less.  Does the CONTRAST express it?

  PH1  M3x + (MECH6 - MECH6noT)        the six differences only (the spill-over contrast)
  PH2  M3x + MECH6noT                  the six noT aggregates instead of the all-hand ones
  PH3  M3x + MECH6 + (MECH6 - noT6)    all-hand block plus the contrast

Everything else (rows, l2, cross-fold protocol, transform) is identical to the registered arms.
These numbers are POST HOC: they were computed after seeing the registered result and no build,
no public read and no future registration may treat them as evidence of a gain.
Writes results258_posthoc.json.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, time
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs.step64f_M3_candidate import FS_M3
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step60_multiview_blend import ap
from pu_pairs.step62_multiview_candidate import DEFS
from pu_pairs.step258_repro_m3x import groups, crossfold
from pu_pairs.step258_blend import block6, AGG

OUT = C + 'experiments/step258_mech_block_view_20260918/'
DEST = OUT + 'results258_posthoc.json'


def main():
    t0 = time.time()
    d = pl.read_parquet(OUT + 'testbed32.parquet')
    agg = pl.read_parquet(OUT + 'aggregates258.parquet')
    assert agg['pair_id'].to_list() == d['pair_id'].to_list()
    in1, in2, insh = groups(d)
    G = {'f1': in1, 'f2sh': in2 | insh}
    cert = {D: d['cls'].is_in(DEFS[D][0] + DEFS[D][1]).to_numpy() for D in ('A', 'C')}
    ypos = {D: d['cls'].is_in(DEFS[D][0]).to_numpy().astype(float) for D in ('A', 'C')}
    X0 = X_of(d, FS_M3, 'risk_prefix', 'R1_prefix', 'risk_noT')
    B = block6(agg, 'mechplus')
    N = block6(agg, 'mechplusnoT')
    R1 = d['R1_prefix'].to_numpy().astype(bool)
    risk = d['risk_prefix'].to_numpy()
    arms = {'M3x': X0,
            'PH1 M3x+(MECH6-noT6)': np.column_stack([X0, B - N]),
            'PH2 M3x+noT6': np.column_stack([X0, N]),
            'PH3 M3x+MECH6+(MECH6-noT6)': np.column_stack([X0, B, B - N])}
    res, preds = {}, {}
    for name, X in arms.items():
        pred, m1, m2 = crossfold(X, ypos['A'], cert['A'], in1, in2, insh)
        preds[name] = pred
        R = {}
        for D in ('A', 'C'):
            R[f'{D}_pooled'] = round(float(ap(ypos[D][cert[D]], pred[cert[D]])), 6)
            for g, m in G.items():
                R[f'{D}_{g}'] = round(float(ap(ypos[D][cert[D] & m], pred[cert[D] & m])), 6)
        for k, m in (('band[.5,.9)', (risk >= .5) & (risk < .9)), ('R1_pairs', R1)):
            sel = cert['A'] & m
            R[f'sub_{k}'] = round(float(ap(ypos['A'][sel], pred[sel])), 6)
        res[name] = R
        print(f'{name:28s} A_pooled {R["A_pooled"]:.6f} f1 {R["A_f1"]:.6f} f2sh {R["A_f2sh"]:.6f} '
              f'C_pooled {R["C_pooled"]:.6f}', flush=True)

    def rank_from_top(s):
        r = np.empty(len(s), float)
        r[np.argsort(-s, kind='stable')] = np.arange(len(s))
        return r
    r0 = rank_from_top(preds['M3x'])
    out = {'STATUS': 'POST HOC -- computed after the registered G258 verdict; cannot authorise a build',
           'incumbent': res['M3x'],
           'delta_vs_M3x': {n: {k: round(res[n][k] - res['M3x'][k], 6) for k in res['M3x']}
                            for n in arms if n != 'M3x'},
           'rank_moves': {n: {'mean_rank_move_R1': round(float((r0 - rank_from_top(preds[n]))[R1].mean()), 2),
                              'mean_rank_move_notR1': round(float((r0 - rank_from_top(preds[n]))[~R1].mean()), 2)}
                          for n in arms if n != 'M3x'},
           'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(DEST, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: out[k] for k in ('delta_vs_M3x', 'rank_moves')}, indent=1))


if __name__ == '__main__':
    main()
