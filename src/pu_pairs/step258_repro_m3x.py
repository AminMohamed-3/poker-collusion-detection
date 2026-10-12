"""STEP258 step 1: reproduce the production M3x block blend exactly before anything else.

Gates (all must hold, nothing downstream runs otherwise):
  G_fit_M3   final_fit(testbed(), FS_M3) reproduces experiments/viewdis_20260911/multiview_M3_fit.json
  G_fold     dev_views32 fold-match max abs difference == 0.0
  G_fit_M3x  final_fit(dev_views32(testbed()), FS_M3) reproduces outputs_step156_20260915/receipt_step156a.json M3x_fit
  G_ap_M3x   the cross-fold def-A / def-C AP of the incumbent cell reproduce
             outputs_step156_20260915/block_screen.json A.M3x32 / C.M3x32 exactly (the step177/191 gate)

Also caches the 32-draw testbed frame (parquet) so the later STEP258 scripts do not refit the views.
Writes experiments/step258_mech_block_view_20260918/repro258.json.  Reads only; never uploads.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, time
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs.step64f_M3_candidate import testbed, final_fit, FS_M3
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step60_multiview_blend import ap, logreg, predict
from pu_pairs.step62_multiview_candidate import DEFS
from pu_pairs.build_allpairs_fold import fold_pools
from pu_pairs.step156a_C_block_M3x import dev_views32

OUT = C + 'experiments/step258_mech_block_view_20260918/'
DEST = OUT + 'repro258.json'
CACHE = OUT + 'testbed32.parquet'
L2 = 10.0


def groups(d):
    """f1 = fold-1-only tables, f2 = fold-2-only tables, sh = tables in both pools (step177)."""
    f1, f2 = set(fold_pools(1)), set(fold_pools(2))
    shared = f1 & f2
    tab = d['table_id'].to_numpy()
    in1 = np.array([t in f1 and t not in shared for t in tab])
    in2 = np.array([t in f2 and t not in shared for t in tab])
    insh = np.array([t in shared for t in tab])
    assert (in1 | in2 | insh).all()
    return in1, in2, insh


def crossfold(X, y, cert_train, in1, in2, insh):
    """step177 protocol verbatim: the fold-1-only fit scores fold-2-only + shared tables,
    the fold-2-only fit scores fold-1-only tables."""
    pred = np.full(X.shape[0], np.nan)
    m1 = logreg(X[in1 & cert_train], y[in1 & cert_train], L2)
    pred[in2 | insh] = predict(m1, X[in2 | insh])
    m2 = logreg(X[in2 & cert_train], y[in2 & cert_train], L2)
    pred[in1] = predict(m2, X[in1])
    assert np.isfinite(pred).all()
    return pred, m1, m2


def score_pred(d, pred):
    in1, in2, insh = groups(d)
    G = {'f1': in1, 'f2sh': in2 | insh}
    out = {}
    for D in ('A', 'C'):
        cert = d['cls'].is_in(DEFS[D][0] + DEFS[D][1]).to_numpy()
        y = d['cls'].is_in(DEFS[D][0]).to_numpy().astype(float)
        out[f'{D}_pooled'] = ap(y[cert], pred[cert])
        for g, m in G.items():
            out[f'{D}_{g}'] = ap(y[cert & m], pred[cert & m])
    return out


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    d0 = testbed()
    m3, info3 = final_fit(d0, FS_M3)
    stored3 = json.load(open(C + 'experiments/viewdis_20260911/multiview_M3_fit.json', encoding='utf-8'))
    g_fit_m3 = all(abs(info3['weights_standardised'][v] - stored3['weights_standardised'][v]) < 1e-9 for v in FS_M3) \
        and info3['bias'] == stored3['bias']
    assert g_fit_m3, (info3['weights_standardised'], stored3['weights_standardised'])

    d, gate_fold = dev_views32(d0)
    assert gate_fold == 0.0, gate_fold
    mx, infox = final_fit(d, FS_M3)
    storedx = json.load(open(C + 'outputs_step156_20260915/receipt_step156a.json', encoding='utf-8'))['M3x_fit']
    g_fit_m3x = all(abs(infox['weights_standardised'][v] - storedx['weights_standardised'][v]) < 1e-9 for v in FS_M3) \
        and infox['bias'] == storedx['bias'] and infox['mu'] == storedx['mu'] and infox['sd'] == storedx['sd'] \
        and infox['AP_in_sample'] == storedx['AP_in_sample']
    assert g_fit_m3x, (infox, storedx)

    in1, in2, insh = groups(d)
    X = X_of(d, FS_M3, 'risk_prefix', 'R1_prefix', 'risk_noT')
    cert = {D: d['cls'].is_in(DEFS[D][0] + DEFS[D][1]).to_numpy() for D in ('A', 'C')}
    ypos = {D: d['cls'].is_in(DEFS[D][0]).to_numpy().astype(float) for D in ('A', 'C')}
    # step177 cells: the A gate is the def-A-trained cell, the C gate the def-C-trained cell
    got = {}
    for R in ('A', 'C'):
        pr, _, _ = crossfold(X, ypos[R], cert[R], in1, in2, insh)
        got[R] = score_pred(d, pr)
    ref = json.load(open(C + 'outputs_step156_20260915/block_screen.json', encoding='utf-8'))
    g_ap = {}
    for D in ('A', 'C'):
        for k, rk in (('pooled', 'pooled'), ('f1', 'test_f1'), ('f2sh', 'test_f2sh')):
            g_ap[f'{D}_{k}'] = (got[D][f'{D}_{k}'] == ref[D]['M3x32'][rk])
    assert all(g_ap.values()), (got, {D: ref[D]['M3x32'] for D in ('A', 'C')})
    incumbent = got['A']          # the production cell: def-A training rows, def C as a guard
    certA, yA = cert['A'], ypos['A']

    d.write_parquet(CACHE)
    out = {'step': 258, 'what': 'M3x reproduction gate',
           'G_fit_M3_reproduces_multiview_M3_fit': bool(g_fit_m3),
           'G_fold_match_maxdiff': float(gate_fold),
           'G_fit_M3x_reproduces_receipt_step156a': bool(g_fit_m3x),
           'G_ap_M3x_reproduces_block_screen': {k: bool(v) for k, v in g_ap.items()},
           'incumbent_M3x_crossfold_defA_trained': {k: float(v) for k, v in incumbent.items()},
           'defC_trained_cell_crossfold': {k: float(v) for k, v in got['C'].items()},
           'block_screen_reference': {D: ref[D]['M3x32'] for D in ('A', 'C')},
           'M3x_fit': infox,
           'n_rows': int(d.height),
           'n_certA': int(certA.sum()), 'n_posA': int(yA[certA].sum()),
           'group_sizes': {'f1': int(in1.sum()), 'f2only': int(in2.sum()), 'shared': int(insh.sum())},
           'testbed_cache': CACHE, 'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(DEST, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'M3x_fit'}, indent=1), flush=True)
    print('ALL REPRODUCTION GATES PASS')


if __name__ == '__main__':
    main()
