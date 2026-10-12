"""STEP258b: band-restricted re-sort of the incumbent M3x testbed order (registration258b.json).

POST HOC with respect to the STEP258 band table (declared in registration258b.json).

Every arm score is the CROSS-FOLD prediction of the pre-registered STEP258 run, recomputed here from
the same cached inputs with the same step177 protocol -- nothing is refitted in-sample. A band set S
keeps the positions it occupies in the frozen M3x order and those positions are handed out to the
pairs of S in descending order of the arm's score; every pair outside S keeps its M3x position.

Bands by frozen prefix risk: cd = [.5,.9), b = [.9,.99).  Variants: cd, b, cd+b.
Arms: MECH6, PERM6, PLUS6, MECH6-noT6 (diagnostic), TOP3KEY (no fit at all).
Readouts: def-A pooled / f1 / f2sh, def-C pooled, band positive-negative counts, and a paired
stratified bootstrap (2,000 draws, same draws for every arm) 90 % interval of the pooled def-A delta.

Writes results258b.json.  Development phase only; never uploads.
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
DEST = OUT + 'results258b.json'
NBOOT = 2000
BSEED = 2058


def resort(base_pred, arm_score, S):
    """Return a score whose order equals the M3x order with the band S re-sorted by arm_score."""
    n = len(base_pred)
    order = np.argsort(-base_pred, kind='stable')          # frozen M3x order (indices, best first)
    pos = np.empty(n, np.int64)
    pos[order] = np.arange(n)
    idx = np.flatnonzero(S)
    slots = np.sort(pos[idx])                              # the positions the band occupies
    new_order_of_band = idx[np.argsort(-arm_score[idx], kind='stable')]
    newpos = pos.copy()
    newpos[new_order_of_band] = slots
    assert len(set(newpos.tolist())) == n
    return -newpos.astype(float)


def main():
    t0 = time.time()
    assert not os.path.exists(DEST), 'preserve a completed run'
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

    scores = {}
    scores['M3x'], _, _ = crossfold(X0, ypos['A'], cert['A'], in1, in2, insh)
    for nm, X in (('MECH6', np.column_stack([X0, B])),
                  ('PERM6', np.column_stack([X0, block6(agg, 'perm')])),
                  ('PLUS6', np.column_stack([X0, block6(agg, 'plus')])),
                  ('MECH6-noT6', np.column_stack([X0, B - N]))):
        scores[nm], _, _ = crossfold(X, ypos['A'], cert['A'], in1, in2, insh)
    scores['TOP3KEY'] = agg['mechplus_top3_logit'].to_numpy().astype(float)   # no fit at all

    # gate: the recomputed arm scores must reproduce the STEP258 / posthoc AP numbers exactly
    ref = json.load(open(OUT + 'results258.json', encoding='utf-8'))
    refp = json.load(open(OUT + 'results258_posthoc.json', encoding='utf-8'))
    chk = {}
    for nm, key in (('M3x', 'M3x'), ('MECH6', 'M3x+MECH6'), ('PERM6', 'M3x+PERM6'), ('PLUS6', 'M3x+PLUS6')):
        got = round(float(ap(ypos['A'][cert['A']], scores[nm][cert['A']])), 6)
        chk[nm] = (got == round(ref['arms'][key]['A_pooled'], 6))
    got = round(float(ap(ypos['A'][cert['A']], scores['MECH6-noT6'][cert['A']])), 6)
    chk['MECH6-noT6'] = (got == round(refp['incumbent']['A_pooled']
                                      + refp['delta_vs_M3x']['PH1 M3x+(MECH6-noT6)']['A_pooled'], 6))
    assert all(chk.values()), chk

    risk = d['risk_prefix'].to_numpy()
    bands = {'cd': (risk >= .5) & (risk < .9), 'b': (risk >= .9) & (risk < .99)}
    stored = d['band'].to_numpy() if 'band' in d.columns else None
    band_col_check = None
    if stored is not None:
        band_col_check = {'stored_band_column_is_risk_prefix_based':
                          bool((stored[bands['cd']] == '[.5,.9)').all() and (stored[bands['b']] == '[.9,.99)').all()),
                          'stored_counts': {k: int((stored == k).sum()) for k in ('[.99,1]', '[.9,.99)', '[.5,.9)')},
                          'risk_prefix_counts': {'[.5,.9)': int(bands['cd'].sum()), '[.9,.99)': int(bands['b'].sum()),
                                                 '[.99,1]': int((risk >= .99).sum())}}
    variants = {'cd': bands['cd'], 'b': bands['b'], 'cd+b': bands['cd'] | bands['b']}

    def readouts(pred):
        R = {}
        for D in ('A', 'C'):
            R[f'{D}_pooled'] = float(ap(ypos[D][cert[D]], pred[cert[D]]))
            for g, m in G.items():
                R[f'{D}_{g}'] = float(ap(ypos[D][cert[D] & m], pred[cert[D] & m]))
        return R

    base = readouts(scores['M3x'])

    # ---- one shared set of bootstrap draws (paired across arms), stratified by cross-fold group
    rng = np.random.default_rng(BSEED)
    certA = cert['A']
    rowsA = np.flatnonzero(certA)
    strata = [rowsA[in1[rowsA]], rowsA[(in2 | insh)[rowsA]]]
    draws = [np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata]) for _ in range(NBOOT)]
    yA = ypos['A']

    def boot_delta(pred_arm, pred_base):
        out = np.empty(NBOOT)
        for i, ix in enumerate(draws):
            yy = yA[ix]
            if yy.sum() == 0 or yy.sum() == len(yy):
                out[i] = 0.0
                continue
            out[i] = ap(yy, pred_arm[ix]) - ap(yy, pred_base[ix])
        return out

    res = {'incumbent_M3x': {k: round(v, 6) for k, v in base.items()},
           'band_sizes': {v: {'n_pairs': int(m.sum()),
                              'n_certA': int((m & certA).sum()),
                              'n_certA_pos': int(yA[m & certA].sum()),
                              'n_certA_neg': int((m & certA).sum() - yA[m & certA].sum())}
                          for v, m in variants.items()},
           'band_column_check': band_col_check,
           'gate_arm_scores_replay_step258': {k: bool(v) for k, v in chk.items()},
           'arms': {}}

    for arm in ('MECH6', 'PERM6', 'PLUS6', 'MECH6-noT6', 'TOP3KEY'):
        res['arms'][arm] = {}
        for vn, S in variants.items():
            pred = resort(scores['M3x'], scores[arm], S)
            R = readouts(pred)
            bd = boot_delta(pred, scores['M3x'])
            lo, hi = np.quantile(bd, [.05, .95])
            res['arms'][arm][vn] = {
                'levels': {k: round(v, 6) for k, v in R.items()},
                'delta': {k: round(R[k] - base[k], 6) for k in base},
                'bootstrap_defA_pooled_delta': {'mean': round(float(bd.mean()), 6),
                                                'p05': round(float(lo), 6), 'p95': round(float(hi), 6),
                                                'share_gt_0': round(float((bd > 0).mean()), 4),
                                                'n_draws': NBOOT}}
            print(f'{arm:12s} {vn:5s} dA_pooled {R["A_pooled"] - base["A_pooled"]:+.6f} '
                  f'f1 {R["A_f1"] - base["A_f1"]:+.6f} f2sh {R["A_f2sh"] - base["A_f2sh"]:+.6f} '
                  f'dC {R["C_pooled"] - base["C_pooled"]:+.6f} boot90 '
                  f'[{lo:+.6f},{hi:+.6f}]', flush=True)

    g = res['arms']['MECH6']['cd']
    gate = {'c1_pooled_ge_.002': bool(g['delta']['A_pooled'] >= .002),
            'c2_both_directions_gt0': bool(g['delta']['A_f1'] > 0 and g['delta']['A_f2sh'] > 0),
            'c3_defC_pooled_ge0': bool(g['delta']['C_pooled'] >= 0),
            'c4_boot_p05_gt_-.001': bool(g['bootstrap_defA_pooled_delta']['p05'] > -.001)}
    gate['PASS'] = bool(all(gate.values()))
    pm = res['arms']['PERM6']['cd']['delta']['A_pooled']
    pl_ = res['arms']['PLUS6']['cd']['delta']['A_pooled']
    res['gate_G258b'] = gate
    res['control_reading'] = {
        'MECH6_cd_pooled_delta': g['delta']['A_pooled'], 'PERM6_cd_pooled_delta': pm,
        'PLUS6_cd_pooled_delta': pl_,
        'controls_match_MECH6': bool(max(pm, pl_) >= g['delta']['A_pooled'] - .0005),
        'note': ('If controls_match_MECH6 is true the result reads as "generic event-ness helps '
                 'inside cd", not as "the STEP254 mechanical block helps".')}
    res['POST_HOC'] = 'band chosen after seeing the STEP258 band table; see registration258b.json'
    res['seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(DEST, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({'gate_G258b': gate, 'control_reading': res['control_reading']}, indent=1))
    print('wrote', DEST)


if __name__ == '__main__':
    main()
