"""STEP262 step 1 (FIRST GATE): reproduce the recorded dev reading of the PRODUCTION tail stack.

The production tail key is FINAL2 (the STEP163 stack; STEP168's arch3 round kept no type, so
FINAL2 is still the stack inside the current base file):

    FINAL2 = SXN3_{SHORT3,ALL5,LIN30,LIN300,FAMMAX} + ARCH_{XGB3,ET,MLP} + ARCH2_{MLP2,HGB5}

Recorded readings that must be reproduced to the printed precision:
    experiments/arch3_views_20260915/arch3_dev_eval.json  -> FINAL2_16_abs   (the STEP168 base stack)
    experiments/arch2_views_20260915/arch2_dev_eval.json  -> FINAL16_abs     (the STEP163 base stack)

Nothing is written by this module except experiments/step262_mech_tail_view_20260918/repro262.json.
It also exports the harness pieces the STEP262 arms reuse (stack_V / build_D / measure / BASE), so
every arm is measured by exactly this code path.  Development labels only.

Usage:  python step262_repro_tail.py
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '4')
import sys, json
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl                                        # noqa: E402
from pu_pairs import step159c_C_combine_dev as S159                     # noqa: E402

E = C + 'experiments/'
X, VD = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/'
OUT = E + 'step262_mech_tail_view_20260918/'
S159.DIRS['ARCH2'] = (E + 'arch2_views_20260915/', 'arch2_dev_fold')
S159.DIRS['ARCH3'] = (E + 'arch3_views_20260915/', 'arch3_dev_fold')
VT, NEW_VT = S159.VT, S159.NEW_VT
A2, A3, LG = ['MLP2', 'HGB5'], ['RF5', 'DART5', 'MLPS3'], ['MLP2', 'HGB5', 'LGBB3']
BASE = [f'SXN3_{v}' for v in VT] + [f'ARCH_{v}' for v in NEW_VT] + [f'ARCH2_{v}' for v in A2]
SURF = ('prefix40', 'prefix60', 'prefix80')


def stack_V(variant='arch3'):
    """The V frame of the recorded reading. 'arch3' = step168b's joins, 'arch2' = step163b's."""
    V = S159.type_means('SXN3', VT).join(S159.type_means('ARCH', NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    if variant == 'arch3':
        V = V.join(S159.type_means('ARCH2', A2), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
        V = V.join(S159.type_means('ARCH3', A3), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    elif variant == 'arch2':
        V = V.join(S159.type_means('ARCH2', LG), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    else:
        raise ValueError(variant)
    return V


def build_D(V, extra=None):
    """step163b / step168b verbatim.  `extra`: optional list of frames with pair_id/fold/surface + type
    columns, LEFT-joined onto V before the surface split (a new type must cover every D row)."""
    if extra:
        for f in extra:
            V = V.join(f, on=['pair_id', 'fold', 'surface'], how='left', validate='1:1')
    R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    D = {}
    for p in SURF:
        parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
                 .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
                 .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
        D[p] = (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
                .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))
    return D


def measure(D, cols):
    """step163b.measure verbatim."""
    S = {}
    for p, d in D.items():
        key = d.select(pl.mean_horizontal(cols)).to_series().to_numpy()
        assert np.isfinite(key).all()
        s, lab = d['risk'].to_numpy().astype(float), d['label'].to_numpy()
        fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
        yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
        yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
        sc = S159.permute(s, key, s < .5)
        S[p] = {'PP90': S159.ap(yp, sc), 'PPcls': S159.ap(yc, sc), 'RAW': S159.ap(yr, sc), 'B': S159.bmap(sc, lab, fam, pred)}
    return S


def keyed(D, cols):
    """The permuted score and the raw key per surface (used by the arms for rank-move diagnostics)."""
    out = {}
    for p, d in D.items():
        key = d.select(pl.mean_horizontal(cols)).to_series().to_numpy()
        s = d['risk'].to_numpy().astype(float)
        out[p] = (S159.permute(s, key, s < .5), key, d)
    return out


def cmp(got, want, tol=5e-12):
    bad = {}
    for p in want:
        for m in want[p]:
            dv = abs(got[p][m] - want[p][m])
            if dv > tol:
                bad[f'{p}.{m}'] = {'got': got[p][m], 'want': want[p][m], 'abs_diff': dv}
    return {'max_abs_diff': max((abs(got[p][m] - want[p][m]) for p in want for m in want[p]), default=0.0),
            'mismatches': bad, 'exact': not bad}


def main():
    os.makedirs(OUT, exist_ok=True)
    dest = OUT + 'repro262.json'
    assert not os.path.exists(dest), dest
    a2 = json.load(open(E + 'arch2_views_20260915/arch2_dev_eval.json', encoding='utf-8'))
    a3 = json.load(open(E + 'arch3_views_20260915/arch3_dev_eval.json', encoding='utf-8'))
    assert a3['kept_types'] == [], a3['kept_types']
    assert a2['kept_types'] == A2, a2['kept_types']
    rec = {'production_tail_stack': BASE, 'n_type_columns': len(BASE),
           'why': 'arch3_dev_eval.json kept_types == [] -> FINAL2 is the stack inside the current base file'}

    V3 = stack_V('arch3')
    D3 = build_D(V3)
    rec['rows_per_surface'] = {p: int(D3[p].height) for p in D3}
    rec['pairs_per_surface_tail'] = {p: int((D3[p]['risk'].to_numpy() < .5).sum()) for p in D3}
    got3 = measure(D3, BASE)
    rec['FINAL2_16_abs_reproduced'] = got3
    rec['gate_FINAL2_16_abs'] = cmp(got3, a3['FINAL2_16_abs'])
    print('FINAL2_16_abs', json.dumps(rec['gate_FINAL2_16_abs'])[:400], flush=True)

    V2 = stack_V('arch2')
    D2 = build_D(V2)
    got2 = measure(D2, [c for c in BASE if not c.startswith('ARCH2_')])
    rec['FINAL16_abs_reproduced'] = got2
    rec['gate_FINAL16_abs'] = cmp(got2, a2['FINAL16_abs'])
    print('FINAL16_abs', json.dumps(rec['gate_FINAL16_abs'])[:400], flush=True)
    got2f = measure(D2, BASE)
    rec['FINAL2_minus_FINAL16_reproduced'] = S159.rule({p: {'a': got2f[p], 'b': got2[p]} for p in got2f}, 'a', 'b')
    rec['FINAL2_minus_FINAL16_recorded'] = a2['FINAL2_minus_FINAL16']
    rec['gate_FINAL2_minus_FINAL16'] = {
        'max_abs_diff': max(abs(rec['FINAL2_minus_FINAL16_reproduced'][k] - a2['FINAL2_minus_FINAL16'][k])
                            for k in ('mean_dPP90', 'mean_dPPcls', 'mean_comp'))}
    rec['gate_passes'] = bool(rec['gate_FINAL2_16_abs']['exact'] and rec['gate_FINAL16_abs']['exact'])
    json.dump(rec, open(dest, 'w', encoding='utf-8'), indent=1)
    print('GATE', rec['gate_passes'], json.dumps(rec['rows_per_surface']), flush=True)


if __name__ == '__main__':
    main()
