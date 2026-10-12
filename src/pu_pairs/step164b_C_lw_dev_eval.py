"""Step 164b (Claude): registered dev reading of outputs_step164_20260915/prediction_step164.json. Base stack = FINAL2_16 of STEP163 (SXN3
SHORT3/ALL5/LIN30/LIN300/FAMMAX + STEP157 XGB3/ET/MLP + STEP163 MLP2/HGB5, draws k < 16). One arm: LW16 = the same stack with SXN3 ALL5, LIN30,
LIN300 and FAMMAX replaced by their length-weighted versions (step164a). LW16 replaces FINAL2_16 iff it passes the three conditions
((i) mean dPP90 > 0, (ii) every surface >= -.0005, (iii) mean dPPcls > 0). STEP131 surfaces, labels and AP as step159c / step163b.
Output experiments/length_weights_20260915/lw_dev_eval.json. Development labels only."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, hashlib
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs import step159c_C_combine_dev as S159

E = C + 'experiments/'
X, VD = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/'
S159.DIRS['ARCH2'] = (E + 'arch2_views_20260915/', 'arch2_dev_fold')
S159.DIRS['LW'] = (E + 'length_weights_20260915/', 'lw_dev_fold')
VT, NEW_VT, ARCH2_KEPT, LW_T = S159.VT, S159.NEW_VT, ['MLP2', 'HGB5'], ['ALL5', 'LIN30', 'LIN300', 'FAMMAX']


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(open(C + 'outputs_step164_20260915/prediction_step164.json', 'rb').read()).hexdigest() == reg_sha
    a2 = json.load(open(E + 'arch2_views_20260915/arch2_dev_eval.json', encoding='utf-8')); assert a2['kept_types'] == ARCH2_KEPT
    V = S159.type_means('SXN3', VT).join(S159.type_means('ARCH', NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    V = V.join(S159.type_means('ARCH2', ARCH2_KEPT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    V = V.join(S159.type_means('LW', LW_T), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    base = [f'SXN3_{v}' for v in VT] + [f'ARCH_{v}' for v in NEW_VT] + [f'ARCH2_{v}' for v in ARCH2_KEPT]
    arm = [f'LW_{v}' if v in LW_T else f'SXN3_{v}' for v in VT] + [f'ARCH_{v}' for v in NEW_VT] + [f'ARCH2_{v}' for v in ARCH2_KEPT]
    assert len(base) == len(arm) == 10
    R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    D = {}
    for p in ('prefix40', 'prefix60', 'prefix80'):
        parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
                 .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
                 .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
        D[p] = (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
                .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))

    def measure(cols):
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
    out = {'registration_sha256': reg_sha, 'base': base, 'arm': arm}
    b, a = measure(base), measure(arm)
    out['FINAL2_16_abs'], out['LW16_abs'] = b, a
    # descriptive only (no decision): each LW type swapped alone
    out['single_swaps_descriptive'] = {}
    for t in LW_T:
        one = [f'LW_{v}' if v == t else c for v, c in zip(VT + [None] * 5, base)]
        r1 = S159.rule({p: {'a': x, 'b': b[p]} for p, x in measure(one).items()}, 'a', 'b')
        out['single_swaps_descriptive'][t] = {k: v for k, v in r1.items() if k != 'passes'}
    r = S159.rule({p: {'a': a[p], 'b': b[p]} for p in a}, 'a', 'b')
    r['per_surface'] = {p: {m: a[p][m] - b[p][m] for m in a[p]} for p in a}
    out['LW16_minus_FINAL2_16'] = r
    out['LW_replaces'] = bool(r['passes'])
    out['prediction_vs_step163_file'] = {'point': .65 * .37 * r['mean_comp'] if r['passes'] else 0.0, 'band80_halfwidth': .0025 if r['passes'] else 0.0}
    json.dump(out, open(E + 'length_weights_20260915/lw_dev_eval.json', 'w', encoding='utf-8'), indent=1)
    print('LW16-FINAL2_16', json.dumps({k: v for k, v in r.items() if k != 'per_surface'}), 'REPLACES', out['LW_replaces'],
          'SINGLE', json.dumps(out['single_swaps_descriptive']), 'PRED', json.dumps(out['prediction_vs_step163_file']), flush=True)


if __name__ == '__main__':
    main()
