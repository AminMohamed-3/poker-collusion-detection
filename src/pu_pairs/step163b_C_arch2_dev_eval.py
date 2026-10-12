"""Step 163b (Claude): registered dev reading of outputs_step163_20260915/prediction_step163.json. Base stack = FINAL16 of STEP159 (SXN3 five
types + STEP157 XGB3/ET/MLP, draws k < 16). Greedy in the fixed order MLP2, HGB5, LGBB3: a type joins the stack iff stack+type beats the current
stack under the three conditions ((i) mean dPP90 > 0, (ii) every surface >= -.0005, (iii) mean dPPcls > 0). STEP131 surfaces, labels and AP as
step159c. Output experiments/arch2_views_20260915/arch2_dev_eval.json. Development labels only."""
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
VT, NEW_VT, ORDER = S159.VT, S159.NEW_VT, ['MLP2', 'HGB5', 'LGBB3']


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(open(C + 'outputs_step163_20260915/prediction_step163.json', 'rb').read()).hexdigest() == reg_sha
    V = S159.type_means('SXN3', VT).join(S159.type_means('ARCH', NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    V = V.join(S159.type_means('ARCH2', ORDER), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    base = [f'SXN3_{v}' for v in VT] + [f'ARCH_{v}' for v in NEW_VT]
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
    out = {'registration_sha256': reg_sha, 'steps': []}
    stack = list(base); cur = measure(stack); out['FINAL16_abs'] = cur
    for t in ORDER:
        cand = measure(stack + [f'ARCH2_{t}'])
        r = S159.rule({p: {'a': cand[p], 'b': cur[p]} for p in cand}, 'a', 'b')
        r['type'] = t; r['per_surface'] = {p: {m: cand[p][m] - cur[p][m] for m in cand[p]} for p in cand}
        out['steps'].append(r)
        print(t, json.dumps({k: v for k, v in r.items() if k != 'per_surface'}), flush=True)
        if r['passes']:
            stack.append(f'ARCH2_{t}'); cur = cand
    fin = measure(stack)
    out['kept_types'] = [c.split('_', 1)[1] for c in stack if c.startswith('ARCH2_')]
    out['FINAL2_minus_FINAL16'] = S159.rule({p: {'a': fin[p], 'b': out['FINAL16_abs'][p]} for p in fin}, 'a', 'b')
    out['prediction_vs_step159_file'] = {'point': .65 * .37 * out['FINAL2_minus_FINAL16']['mean_comp'] if out['kept_types'] else 0.0, 'band80_halfwidth': .0025 if out['kept_types'] else 0.0}
    json.dump(out, open(E + 'arch2_views_20260915/arch2_dev_eval.json', 'w', encoding='utf-8'), indent=1)
    print('KEPT', out['kept_types'], 'FINAL2-FINAL16', json.dumps(out['FINAL2_minus_FINAL16']), 'PRED', json.dumps(out['prediction_vs_step159_file']), flush=True)


if __name__ == '__main__':
    main()
