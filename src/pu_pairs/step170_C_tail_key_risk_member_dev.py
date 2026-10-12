"""Step 170 (Claude): does the pair model's own risk help inside the tail key? Since STEP150 the tail (risk < .5) is ordered by view keys
only; risk itself is not a member. Arm K11 = (10 x FINAL2_16 key + lg(risk)) / 11, i.e. lg(risk) as an eleventh equal-weight member (no
weight search). Primary: tail permuted by K11 vs by the FINAL2_16 key, STEP131 surfaces (folds 1-2, prefix-40/60/80, shared_hands >= 38),
three conditions (ER3). Descriptive: both keys with the STEP169 N10_M50 boundary merge. Output experiments/tail_risk_member_20260915/k11_170.json.
Usage: REGSHA"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, hashlib
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs import step169_C_boundary_merge_dev as S169
S159 = S169.S159
REG = C + 'outputs_step170_20260915/prediction_step170.json'


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(open(REG, 'rb').read()).hexdigest() == reg_sha
    dest = S169.E + 'tail_risk_member_20260915/k11_170.json'; os.makedirs(os.path.dirname(dest), exist_ok=True); assert not os.path.exists(dest)
    V = S159.type_means('SXN3', S169.VT).join(S159.type_means('ARCH', S169.NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    V = V.join(S159.type_means('ARCH2', S169.A2), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    base = [f'SXN3_{v}' for v in S169.VT] + [f'ARCH_{v}' for v in S169.NEW_VT] + [f'ARCH2_{v}' for v in S169.A2]
    assert len(base) == 10
    V = V.with_columns(pl.mean_horizontal(base).alias('key'))
    R1 = pl.read_parquet(S169.VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    res = {}
    for p in ('prefix40', 'prefix60', 'prefix80'):
        parts = [pl.read_parquet(S169.X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
                 .join(pl.read_parquet(S169.X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
                 .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
        d = (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
             .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))
        s, lab, key = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['key'].to_numpy()
        T = s < .5
        assert np.isfinite(key[T]).all()
        k11 = (10 * key + S159.lg(s)) / 11
        fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
        yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
        yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
        met = lambda sc: {'PP90': S159.ap(yp, sc), 'PPcls': S159.ap(yc, sc), 'RAW': S159.ap(yr, sc), 'B': S159.bmap(sc, lab, fam, pred)}
        tl = T & np.isfinite(key)
        res[p] = {'FINAL2_16': met(S159.permute(s, key, T)), 'K11': met(S159.permute(s, k11, T)),
                  'FINAL2_16_merge': met(S169.merged(s, key, 10, 50)), 'K11_merge': met(S169.merged(s, k11, 10, 50)),
                  'tail_spearman_key_lgrisk': float(pl.DataFrame({'a': key[tl], 'b': s[tl]}).select(pl.corr('a', 'b', method='spearman')).item())}
    out = {'registration_sha256': reg_sha, 'surfaces': res,
           'primary_K11_vs_FINAL2_16': S159.rule({p: {'a': res[p]['K11'], 'b': res[p]['FINAL2_16']} for p in res}, 'a', 'b'),
           'descriptive_K11_merge_vs_FINAL2_16_merge': S159.rule({p: {'a': res[p]['K11_merge'], 'b': res[p]['FINAL2_16_merge']} for p in res}, 'a', 'b')}
    json.dump(out, open(dest, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'surfaces'}, indent=1))
    print({p: round(res[p]['tail_spearman_key_lgrisk'], 3) for p in res})


if __name__ == '__main__':
    main()
