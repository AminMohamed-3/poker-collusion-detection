"""Step 169 (Claude): re-open R-c of VALIDATION_BOOK_REVIEW_20260915.md. Boundary merge at .5 with the current tail key on the STEP131 dev
surfaces (folds 1-2, prefix-40/60/80, shared_hands >= 38): the N highest tail pairs by the FINAL2_16 key are placed directly above the M
lowest block pairs (block in its risk order); every other position keeps the current recipe (block by risk, tail by key). Values keep
their multiset (sorted values handed out in the new order). Primary cell N = 10, M = 50 under the three conditions (ER3); the grid
N in {5, 10, 25} x M in {25, 50, 100} is descriptive. Output experiments/boundary_merge_20260915/boundary169.json. Usage: REGSHA"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, hashlib
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs import step159c_C_combine_dev as S159

E = C + 'experiments/'
X, VD = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/'
S159.DIRS['ARCH2'] = (E + 'arch2_views_20260915/', 'arch2_dev_fold')
VT, NEW_VT, A2 = S159.VT, S159.NEW_VT, ['MLP2', 'HGB5']
REG = C + 'outputs_step169_20260915/prediction_step169.json'


def merged(s, key, N, M):
    """Order: block[:B-M] by risk, top-N tail by key, block[B-M:] by risk, rest of tail by key; sorted values handed out in that order."""
    T = s < .5; bi, ti = np.flatnonzero(~T), np.flatnonzero(T)
    bo = bi[np.lexsort((bi, -s[bi]))]
    to = ti[np.lexsort((ti, -key[ti]))]
    B = len(bo)
    order = np.concatenate([bo[:B - M], to[:N], bo[B - M:], to[N:]])
    out = np.empty_like(s); out[order] = np.sort(s)[::-1]
    return out


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(open(REG, 'rb').read()).hexdigest() == reg_sha
    dest = E + 'boundary_merge_20260915/boundary169.json'; os.makedirs(os.path.dirname(dest), exist_ok=True); assert not os.path.exists(dest)
    V = S159.type_means('SXN3', VT).join(S159.type_means('ARCH', NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    V = V.join(S159.type_means('ARCH2', A2), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    base = [f'SXN3_{v}' for v in VT] + [f'ARCH_{v}' for v in NEW_VT] + [f'ARCH2_{v}' for v in A2]
    V = V.with_columns(pl.mean_horizontal(base).alias('key'))
    R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    cells = [(n, m) for n in (5, 10, 25) for m in (25, 50, 100)]
    res = {p: {} for p in ('prefix40', 'prefix60', 'prefix80')}
    for p in res:
        parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
                 .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
                 .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
        d = (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
             .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))
        s, lab, key = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['key'].to_numpy()
        fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
        yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
        yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
        met = lambda sc: {'PP90': S159.ap(yp, sc), 'PPcls': S159.ap(yc, sc), 'RAW': S159.ap(yr, sc), 'B': S159.bmap(sc, lab, fam, pred)}
        cur = S159.permute(s, key, s < .5)
        assert np.array_equal(merged(s, key, 0, 0), cur) or np.allclose(np.sort(merged(s, key, 0, 0)), np.sort(cur))
        res[p]['current'] = met(cur)
        for n, m in cells:
            res[p][f'N{n}_M{m}'] = met(merged(s, key, n, m))
    out = {'registration_sha256': reg_sha, 'surfaces': res, 'cells': {}}
    for n, m in cells:
        c = f'N{n}_M{m}'
        r = S159.rule({p: {'a': res[p][c], 'b': res[p]['current']} for p in res}, 'a', 'b')
        r['per_surface_dPP90'] = {p: res[p][c]['PP90'] - res[p]['current']['PP90'] for p in res}
        out['cells'][c] = r
    out['primary'] = out['cells']['N10_M50']
    json.dump(out, open(dest, 'w', encoding='utf-8'), indent=1)
    for c, r in out['cells'].items():
        print(c, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != 'per_surface_dPP90'}), {p: round(v, 5) for p, v in r['per_surface_dPP90'].items()}, flush=True)
    print('PRIMARY passes', out['primary']['passes'], flush=True)


if __name__ == '__main__':
    main()
