"""Step 173 (Claude): pair-risk route, boundary region by one score (R-c2; follows STEP169 R-c, public +.00050). STEP169 places the N
highest tail pairs by the FINAL2 key directly above the M lowest block pairs, whose own order stays the block order. Here the union of
those M block pairs and N tail pairs is ordered by the FINAL2 key (interleave), so block-bottom pairs the views dislike fall below
tail pairs the views like, and block-bottom pairs the views like rise. Dev surfaces as STEP169 (STEP131 folds 1-2 pooled per surface,
prefix-40/60/80, shared_hands >= 38; dev block order = prefix risk, production = M3x). Values keep their multiset.
Primary (registered): interleave(N=10, M=50) vs merge(N=10, M=50) under ER3. Descriptive: the same at (10,100), (25,50), (25,100), and
a union order by the mean of within-union ranks of risk and key. Output experiments/boundary_merge_20260915/interleave173.json.
Usage: REGSHA"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, hashlib
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs import step159c_C_combine_dev as S159
from pu_pairs.step169_C_boundary_merge_dev import merged

E = C + 'experiments/'
X, VD = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/'
S159.DIRS['ARCH2'] = (E + 'arch2_views_20260915/', 'arch2_dev_fold')
VT, NEW_VT, A2 = S159.VT, S159.NEW_VT, ['MLP2', 'HGB5']
REG = C + 'outputs_step173_20260915/prediction_step173.json'


def interleave(s, key, N, M, how='key'):
    """block[:B-M] by risk; union(block[B-M:], top-N tail by key) ordered by key (or mean within-union rank of risk and key); rest of tail by key."""
    T = s < .5; bi, ti = np.flatnonzero(~T), np.flatnonzero(T)
    bo = bi[np.lexsort((bi, -s[bi]))]
    to = ti[np.lexsort((ti, -key[ti]))]
    B = len(bo)
    U = np.concatenate([bo[B - M:], to[:N]])
    if how == 'key':
        u = U[np.lexsort((U, -key[U]))]
    else:
        rr = np.argsort(np.argsort(-s[U], kind='mergesort'), kind='mergesort'); rk = np.argsort(np.argsort(-key[U], kind='mergesort'), kind='mergesort')
        u = U[np.lexsort((U, rr + rk))]
    order = np.concatenate([bo[:B - M], u, to[N:]])
    out = np.empty_like(s); out[order] = np.sort(s)[::-1]
    return out


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(open(REG, 'rb').read()).hexdigest() == reg_sha
    dest = E + 'boundary_merge_20260915/interleave173.json'; assert not os.path.exists(dest)
    V = S159.type_means('SXN3', VT).join(S159.type_means('ARCH', NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    V = V.join(S159.type_means('ARCH2', A2), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    base = [f'SXN3_{v}' for v in VT] + [f'ARCH_{v}' for v in NEW_VT] + [f'ARCH2_{v}' for v in A2]
    V = V.with_columns(pl.mean_horizontal(base).alias('key'))
    R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    cells = [(10, 50), (10, 100), (25, 50), (25, 100)]
    S = {}
    ref169 = json.load(open(E + 'boundary_merge_20260915/boundary169.json', encoding='utf-8'))
    for p in ('prefix40', 'prefix60', 'prefix80'):
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
        S[p] = {}
        for n, m in cells:
            mg = met(merged(s, key, n, m))
            if (n, m) == (10, 50):
                assert abs(mg['PP90'] - ref169['surfaces'][p]['N10_M50']['PP90']) < 1e-12, 'STEP169 merge replay'
            S[p][f'merge_N{n}_M{m}'] = mg
            S[p][f'inter_N{n}_M{m}'] = met(interleave(s, key, n, m))
            S[p][f'interRR_N{n}_M{m}'] = met(interleave(s, key, n, m, how='rankmean'))
    res = {'registration_sha256': reg_sha, 'surfaces': S}
    res['primary'] = S159.rule({p: {'a': S[p]['inter_N10_M50'], 'b': S[p]['merge_N10_M50']} for p in S}, 'a', 'b')
    res['descriptive'] = {f'{arm}_N{n}_M{m}': S159.rule({p: {'a': S[p][f'{arm}_N{n}_M{m}'], 'b': S[p][f'merge_N{n}_M{m}']} for p in S}, 'a', 'b')
                          for arm in ('inter', 'interRR') for n, m in cells}
    json.dump(res, open(dest, 'w', encoding='utf-8'), indent=1)
    print('PRIMARY', json.dumps(res['primary']))
    for k, v in res['descriptive'].items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != 'mean_comp'}))


if __name__ == '__main__':
    main()
