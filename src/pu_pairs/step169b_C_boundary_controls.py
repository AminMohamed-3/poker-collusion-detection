"""Step 169b (Claude): descriptive controls for the STEP169 boundary merge (registered primary N10_M50 passed ER3). Not a gate; no change to the
STEP169 verdict. Same dev surfaces and code path as step169. Reports per surface and per fold: (a) primary deltas by fold; (b) random-pick
control: 10 tail pairs drawn uniformly from the tail placed above the 50 lowest block pairs (100 draws); (c) the next-ten control: tail pairs
ranked 11-20 and 21-30 by the key; (d) PP90-positive counts among the top-10 key tail pairs and the 50 lowest block pairs. Output
experiments/boundary_merge_20260915/controls169b.json."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs import step169_C_boundary_merge_dev as S169
S159 = S169.S159


def placed(s, picks, M):
    """block[:B-M] by risk, picks (tail indices, given order), block[B-M:], rest of tail by key order already fixed by caller."""
    return picks


def order_with(s, key, tail_sel, M):
    T = s < .5; bi, ti = np.flatnonzero(~T), np.flatnonzero(T)
    bo = bi[np.lexsort((bi, -s[bi]))]; to = ti[np.lexsort((ti, -key[ti]))]
    B = len(bo); sel = to[tail_sel] if tail_sel is not None else None
    rest = to[np.setdiff1d(np.arange(len(to)), tail_sel)]
    order = np.concatenate([bo[:B - M], sel, bo[B - M:], rest])
    out = np.empty_like(s); out[order] = np.sort(s)[::-1]
    return out, bo, to


def main():
    dest = S169.E + 'boundary_merge_20260915/controls169b.json'; assert not os.path.exists(dest)
    V = S159.type_means('SXN3', S169.VT).join(S159.type_means('ARCH', S169.NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    V = V.join(S159.type_means('ARCH2', S169.A2), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    base = [f'SXN3_{v}' for v in S169.VT] + [f'ARCH_{v}' for v in S169.NEW_VT] + [f'ARCH2_{v}' for v in S169.A2]
    V = V.with_columns(pl.mean_horizontal(base).alias('key'))
    R1 = pl.read_parquet(S169.VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    out = {}
    for p in ('prefix40', 'prefix60', 'prefix80'):
        parts = [pl.read_parquet(S169.X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
                 .join(pl.read_parquet(S169.X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
                 .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
        d = (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
             .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))
        s, lab, key = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['key'].to_numpy()
        T = s < .5
        assert np.isfinite(key[T]).all()
        yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
        yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
        cur = S159.permute(s, key, T)
        c90 = S159.ap(yp, cur); ccl = S159.ap(yc, cur)
        r = {}
        prim, bo, to = order_with(s, key, np.arange(10), 50)
        assert np.array_equal(prim, S169.merged(s, key, 10, 50))
        r['primary'] = {'dPP90': S159.ap(yp, prim) - c90, 'dPPcls': S159.ap(yc, prim) - ccl}
        r['pp90_pos_top10_key_tail'] = int(yp[to[:10]].sum()); r['pp90_pos_lowest50_block'] = int(yp[bo[-50:]].sum())
        r['raw_pos_top10_key_tail'] = int(yr[to[:10]].sum()); r['raw_pos_lowest50_block'] = int(yr[bo[-50:]].sum())
        for lo in (10, 20):
            a, _, _ = order_with(s, key, np.arange(lo, lo + 10), 50)
            r[f'next_ten_{lo + 1}_{lo + 10}'] = {'dPP90': S159.ap(yp, a) - c90, 'pp90_pos': int(yp[to[lo:lo + 10]].sum())}
        rng = np.random.default_rng(1690); rd = []
        for _ in range(100):
            a, _, _ = order_with(s, key, np.sort(rng.choice(len(to), 10, replace=False)), 50)
            rd.append(S159.ap(yp, a) - c90)
        r['random10_dPP90'] = {'mean': float(np.mean(rd)), 'max': float(np.max(rd)), 'share_ge_primary': float(np.mean(np.array(rd) >= r['primary']['dPP90']))}
        fold = d['fold'].to_numpy()
        for f in (1, 2):
            m = fold == f
            sf, kf = s[m], key[m]
            a = S169.merged(sf, kf, 10, round(50 * (~(sf < .5)).sum() / (~T).sum()))
            cf = S159.permute(sf, kf, sf < .5)
            r[f'fold{f}_scaledM'] = {'block': int((sf >= .5).sum()), 'M': round(50 * (~(sf < .5)).sum() / (~T).sum()), 'dPP90': S159.ap(yp[m], a) - S159.ap(yp[m], cf)}
        out[p] = r
        print(p, json.dumps(r), flush=True)
    json.dump(out, open(dest, 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
