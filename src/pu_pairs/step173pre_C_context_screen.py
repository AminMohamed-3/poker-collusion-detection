"""Step 173pre (Claude): descriptive screen for pair-risk route PR1 (context-aligned first-stage scores, the pair analogue of LP).
No registration depends on its numbers being positive; it decides whether PR1 is worth registering. Development labels only.
Surfaces as step168b (STEP131 folds 1-2, prefix-40/60/80, shared_hands >= 38). Current order = block by fold prefix risk, tail by the
FINAL2_16 key (step159c permute), score sc. Within-phase context of each pair from sc only (pair list player_1/player_2 and table_id,
as evaluation_pairs.csv provides; no metadata, no cross-phase linkage, no labels):
  m_other = max sc over the two players' other pairs; mutual_top = sc above both players' other pairs; gap = m_other - sc (clipped);
  n_tab_above = pairs of the same table with higher sc.
Reads: (1) rates of mutual_top / gap among PP90 positives vs others in the top region (top 3 x block rows);
(2) one fixed cross-fold logistic re-rank of that region (fit on the other fold's same surface: features lg(sc), mutual_top, gap,
log1p(n_tab_above); value multiset kept) -> dPP90 / dPPcls / dRAW. Output experiments/pair_context_20260915/context_screen.json."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '3'
import sys, json
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs import step159c_C_combine_dev as S159
from pu_pairs.step60_multiview_blend import logreg, predict

E = C + 'experiments/'
X, VD = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/'
S159.DIRS['ARCH2'] = (E + 'arch2_views_20260915/', 'arch2_dev_fold')
OUT = E + 'pair_context_20260915/'


def frames():
    V = S159.type_means('SXN3', S159.VT).join(S159.type_means('ARCH', S159.NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    V = V.join(S159.type_means('ARCH2', ['MLP2', 'HGB5']), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    cols = [c for c in V.columns if c not in ('pair_id', 'fold', 'surface')]
    assert len(cols) == 10
    V = V.with_columns(pl.mean_horizontal(cols).alias('key')).select('pair_id', 'fold', 'surface', 'key')
    R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    D = {}
    for p in ('prefix40', 'prefix60', 'prefix80'):
        for f in (1, 2):
            d = (pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'player_1', 'player_2', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
                 .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
                 .with_columns(pl.lit(f).alias('fold')).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
                 .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))
            D[(p, f)] = d
    return D


def context(d, sc):
    f = d.select('pair_id', 'player_1', 'player_2', 'table_id').with_columns(pl.Series('sc', sc))
    long = pl.concat([f.select('pair_id', pl.col('player_1').alias('pl'), 'sc'), f.select('pair_id', pl.col('player_2').alias('pl'), 'sc')])
    # max and second max per player, so "other pairs" excludes the pair itself
    top2 = long.group_by('pl').agg(pl.col('sc').max().alias('m1'), pl.col('sc').sort(descending=True).slice(1, 1).first().alias('m2'))
    def other(col):
        j = f.select('pair_id', col, 'sc').join(top2, left_on=col, right_on='pl', how='left')
        return np.where(j['sc'].to_numpy() >= j['m1'].to_numpy(), j['m2'].fill_null(0.0).to_numpy(), j['m1'].to_numpy())
    o1, o2 = other('player_1'), other('player_2')
    m_other = np.maximum(o1, o2)
    n_tab = f.with_columns(pl.col('sc').rank('min', descending=True).over('table_id').alias('r'))['r'].to_numpy() - 1
    return {'m_other': m_other, 'mutual_top': ((sc > o1) & (sc > o2)).astype(float), 'gap': np.clip(m_other - sc, -1, 1), 'n_tab_above': n_tab.astype(float)}


def labels(d):
    lab = d['label'].to_numpy(); unl = lab == -1
    rf, r1 = d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
    yr = lab == 1
    return (yr | (unl & (rf >= .9))).astype(int), (yr | (unl & (rf >= .9) & ~r1)).astype(int), yr.astype(int)


def main():
    os.makedirs(OUT, exist_ok=True)
    D = frames()
    prep = {}
    for (p, f), d in D.items():
        s = d['risk'].to_numpy().astype(float); key = d['key'].to_numpy()
        assert np.isfinite(key).all()
        sc = S159.permute(s, key, s < .5)
        ctx = context(d, sc)
        yp, yc, yr = labels(d)
        nblk = int((s >= .5).sum()); top = np.argsort(-sc, kind='mergesort')[:3 * nblk]
        region = np.zeros(len(sc), bool); region[top] = True
        Z = np.column_stack([S159.lg(np.clip(sc, 1e-6, 1 - 1e-6)), ctx['mutual_top'], ctx['gap'], np.log1p(ctx['n_tab_above'])])
        prep[(p, f)] = dict(sc=sc, Z=Z, yp=yp, yc=yc, yr=yr, region=region, ctx=ctx, nblk=nblk)
    res = {'note': 'descriptive screen; no registration depends on the sign', 'surfaces': {}}
    for (p, f), q in prep.items():
        g = 3 - f; o = prep[(p, g)]
        w, b, m2, s2 = logreg(o['Z'][o['region']], o['yp'][o['region']].astype(float), l2=10., iters=60)
        z = predict((w, b, m2, s2), q['Z'])
        new = q['sc'].copy(); idx = np.where(q['region'])[0]
        new[idx[np.argsort(-z[idx], kind='mergesort')]] = np.sort(q['sc'][idx])[::-1]
        R = q['region']; pos = q['yp'][R] == 1
        mt, gp = q['ctx']['mutual_top'][R], q['ctx']['gap'][R]
        res['surfaces'][f'{p}_fold{f}'] = {
            'block_rows': q['nblk'], 'region_rows': int(R.sum()), 'region_pp90_pos': int(pos.sum()),
            'mutual_top_rate_pos_vs_other': [round(float(mt[pos].mean()), 3), round(float(mt[~pos].mean()), 3)],
            'gap_median_pos_vs_other': [round(float(np.median(gp[pos])), 4), round(float(np.median(gp[~pos])), 4)],
            'std_weights_fit_on_other_fold': dict(zip(['lg_sc', 'mutual_top', 'gap', 'log_n_tab_above'], [round(float(x), 3) for x in w])),
            'dPP90': round(S159.ap(q['yp'], new) - S159.ap(q['yp'], q['sc']), 5), 'dPPcls': round(S159.ap(q['yc'], new) - S159.ap(q['yc'], q['sc']), 5),
            'dRAW': round(S159.ap(q['yr'], new) - S159.ap(q['yr'], q['sc']), 5), 'PP90_base': round(S159.ap(q['yp'], q['sc']), 5)}
        print(p, f, json.dumps(res['surfaces'][f'{p}_fold{f}']), flush=True)
    for m in ('dPP90', 'dPPcls', 'dRAW'):
        res['mean_' + m] = round(float(np.mean([v[m] for v in res['surfaces'].values()])), 5)
    json.dump(res, open(OUT + 'context_screen.json', 'w', encoding='utf-8'), indent=1)
    print('MEAN', res['mean_dPP90'], res['mean_dPPcls'], res['mean_dRAW'], flush=True)


if __name__ == '__main__':
    main()
