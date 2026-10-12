"""Step 176 (Claude): pair-risk route, boundary arrangement under a faithful block order. STEP169 (merge, public +.0005) and STEP173
(interleave, file built, not read) were selected on dev surfaces whose block order is the fold prefix risk. Production orders the block by
M3x (the M3 logistic blend with 32-draw SHORT3/ALL5), which on the step 64f certified testbed is far better than prefix risk (def A AP .969
vs .943), so the STEP173 descriptive gains of deeper cells (+.005 to +.007 at M = 100) may be prefix-risk artefacts. On the prefix-60 dev
surface every block pair has a cross-fold M3x32 score (gates in smoke176: testbed AP replay exact; merge and interleave replays exact).
Grid (29 cells): none; merge N in {10,25,50} x M in {25,50,100,150}; inter N in {0,10,25,50} x M in {25,50,100,150}. Block order = M3x;
tail order = FINAL key; union of inter ordered by the key; values keep their multiset. Incumbent = merge_N10_M50 (the .90821 recipe).
Groups f1 (f1-only tables; M3x from the fold-2 fit) and f2sh (f2-only and shared tables; M3x from the fold-1 fit): PP90 / PPcls per group
= AP over that group's rows under the pooled arrangement. Honest selection: pick on one group, score on the other; G = positive-weighted
mean held delta vs incumbent. Final = best pooled PP90. Proceed iff G > 0, final != incumbent, pooled dPP90 > 0, each group dPP90 >= -.0005,
pooled dPPcls > 0. Descriptive: the same grid with block order = prefix risk; B.
Output experiments/boundary_m3x_20260915/boundary176.json. Usage: REGSHA"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, hashlib, time
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs.step64f_M3_candidate import testbed, FS_M3
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step60_multiview_blend import ap as ap_b, logreg, predict
from pu_pairs.step62_multiview_candidate import L2, DEFS
from pu_pairs.build_allpairs_fold import fold_pools
from pu_pairs import step159c_C_combine_dev as S159
from pu_pairs.step169_C_boundary_merge_dev import merged
from pu_pairs.step173_C_boundary_interleave_dev import interleave

E = C + 'experiments/'
X, VD, TK = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/', E + 'tail_keys_20260914/'
REG = C + 'outputs_step176_20260915/prediction_step176.json'
DEST = E + 'boundary_m3x_20260915/boundary176.json'
lg, sig = S159.lg, S159.sig
NS_MERGE, NS_INTER, MS = (10, 25, 50), (0, 10, 25, 50), (25, 50, 100, 150)
INC = 'merge_N10_M50'


def cells():
    out = ['none'] + [f'merge_N{n}_M{m}' for n in NS_MERGE for m in MS] + [f'inter_N{n}_M{m}' for n in NS_INTER for m in MS]
    assert len(out) == 29 and INC in out
    return out


def arrange(s, key, bs, cell):
    """Block (s >= .5) by bs desc, tail by key desc; merge puts top-N tail above the bottom-M block; inter orders their union by key."""
    T = s < .5; bi, ti = np.flatnonzero(~T), np.flatnonzero(T)
    bo = bi[np.lexsort((bi, -bs[bi]))]; to = ti[np.lexsort((ti, -key[ti]))]; B = len(bo)
    if cell == 'none':
        order = np.concatenate([bo, to])
    else:
        how, n, m = cell.split('_'); N, M = int(n[1:]), int(m[1:])
        if how == 'merge':
            order = np.concatenate([bo[:B - M], to[:N], bo[B - M:], to[N:]])
        else:
            U = np.concatenate([bo[B - M:], to[:N]]); order = np.concatenate([bo[:B - M], U[np.lexsort((U, -key[U]))], to[N:]])
    out = np.empty_like(s); out[order] = np.sort(s)[::-1]
    return out


def m3x_scores():
    SX = pl.concat([pl.read_parquet(TK + f) for f in sorted(os.listdir(TK)) if f.startswith('sx_dev_fold') and f.endswith('.parquet')]).filter(pl.col('surface') == 'prefix60')
    SXL = SX.with_columns([pl.Series(f'L_{v}', lg(SX[f'r_{v}'])) for v in ('SHORT3', 'ALL5')])
    a = SXL.filter(pl.col('k') < 32).group_by('pair_id', 'fold').agg(pl.col('L_SHORT3').mean(), pl.col('L_ALL5').mean())
    k0 = SX.filter(pl.col('k') == 0).select('pair_id', 'fold', pl.col('r_SHORT3').alias('k0_S3'), pl.col('r_ALL5').alias('k0_A5'))
    d = testbed()
    j = d.select('pair_id', 'r_SHORT3', 'r_ALL5').join(k0, on='pair_id', how='left')
    j = j.with_columns(((pl.col('k0_S3') - pl.col('r_SHORT3')).abs() + (pl.col('k0_A5') - pl.col('r_ALL5')).abs()).alias('dif')).sort('pair_id', 'dif').unique(subset=['pair_id'], keep='first', maintain_order=True)
    gate_k0 = float(j['dif'].max())
    d = d.join(j.select('pair_id', pl.col('fold').alias('fold_m')), on='pair_id', how='left', validate='1:1')
    d = d.join(a.rename({'fold': 'fold_m'}), on=['pair_id', 'fold_m'], how='left', validate='1:1')
    assert d['L_SHORT3'].null_count() == 0
    d = d.with_columns(pl.Series('r_SHORT3', sig(d['L_SHORT3'].to_numpy())), pl.Series('r_ALL5', sig(d['L_ALL5'].to_numpy())))
    posset, negset = DEFS['A']
    cert = d['cls'].is_in(posset + negset).to_numpy(); y = d['cls'].is_in(posset).to_numpy().astype(float)
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2; tab = d['table_id'].to_numpy()
    in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
    assert (in1 | in2 | insh).all()
    Xm = X_of(d, FS_M3, 'risk_prefix', 'R1_prefix', 'risk_noT'); pred = np.full(d.height, np.nan)
    m1 = logreg(Xm[in1 & cert], y[in1 & cert], L2); pred[in2 | insh] = predict(m1, Xm[in2 | insh])
    m2 = logreg(Xm[in2 & cert], y[in2 & cert], L2); pred[in1] = predict(m2, Xm[in1])
    ref = json.load(open(C + 'outputs_step156_20260915/block_screen.json', encoding='utf-8'))['A']['M3x32']
    got = {'pooled': ap_b(y[cert], pred[cert]), 'test_f2sh': ap_b(y[cert & (in2 | insh)], pred[cert & (in2 | insh)]), 'test_f1': ap_b(y[cert & in1], pred[cert & in1])}
    assert gate_k0 == 0.0 and all(got[k] == ref[k] for k in got), (gate_k0, got, ref)
    return d.select('pair_id').with_columns(pl.Series('m3x', pred))


def surface():
    V = S159.type_means('SXN3', S159.VT).join(S159.type_means('ARCH', S159.NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    S159.DIRS['ARCH2'] = (E + 'arch2_views_20260915/', 'arch2_dev_fold')
    V = V.join(S159.type_means('ARCH2', ['MLP2', 'HGB5']), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    base = [f'SXN3_{v}' for v in S159.VT] + [f'ARCH_{v}' for v in S159.NEW_VT] + [f'ARCH2_{v}' for v in ['MLP2', 'HGB5']]
    V = V.with_columns(pl.mean_horizontal(base).alias('key'))
    R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    p = 'prefix60'
    parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
             .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
             .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
    return (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
            .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))


def main():
    t0 = time.time()
    reg_sha = sys.argv[1]
    assert hashlib.sha256(open(REG, 'rb').read()).hexdigest() == reg_sha
    reg = json.load(open(REG, encoding='utf-8'))
    assert hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest() == reg['runner_sha256']
    os.makedirs(os.path.dirname(DEST), exist_ok=True); assert not os.path.exists(DEST)
    d = surface().join(m3x_scores(), on='pair_id', how='left', validate='1:1')
    s, lab, key = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['key'].to_numpy()
    assert d.filter(pl.col('risk') >= .5)['m3x'].null_count() == 0 and d['key'].null_count() == 0
    bs = d['m3x'].fill_null(-1.0).to_numpy()
    fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
    yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
    yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
    ref173 = json.load(open(E + 'boundary_merge_20260915/interleave173.json', encoding='utf-8'))['surfaces']['prefix60']
    assert S159.ap(yp, arrange(s, key, s, INC)) == ref173['merge_N10_M50']['PP90'] == S159.ap(yp, merged(s, key, 10, 50))
    assert S159.ap(yp, arrange(s, key, s, 'inter_N10_M50')) == ref173['inter_N10_M50']['PP90'] == S159.ap(yp, interleave(s, key, 10, 50))
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2
    grp = np.array(['f1' if (t in f1 and t not in shared) else 'f2sh' for t in d['table_id'].to_numpy()])
    G_ = {'f1': grp == 'f1', 'f2sh': grp == 'f2sh'}
    npos = {g: int(yp[m].sum()) for g, m in G_.items()}
    ids = cells()
    res = {}
    for order_name, bsc in (('M3x', bs), ('prefix_risk', s)):
        R = {}
        for c in ids:
            sc = arrange(s, key, bsc, c)
            R[c] = {'PP90': S159.ap(yp, sc), 'PPcls': S159.ap(yc, sc), 'B': S159.bmap(sc, lab, fam, pred)}
            for g, m in G_.items():
                R[c][f'PP90_{g}'] = S159.ap(yp[m], sc[m]); R[c][f'PPcls_{g}'] = S159.ap(yc[m], sc[m])
        res[order_name] = R
    R = res['M3x']
    rd = lambda x: round(float(x), 6)
    held = {}
    for on, off in (('f1', 'f2sh'), ('f2sh', 'f1')):
        pick = max(ids, key=lambda c: (R[c][f'PP90_{on}'], c == INC, -ids.index(c)))
        held[on] = {'cell': pick, 'selection_delta': rd(R[pick][f'PP90_{on}'] - R[INC][f'PP90_{on}']), 'held_group': off, 'held_delta': rd(R[pick][f'PP90_{off}'] - R[INC][f'PP90_{off}'])}
    G = (held['f1']['held_delta'] * npos['f2sh'] + held['f2sh']['held_delta'] * npos['f1']) / (npos['f1'] + npos['f2sh'])
    fin = max(ids, key=lambda c: (R[c]['PP90'], c == INC, -ids.index(c)))
    fv = {'dPP90': rd(R[fin]['PP90'] - R[INC]['PP90']), 'dPPcls': rd(R[fin]['PPcls'] - R[INC]['PPcls']), 'dB': rd(R[fin]['B'] - R[INC]['B']),
          'dPP90_f1': rd(R[fin]['PP90_f1'] - R[INC]['PP90_f1']), 'dPP90_f2sh': rd(R[fin]['PP90_f2sh'] - R[INC]['PP90_f2sh'])}
    proceed = bool(G > 0 and fin != INC and fv['dPP90'] > 0 and min(fv['dPP90_f1'], fv['dPP90_f2sh']) >= -.0005 and fv['dPPcls'] > 0)
    grid = {o: {c: {k: rd(v - res[o][INC][k]) for k, v in res[o][c].items()} for c in ids} for o in res}
    out = {'registration_sha256': reg_sha, 'surface': {'rows': d.height, 'block': int((s >= .5).sum()), 'tail': int((s < .5).sum()), 'pos_PP90': npos},
           'incumbent': {o: {k: rd(v) for k, v in res[o][INC].items()} for o in res}, 'cross_group': held, 'G_honest': rd(G), 'final_cell': fin, 'final_vs_incumbent': fv,
           'proceed': proceed, 'grid_delta_vs_incumbent': grid, 'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(DEST, 'w', encoding='utf-8'), indent=1)
    print('G', out['G_honest'], json.dumps(held), 'final', fin, json.dumps(fv), 'PROCEED', proceed, flush=True)


if __name__ == '__main__':
    main()
