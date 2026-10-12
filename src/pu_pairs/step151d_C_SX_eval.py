"""Step 151d (Claude): registered readings of part B (prediction_step151.json sha e4868079...). Gate: draw k = 0 must reproduce the
stored STEP131 views (r_SHORT3, r_ALL5, r_LIN30, r_LIN300, r_FAMMAX) on all three surfaces with max abs diff <= 1e-6, else abort.
Surfaces, eligibility, label definitions and AP as scratchpad tailkeys_screen.py / tv131_eval.py. Development labels only."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, hashlib
sys.dont_write_bytecode = True
import numpy as np, polars as pl
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
E = C + 'experiments/'
O, X, VD, TK = E + 'tail_view_confirmation_a8_20260913/', E + 'pu_pairs_20260911/', E + 'viewdis_20260911/', E + 'tail_keys_20260914/'
REG = 'e4868079522a7807905424b0067e0f9be054e8a5a44ffe038dbd2496543f5c90'
assert hashlib.sha256(open(C + 'outputs_v1_followups_20260914/prediction_step151.json', 'rb').read()).hexdigest() == REG
VT = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']
FAM = {1: 'directed_transfer', 2: 'soft_play', 3: 'coordinated_isolation'}


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def ap(y, s):
    o = np.argsort(-s, kind='mergesort'); yy = y[o]; tp = np.cumsum(yy)
    return float((tp / np.arange(1, len(yy) + 1) * yy).sum() / max(1, int(yy.sum())))


def permute(s, z, blk):
    s2 = s.copy(); idx = np.where(blk)[0]; s2[idx[np.argsort(-z[idx], kind='mergesort')]] = np.sort(s[idx])[::-1]
    return s2


def bmap(sc, lab, fam, pred):
    return float(np.mean([ap(((lab == 1) & (fam == FAM[c])).astype(int), np.where(pred == c, sc, 0.0)) for c in (1, 2, 3)]))


SX = pl.concat([pl.read_parquet(TK + f) for f in sorted(os.listdir(TK)) if f.startswith('sx_dev_fold') and f.endswith('.parquet')])
assert SX.select('fold', 'k').unique().height == 16, SX.select('fold', 'k').unique().sort('fold', 'k')
stored = pl.read_parquet(O + 'views_prefix40_60_80.parquet', columns=['pair_id', 'fold', 'surface'] + [f'r_{v}' for v in VT])
g = SX.filter(pl.col('k') == 0).join(stored, on=['pair_id', 'fold', 'surface'], how='inner', suffix='_st')
gate = {'rows': g.height, 'stored_rows': stored.height}
gate.update({v: float((g[f'r_{v}'] - g[f'r_{v}_st']).abs().max()) for v in VT})
ok = g.height == stored.height and all(gate[v] <= 1e-6 for v in VT)
res = {'registration_sha256': REG, 'gate_k0_reproduces_stored': ok, 'gate': gate}
if not ok:
    json.dump(res, open(TK + 'sx_dev_eval.json', 'w', encoding='utf-8'), indent=1); sys.exit(print('ABORT gate', gate))
# per (pair, fold, surface): view-type logit mean over draws, for K = 1, 2, 4, 8
agg = {}
SXL = SX.with_columns([pl.Series(f'L_{v}', lg(SX[f'r_{v}'])) for v in VT])
for K in (1, 2, 4, 8):
    a = SXL.filter(pl.col('k') < K).group_by('pair_id', 'fold', 'surface').agg([pl.col(f'L_{v}').mean() for v in VT])
    agg[K] = a.with_columns(pl.mean_horizontal([f'L_{v}' for v in VT]).alias(f'SX{K}')).select('pair_id', 'fold', 'surface', f'SX{K}')
R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})


def surface(p):
    parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
             .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
             .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
    d = pl.concat(parts).join(stored.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
    for K in agg:
        d = d.join(agg[K].filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
    return d.unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id')


res['surfaces'] = {}
for p in ('prefix40', 'prefix60', 'prefix80'):
    d = surface(p)
    assert all(d[f'SX{K}'].null_count() == 0 for K in agg)
    s, lab, fold, tab = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['fold'].to_numpy(), d['table_id'].to_numpy()
    fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
    yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
    yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
    T = s < .5
    KEYS = {'P': lg(d['r_SHORT3']) + lg(d['r_ALL5']), 'S5': np.mean([lg(d[f'r_{v}']) for v in VT], axis=0)}
    KEYS.update({f'SX{K}': d[f'SX{K}'].to_numpy() for K in agg})
    assert np.allclose(KEYS['SX1'], KEYS['S5'], atol=1e-9)
    A = {k: permute(s, z, T) for k, z in KEYS.items()}
    M = {k: {'PP90': ap(yp, sc), 'PPcls': ap(yc, sc), 'RAW': ap(yr, sc), 'B': bmap(sc, lab, fam, pred)} for k, sc in A.items()}
    out = {'n': len(s), 'n_tail': int(T.sum()), 'tail_pos_PP90': int(yp[T].sum()), 'abs': M}
    for ref in ('S5', 'P'):
        out[f'vs_{ref}'] = {k: {**{m: M[k][m] - M[ref][m] for m in M[k]},
                                'fold_PP90': {int(f): ap(yp[fold == f], A[k][fold == f]) - ap(yp[fold == f], A[ref][fold == f]) for f in (1, 2)}} for k in A if k != ref}
    rb = np.random.default_rng(13); ut = np.unique(tab); by = {t: np.where(tab == t)[0] for t in ut}; bt = []
    for _ in range(300):
        idx = np.concatenate([by[t] for t in rb.choice(ut, len(ut), replace=True)])
        bt.append(ap(yp[idx], A['SX8'][idx]) - ap(yp[idx], A['S5'][idx]))
    out['boot_SX8_minus_S5_PP90'] = [float(np.quantile(bt, .05)), float(np.median(bt)), float(np.quantile(bt, .95)), float(np.mean(np.array(bt) > 0))]
    res['surfaces'][p] = out
    v = out['vs_S5']
    print(p, 'tailpos', out['tail_pos_PP90'], ' '.join(f"{k}: dPP90 {v[k]['PP90']:+.4f} dPPcls {v[k]['PPcls']:+.4f} dB {v[k]['B']:+.4f}" for k in ('SX2', 'SX4', 'SX8', 'P')),
          'boot', [round(x, 4) for x in out['boot_SX8_minus_S5_PP90']], flush=True)
S = res['surfaces']
d90 = [S[p]['vs_S5']['SX8']['PP90'] for p in S]; dcl = [S[p]['vs_S5']['SX8']['PPcls'] for p in S]
res['rule'] = {'mean_dPP90_SX8_minus_S5': float(np.mean(d90)), 'min_surface_dPP90': float(np.min(d90)), 'mean_dPPcls': float(np.mean(dcl)),
               'i': bool(np.mean(d90) > 0), 'ii': bool(np.min(d90) >= -.0005), 'iii': bool(np.mean(dcl) > 0)}
res['rule']['SX_REPLACES_S5'] = res['rule']['i'] and res['rule']['ii'] and res['rule']['iii']
res['rule']['mean_comp_SX8_minus_S5'] = float(np.mean([.7 * S[p]['vs_S5']['SX8']['PP90'] + .1 * S[p]['vs_S5']['SX8']['B'] for p in S]))
json.dump(res, open(TK + 'sx_dev_eval.json', 'w', encoding='utf-8'), indent=1)
print('RULE', json.dumps(res['rule']))
