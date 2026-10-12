"""Step 158b (Claude): registered readings of outputs_step156_20260915/prediction_step158.json (sha b46f8803): SXB16 (80% table-bagged draws)
against SXN16 (STEP156 Part B), plus descriptive SXB8 - SXN8 and the 32-member mix (SXN16 + SXB16 draws) against SXN16.
STEP131 surfaces, labels and AP as step152b. Output experiments/bagged_views_20260915/sxb_dev_eval.json. Development labels only."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, hashlib
sys.dont_write_bytecode = True
import numpy as np, polars as pl
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
E = C + 'experiments/'
X, VD, TN, TB = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/', E + 'tail_keys_20260915_neg/', E + 'bagged_views_20260915/'
REG = 'b46f88038938d7302c2c6e092ceaabcffef6697dede602732676ea0ace23a9e3'
assert hashlib.sha256(open(C + 'outputs_step156_20260915/prediction_step158.json', 'rb').read()).hexdigest() == REG
VT = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']
FAM = {1: 'directed_transfer', 2: 'soft_play', 3: 'coordinated_isolation'}


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def ap(y, s):
    o = np.argsort(-s, kind='mergesort'); yy = y[o]; tp = np.cumsum(yy)
    return float((tp / np.arange(1, len(yy) + 1) * yy).sum() / max(1, int(yy.sum())))


def permute(s, z, blk):
    s2 = s.copy(); idx = np.where(blk)[0]; s2[idx[np.argsort(-z[idx], kind='mergesort')]] = np.sort(s[idx])[::-1]; return s2


def bmap(sc, lab, fam, pred):
    return float(np.mean([ap(((lab == 1) & (fam == FAM[c])).astype(int), np.where(pred == c, sc, 0.0)) for c in (1, 2, 3)]))


def load(folder, prefix):
    d = pl.concat([pl.read_parquet(folder + f) for f in sorted(os.listdir(folder)) if f.startswith(prefix) and f.endswith('.parquet')]).filter(pl.col('k') < 16)
    assert d.select('fold', 'k').unique().height == 32
    return d.with_columns([pl.Series(f'L_{v}', lg(d[f'r_{v}'])) for v in VT])


def key(d, K, name):
    a = d.filter(pl.col('k') < K).group_by('pair_id', 'fold', 'surface').agg([pl.col(f'L_{v}').mean() for v in VT])
    return a.select('pair_id', 'fold', 'surface', pl.mean_horizontal([f'L_{v}' for v in VT]).alias(name))


N, B = load(TN, 'sxn_dev_fold'), load(TB, 'sxb_dev_fold')
assert N.height == B.height
MIX = pl.concat([N.select(B.columns), B]).group_by('pair_id', 'fold', 'surface').agg([pl.col(f'L_{v}').mean() for v in VT]).select('pair_id', 'fold', 'surface', pl.mean_horizontal([f'L_{v}' for v in VT]).alias('MIX32'))
AG = [key(N, 8, 'SXN8'), key(N, 16, 'SXN16'), key(B, 8, 'SXB8'), key(B, 16, 'SXB16'), MIX]
R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
res = {'registration_sha256': REG, 'surfaces': {}}
for p in ('prefix40', 'prefix60', 'prefix80'):
    parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
             .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
             .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
    d = pl.concat(parts)
    for a in AG:
        d = d.join(a.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
    d = d.unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id')
    keys = ['SXN8', 'SXN16', 'SXB8', 'SXB16', 'MIX32']
    assert all(d[k].null_count() == 0 for k in keys)
    s, lab, fold, tab = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['fold'].to_numpy(), d['table_id'].to_numpy()
    fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
    yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
    yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
    T = s < .5
    A = {k: permute(s, d[k].to_numpy(), T) for k in keys}
    M = {k: {'PP90': ap(yp, sc), 'PPcls': ap(yc, sc), 'RAW': ap(yr, sc), 'B': bmap(sc, lab, fam, pred)} for k, sc in A.items()}
    def dd(a, b):
        return {**{m: M[a][m] - M[b][m] for m in M[a]}, 'fold_PP90': {int(f): ap(yp[fold == f], A[a][fold == f]) - ap(yp[fold == f], A[b][fold == f]) for f in (1, 2)}}
    out = {'n': len(s), 'tail_pos_PP90': int(yp[T].sum()), 'abs': M, 'SXB16_minus_SXN16': dd('SXB16', 'SXN16'), 'SXB8_minus_SXN8': dd('SXB8', 'SXN8'), 'MIX32_minus_SXN16': dd('MIX32', 'SXN16')}
    rb = np.random.default_rng(13); ut = np.unique(tab); by = {t: np.where(tab == t)[0] for t in ut}; bt = []
    for _ in range(300):
        idx = np.concatenate([by[t] for t in rb.choice(ut, len(ut), replace=True)])
        bt.append(ap(yp[idx], A['SXB16'][idx]) - ap(yp[idx], A['SXN16'][idx]))
    out['boot_SXB16_minus_SXN16_PP90'] = [float(np.quantile(bt, .05)), float(np.median(bt)), float(np.quantile(bt, .95)), float(np.mean(np.array(bt) > 0))]
    res['surfaces'][p] = out
    v = out['SXB16_minus_SXN16']
    print(p, 'tailpos', out['tail_pos_PP90'], f"SXB16-SXN16 dPP90 {v['PP90']:+.4f} dPPcls {v['PPcls']:+.4f} dB {v['B']:+.4f} folds {v['fold_PP90'][1]:+.4f}/{v['fold_PP90'][2]:+.4f}",
          f"| SXB8-SXN8 {out['SXB8_minus_SXN8']['PP90']:+.4f} | MIX32-SXN16 {out['MIX32_minus_SXN16']['PP90']:+.4f}", 'boot', [round(x, 4) for x in out['boot_SXB16_minus_SXN16_PP90']], flush=True)
S = res['surfaces']
d90 = [S[p]['SXB16_minus_SXN16']['PP90'] for p in S]; dcl = [S[p]['SXB16_minus_SXN16']['PPcls'] for p in S]
res['rule'] = {'mean_dPP90': float(np.mean(d90)), 'min_surface_dPP90': float(np.min(d90)), 'mean_dPPcls': float(np.mean(dcl)),
               'i': bool(np.mean(d90) > 0), 'ii': bool(np.min(d90) >= -.0005), 'iii': bool(np.mean(dcl) > 0)}
res['rule']['SXB16_REPLACES_SXN16'] = res['rule']['i'] and res['rule']['ii'] and res['rule']['iii']
res['rule']['mean_comp_SXB16_minus_SXN16'] = float(np.mean([.7 * S[p]['SXB16_minus_SXN16']['PP90'] + .1 * S[p]['SXB16_minus_SXN16']['B'] for p in S]))
json.dump(res, open(TB + 'sxb_dev_eval.json', 'w', encoding='utf-8'), indent=1)
print('RULE', json.dumps(res['rule']))
