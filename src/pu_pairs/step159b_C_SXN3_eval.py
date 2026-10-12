"""Step 159b (Claude): registered part N reading of outputs_step159_20260915/prediction_step159.json (sha a77dcb64): SXN3_16 (3x the
unlabelled-negative dose) against SXN16 (STEP156 Part B). STEP131 surfaces, labels and AP as step152b/156d; descriptive per-type deltas.
Output experiments/tail_keys_20260915_neg3/sxn3_dev_eval.json. Development labels only."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, hashlib
sys.dont_write_bytecode = True
import numpy as np, polars as pl
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
E = C + 'experiments/'
X, VD, TN, T3 = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/', E + 'tail_keys_20260915_neg/', E + 'tail_keys_20260915_neg3/'
REG = 'a77dcb643b129e50b956d4d95a889eb1c5c7b4080ab6dc0bae38011ee53a2252'
assert hashlib.sha256(open(C + 'outputs_step159_20260915/prediction_step159.json', 'rb').read()).hexdigest() == REG
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


def agg(folder, prefix, name):
    d = pl.concat([pl.read_parquet(folder + f) for f in sorted(os.listdir(folder)) if f.startswith(prefix) and f.endswith('.parquet')]).filter(pl.col('k') < 16)
    assert d.select('fold', 'k').unique().height == 32
    d = d.with_columns([pl.Series(f'L_{v}', lg(d[f'r_{v}'])) for v in VT])
    a = d.group_by('pair_id', 'fold', 'surface').agg([pl.col(f'L_{v}').mean().alias(f'{name}_{v}') for v in VT])
    return a.with_columns(pl.mean_horizontal([f'{name}_{v}' for v in VT]).alias(name))


AG = [agg(TN, 'sxn_dev_fold', 'SXN16'), agg(T3, 'sxn3_dev_fold', 'SXN3_16')]
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
    keys = ['SXN16', 'SXN3_16'] + [f'{n}_{v}' for n in ('SXN16', 'SXN3_16') for v in VT]
    assert all(d[k].null_count() == 0 for k in keys)
    s, lab, fold, tab = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['fold'].to_numpy(), d['table_id'].to_numpy()
    fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
    yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
    yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
    T = s < .5
    A = {k: permute(s, d[k].to_numpy(), T) for k in keys}
    M = {k: {'PP90': ap(yp, sc), 'PPcls': ap(yc, sc), 'RAW': ap(yr, sc), 'B': bmap(sc, lab, fam, pred)} for k, sc in A.items()}
    v = {**{m: M['SXN3_16'][m] - M['SXN16'][m] for m in M['SXN16']},
         'fold_PP90': {int(f): ap(yp[fold == f], A['SXN3_16'][fold == f]) - ap(yp[fold == f], A['SXN16'][fold == f]) for f in (1, 2)}}
    out = {'n': len(s), 'tail_pos_PP90': int(yp[T].sum()), 'abs': M, 'SXN3_16_minus_SXN16': v,
           'per_view_PP90': {t: M[f'SXN3_16_{t}']['PP90'] - M[f'SXN16_{t}']['PP90'] for t in VT}}
    rb = np.random.default_rng(13); ut = np.unique(tab); by = {t: np.where(tab == t)[0] for t in ut}; bt = []
    for _ in range(300):
        idx = np.concatenate([by[t] for t in rb.choice(ut, len(ut), replace=True)])
        bt.append(ap(yp[idx], A['SXN3_16'][idx]) - ap(yp[idx], A['SXN16'][idx]))
    out['boot_PP90'] = [float(np.quantile(bt, .05)), float(np.median(bt)), float(np.quantile(bt, .95)), float(np.mean(np.array(bt) > 0))]
    res['surfaces'][p] = out
    print(p, 'tailpos', out['tail_pos_PP90'], f"SXN3_16-SXN16 dPP90 {v['PP90']:+.4f} dPPcls {v['PPcls']:+.4f} dRAW {v['RAW']:+.4f} dB {v['B']:+.4f} folds {v['fold_PP90'][1]:+.4f}/{v['fold_PP90'][2]:+.4f}",
          '| per view', {t: round(x, 4) for t, x in out['per_view_PP90'].items()}, 'boot', [round(x, 4) for x in out['boot_PP90']], flush=True)
S = res['surfaces']
d90 = [S[p]['SXN3_16_minus_SXN16']['PP90'] for p in S]; dcl = [S[p]['SXN3_16_minus_SXN16']['PPcls'] for p in S]
res['rule'] = {'mean_dPP90': float(np.mean(d90)), 'min_surface_dPP90': float(np.min(d90)), 'mean_dPPcls': float(np.mean(dcl)),
               'i': bool(np.mean(d90) > 0), 'ii': bool(np.min(d90) >= -.0005), 'iii': bool(np.mean(dcl) > 0)}
res['rule']['SXN3_16_REPLACES_SXN16'] = res['rule']['i'] and res['rule']['ii'] and res['rule']['iii']
res['rule']['mean_comp_SXN3_16_minus_SXN16'] = float(np.mean([.7 * S[p]['SXN3_16_minus_SXN16']['PP90'] + .1 * S[p]['SXN3_16_minus_SXN16']['B'] for p in S]))
json.dump(res, open(T3 + 'sxn3_dev_eval.json', 'w', encoding='utf-8'), indent=1)
print('RULE', json.dumps(res['rule']))
