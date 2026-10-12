"""Step 156d (Claude): registered Part B readings of outputs_step156_20260915/prediction_step156.json (sha fba91dde): SXN16 vs SX16 (and SXN8 vs SX8,
descriptive) on the STEP131 surfaces (dev folds 1-2, prefix 40/60/80, >= 38 shared hands), label definitions and AP as step152b.
Gate: every (fold, draw k < 16) present for SXN; SX draws k < 16 present. Output experiments/tail_keys_20260915_neg/sxn_dev_eval.json. Development labels only."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, hashlib
sys.dont_write_bytecode = True
import numpy as np, polars as pl
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
E = C + 'experiments/'
X, VD, TK, TN = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/', E + 'tail_keys_20260914/', E + 'tail_keys_20260915_neg/'
REG = 'fba91ddedfcb556ef2be83f0c6ea5a3edea57b703de150479292bcc374a7e43d'
assert hashlib.sha256(open(C + 'outputs_step156_20260915/prediction_step156.json', 'rb').read()).hexdigest() == REG
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


def load(folder, prefix, K):
    d = pl.concat([pl.read_parquet(folder + f) for f in sorted(os.listdir(folder)) if f.startswith(prefix) and f.endswith('.parquet')]).filter(pl.col('k') < K)
    assert d.select('fold', 'k').unique().height == 2 * K, d.select('fold', 'k').unique().height
    return d


def agg(d, K, name):
    L = d.filter(pl.col('k') < K).with_columns([pl.Series(f'L_{v}', lg(d.filter(pl.col('k') < K)[f'r_{v}'])) for v in VT])
    a = L.group_by('pair_id', 'fold', 'surface').agg([pl.col(f'L_{v}').mean().alias(f'{name}{K}_{v}') for v in VT])
    return a.with_columns(pl.mean_horizontal([f'{name}{K}_{v}' for v in VT]).alias(f'{name}{K}'))


SX, SXN = load(TK, 'sx_dev_fold', 16), load(TN, 'sxn_dev_fold', 16)
assert SXN.height == SX.height, (SXN.height, SX.height)
AG = [agg(SX, 8, 'SX'), agg(SX, 16, 'SX'), agg(SXN, 8, 'SXN'), agg(SXN, 16, 'SXN')]
R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})


def surface(p):
    parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
             .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
             .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
    d = pl.concat(parts)
    for a in AG:
        d = d.join(a.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
    return d.unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id')


res = {'registration_sha256': REG, 'surfaces': {}}
for p in ('prefix40', 'prefix60', 'prefix80'):
    d = surface(p)
    keys = ['SX8', 'SX16', 'SXN8', 'SXN16']
    assert all(d[k].null_count() == 0 for k in keys)
    s, lab, fold, tab = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['fold'].to_numpy(), d['table_id'].to_numpy()
    fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
    yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
    yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
    T = s < .5
    KEYS = {k: d[k].to_numpy() for k in keys}
    for v in VT:  # per view type, descriptive
        KEYS[f'SX16_{v}'] = d[f'SX16_{v}'].to_numpy(); KEYS[f'SXN16_{v}'] = d[f'SXN16_{v}'].to_numpy()
    A = {k: permute(s, z, T) for k, z in KEYS.items()}
    M = {k: {'PP90': ap(yp, sc), 'PPcls': ap(yc, sc), 'RAW': ap(yr, sc), 'B': bmap(sc, lab, fam, pred)} for k, sc in A.items()}
    out = {'n': len(s), 'n_tail': int(T.sum()), 'tail_pos_PP90': int(yp[T].sum()), 'abs': M}
    def dd(a, b):
        return {**{m: M[a][m] - M[b][m] for m in M[a]}, 'fold_PP90': {int(f): ap(yp[fold == f], A[a][fold == f]) - ap(yp[fold == f], A[b][fold == f]) for f in (1, 2)}}
    out['SXN16_minus_SX16'] = dd('SXN16', 'SX16'); out['SXN8_minus_SX8'] = dd('SXN8', 'SX8')
    out['per_view_SXN16_minus_SX16_PP90'] = {v: M[f'SXN16_{v}']['PP90'] - M[f'SX16_{v}']['PP90'] for v in VT}
    rb = np.random.default_rng(13); ut = np.unique(tab); by = {t: np.where(tab == t)[0] for t in ut}; bt = []
    for _ in range(300):
        idx = np.concatenate([by[t] for t in rb.choice(ut, len(ut), replace=True)])
        bt.append(ap(yp[idx], A['SXN16'][idx]) - ap(yp[idx], A['SX16'][idx]))
    out['boot_SXN16_minus_SX16_PP90'] = [float(np.quantile(bt, .05)), float(np.median(bt)), float(np.quantile(bt, .95)), float(np.mean(np.array(bt) > 0))]
    tidx = np.where(T)[0]
    for k in ('SX16', 'SXN16'):
        z = KEYS[k][tidx]; rank = np.empty(len(tidx)); rank[np.argsort(-z, kind='mergesort')] = np.arange(len(tidx))
        out[f'tail_pos_ranks_{k}'] = sorted(int(x) for x in rank[yp[tidx] == 1])
    res['surfaces'][p] = out
    v = out['SXN16_minus_SX16']
    print(p, 'tailpos', out['tail_pos_PP90'], f"SXN16-SX16 dPP90 {v['PP90']:+.4f} dPPcls {v['PPcls']:+.4f} dRAW {v['RAW']:+.4f} dB {v['B']:+.4f} folds {v['fold_PP90'][1]:+.4f}/{v['fold_PP90'][2]:+.4f}",
          f"| SXN8-SX8 {out['SXN8_minus_SX8']['PP90']:+.4f}", 'boot', [round(x, 4) for x in out['boot_SXN16_minus_SX16_PP90']],
          'views', {k: round(x, 4) for k, x in out['per_view_SXN16_minus_SX16_PP90'].items()}, flush=True)
    print('  ranks SX16', out['tail_pos_ranks_SX16'], '\n  ranks SXN16', out['tail_pos_ranks_SXN16'], flush=True)
S = res['surfaces']
d90 = [S[p]['SXN16_minus_SX16']['PP90'] for p in S]; dcl = [S[p]['SXN16_minus_SX16']['PPcls'] for p in S]
res['rule'] = {'mean_dPP90': float(np.mean(d90)), 'min_surface_dPP90': float(np.min(d90)), 'mean_dPPcls': float(np.mean(dcl)),
               'i': bool(np.mean(d90) > 0), 'ii': bool(np.min(d90) >= -.0005), 'iii': bool(np.mean(dcl) > 0)}
res['rule']['SXN16_REPLACES_SX16'] = res['rule']['i'] and res['rule']['ii'] and res['rule']['iii']
res['rule']['mean_comp_SXN16_minus_SX16'] = float(np.mean([.7 * S[p]['SXN16_minus_SX16']['PP90'] + .1 * S[p]['SXN16_minus_SX16']['B'] for p in S]))
json.dump(res, open(TN + 'sxn_dev_eval.json', 'w', encoding='utf-8'), indent=1)
print('RULE', json.dumps(res['rule']))
