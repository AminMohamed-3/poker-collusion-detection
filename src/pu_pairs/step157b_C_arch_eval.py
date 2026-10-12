"""Step 157b (Claude): registered readings of outputs_step156_20260915/prediction_step157.json (sha 1a5208f4): SA16 (5 incumbent view types +
XGB3, ET, MLP) against INC16 (SXN16 if STEP156 Part B passed, else SX16). STEP131 surfaces and label definitions as step152b.
Output experiments/arch_views_20260915/arch_dev_eval.json. Development labels only."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, hashlib
sys.dont_write_bytecode = True
import numpy as np, polars as pl
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
E = C + 'experiments/'
X, VD, TK, TN, TA = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/', E + 'tail_keys_20260914/', E + 'tail_keys_20260915_neg/', E + 'arch_views_20260915/'
REG = '1a5208f4aa5e9be7eab9b9106a7c317c87816f8bf1d95dcd0db71005b477aee5'
assert hashlib.sha256(open(C + 'outputs_step156_20260915/prediction_step157.json', 'rb').read()).hexdigest() == REG
NEG = int(json.load(open(TN + 'sxn_dev_eval.json', encoding='utf-8'))['rule']['SXN16_REPLACES_SX16'])
INC_VT = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']; NEW_VT = ['XGB3', 'ET', 'MLP']
FAM = {1: 'directed_transfer', 2: 'soft_play', 3: 'coordinated_isolation'}
K = 16


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def ap(y, s):
    o = np.argsort(-s, kind='mergesort'); yy = y[o]; tp = np.cumsum(yy)
    return float((tp / np.arange(1, len(yy) + 1) * yy).sum() / max(1, int(yy.sum())))


def permute(s, z, blk):
    s2 = s.copy(); idx = np.where(blk)[0]; s2[idx[np.argsort(-z[idx], kind='mergesort')]] = np.sort(s[idx])[::-1]; return s2


def bmap(sc, lab, fam, pred):
    return float(np.mean([ap(((lab == 1) & (fam == FAM[c])).astype(int), np.where(pred == c, sc, 0.0)) for c in (1, 2, 3)]))


def load_mean(folder, prefix, vts):
    d = pl.concat([pl.read_parquet(folder + f) for f in sorted(os.listdir(folder)) if f.startswith(prefix) and f.endswith('.parquet')]).filter(pl.col('k') < K)
    assert d.select('fold', 'k').unique().height == 2 * K
    d = d.with_columns([pl.Series(f'L_{v}', lg(d[f'r_{v}'])) for v in vts])
    return d.group_by('pair_id', 'fold', 'surface').agg([pl.col(f'L_{v}').mean() for v in vts])


inc = load_mean(TN, 'sxn_dev_fold', INC_VT) if NEG else load_mean(TK, 'sx_dev_fold', INC_VT)
new = load_mean(TA, f'arch_dev_neg{NEG}_fold', NEW_VT)
V = inc.join(new, on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1'); assert V.height == inc.height == new.height
ALLV = INC_VT + NEW_VT
V = V.with_columns(pl.mean_horizontal([f'L_{v}' for v in INC_VT]).alias('INC'), pl.mean_horizontal([f'L_{v}' for v in ALLV]).alias('SA'),
                   *[pl.mean_horizontal([f'L_{v}' for v in INC_VT + [n]]).alias(f'INC+{n}') for n in NEW_VT])
R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
res = {'registration_sha256': REG, 'incumbent': 'SXN16' if NEG else 'SX16', 'surfaces': {}}
for p in ('prefix40', 'prefix60', 'prefix80'):
    parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
             .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
             .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
    d = (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
         .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))
    keys = ['INC', 'SA'] + [f'INC+{n}' for n in NEW_VT] + [f'L_{n}' for n in NEW_VT]
    assert all(d[k].null_count() == 0 for k in keys)
    s, lab, fold, tab = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['fold'].to_numpy(), d['table_id'].to_numpy()
    fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
    yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
    yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
    T = s < .5
    A = {k: permute(s, d[k].to_numpy(), T) for k in keys}
    M = {k: {'PP90': ap(yp, sc), 'PPcls': ap(yc, sc), 'RAW': ap(yr, sc), 'B': bmap(sc, lab, fam, pred)} for k, sc in A.items()}
    out = {'n': len(s), 'tail_pos_PP90': int(yp[T].sum()), 'abs': M,
           'minus_INC': {k: {**{m: M[k][m] - M['INC'][m] for m in M[k]}, 'fold_PP90': {int(f): ap(yp[fold == f], A[k][fold == f]) - ap(yp[fold == f], A['INC'][fold == f]) for f in (1, 2)}} for k in keys if k != 'INC'}}
    rb = np.random.default_rng(13); ut = np.unique(tab); by = {t: np.where(tab == t)[0] for t in ut}; bt = []
    for _ in range(300):
        idx = np.concatenate([by[t] for t in rb.choice(ut, len(ut), replace=True)])
        bt.append(ap(yp[idx], A['SA'][idx]) - ap(yp[idx], A['INC'][idx]))
    out['boot_SA_minus_INC_PP90'] = [float(np.quantile(bt, .05)), float(np.median(bt)), float(np.quantile(bt, .95)), float(np.mean(np.array(bt) > 0))]
    res['surfaces'][p] = out
    print(p, 'tailpos', out['tail_pos_PP90'], ' '.join(f"{k}: {out['minus_INC'][k]['PP90']:+.4f}" for k in out['minus_INC']),
          '| SA dPPcls %+.4f dB %+.4f folds %+.4f/%+.4f' % (out['minus_INC']['SA']['PPcls'], out['minus_INC']['SA']['B'], out['minus_INC']['SA']['fold_PP90'][1], out['minus_INC']['SA']['fold_PP90'][2]),
          'boot', [round(x, 4) for x in out['boot_SA_minus_INC_PP90']], flush=True)
S = res['surfaces']
d90 = [S[p]['minus_INC']['SA']['PP90'] for p in S]; dcl = [S[p]['minus_INC']['SA']['PPcls'] for p in S]
res['rule'] = {'mean_dPP90': float(np.mean(d90)), 'min_surface_dPP90': float(np.min(d90)), 'mean_dPPcls': float(np.mean(dcl)),
               'i': bool(np.mean(d90) > 0), 'ii': bool(np.min(d90) >= -.0005), 'iii': bool(np.mean(dcl) > 0)}
res['rule']['SA16_REPLACES_INC16'] = res['rule']['i'] and res['rule']['ii'] and res['rule']['iii']
res['rule']['mean_comp_SA16_minus_INC16'] = float(np.mean([.7 * S[p]['minus_INC']['SA']['PP90'] + .1 * S[p]['minus_INC']['SA']['B'] for p in S]))
json.dump(res, open(TA + 'arch_dev_eval.json', 'w', encoding='utf-8'), indent=1)
print('RULE', json.dumps(res['rule']))
