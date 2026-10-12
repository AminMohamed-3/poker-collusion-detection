"""Screen (Claude, 2026-09-15, descriptive; dev folds 1-2 re-read): behaviour family headroom and multi-draw OVR family predictors.
Risk = STEP131 surface risk with the SX32 tail re-order (production analogue). Predictors fixed before reading:
 V13 (current predicted_class), O32 (argmax of 32-draw mean lg r_DT/r_SP/r_CI), O32T (O32 in tail only), O32B (O32 in block only), ORACLE (true family for label==1)."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json
sys.dont_write_bytecode = True
import numpy as np, polars as pl
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
E = C + 'experiments/'
O, X, VD, TK = E + 'tail_view_confirmation_a8_20260913/', E + 'pu_pairs_20260911/', E + 'viewdis_20260911/', E + 'tail_keys_20260914/'
VT = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']
FAM = {1: 'directed_transfer', 2: 'soft_play', 3: 'coordinated_isolation'}
INV = {v: k for k, v in FAM.items()}
def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))
def ap(y, s):
    o = np.argsort(-s, kind='mergesort'); yy = y[o]; tp = np.cumsum(yy)
    return float((tp / np.arange(1, len(yy) + 1) * yy).sum() / max(1, int(yy.sum())))
def permute(s, z, blk):
    s2 = s.copy(); idx = np.where(blk)[0]; s2[idx[np.argsort(-z[idx], kind='mergesort')]] = np.sort(s[idx])[::-1]; return s2
def bfam(sc, lab, fam, pred):
    return {c: ap(((lab == 1) & (fam == FAM[c])).astype(int), np.where(pred == c, sc, 0.0)) for c in (1, 2, 3)}
SX = pl.concat([pl.read_parquet(TK + f) for f in sorted(os.listdir(TK)) if f.startswith('sx_dev_fold') and f.endswith('.parquet')])
SXL = SX.with_columns([pl.Series(f'L_{v}', lg(SX[f'r_{v}'])) for v in VT + ['DT', 'SP', 'CI']])
A = SXL.group_by('pair_id', 'fold', 'surface').agg([pl.col(f'L_{v}').mean() for v in VT + ['DT', 'SP', 'CI']])
A = A.with_columns(pl.mean_horizontal([f'L_{v}' for v in VT]).alias('SX32'))
A1 = SXL.filter(pl.col('k') == 0).select('pair_id', 'fold', 'surface', pl.col('L_DT').alias('D1'), pl.col('L_SP').alias('S1'), pl.col('L_CI').alias('C1'))
res = {}
for p in ('prefix40', 'prefix60', 'prefix80'):
    parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk']).with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
    d = pl.concat(parts).join(A.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left').join(A1.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
    d = d.unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).sort('pair_id')
    s, lab, fam, pred = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
    T = s < .5
    sc = permute(s, d['SX32'].to_numpy(), T)
    o32 = np.argmax(np.stack([d['L_DT'], d['L_SP'], d['L_CI']], 1), 1) + 1
    o1 = np.argmax(np.stack([d['D1'], d['S1'], d['C1']], 1), 1) + 1
    truef = np.array([INV.get(f, 0) for f in fam])
    orc = np.where(lab == 1, truef, pred)
    P = {'V13': pred, 'O32': o32, 'O1': o1, 'O32T': np.where(T, o32, pred), 'O32B': np.where(T, pred, o32), 'ORACLE': orc}
    pos = lab == 1
    out = {'n': len(s), 'pos': int(pos.sum()), 'pos_block': int((pos & ~T).sum())}
    for k, pr in P.items():
        bf = bfam(sc, lab, fam, pr)
        out[k] = {'B': float(np.mean(list(bf.values()))), 'by_fam': bf, 'wrong_pos_block': int((pos & ~T & (pr != truef)).sum()), 'wrong_pos_tail': int((pos & T & (pr != truef)).sum()),
                  'changed_vs_V13': int((pr != pred).sum()), 'changed_block': int(((pr != pred) & ~T).sum())}
    # agreement of O32 with V13 on all pairs, and confusion among positives
    out['agree_O32_V13_all'] = float(np.mean(o32 == pred))
    wrongV = pos & (pred != truef)
    out['V13_wrong_pos_detail'] = [{'true': int(truef[i]), 'v13': int(pred[i]), 'o32': int(o32[i]), 'risk': round(float(s[i]), 3), 'fold': int(d['fold'].to_numpy()[i]), 'sh': int(d['shared_hands'].to_numpy()[i])} for i in np.where(wrongV)[0]]
    res[p] = out
    print(p, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'by_fam'}) for k, v in out.items() if k != 'V13_wrong_pos_detail'}), flush=True)
    print('  V13 wrong positives', out['V13_wrong_pos_detail'], flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
