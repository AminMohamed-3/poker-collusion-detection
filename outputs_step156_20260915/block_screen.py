"""Descriptive screen (Claude, 2026-09-15): does the M3 block blend gain from multi-draw SHORT3/ALL5? Cross-fold testbed of step 64f (prefix-60 block, def A / C).
Arms fixed before reading: M3 (stored single-draw views, reproduces step 64f), M3x (SHORT3/ALL5 -> 32-draw logit means), M3x16, M3s (SHORT3/ALL5 replaced by one SX32 key column)."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Claude Version/src')
import numpy as np, polars as pl
from pu_pairs.step64f_M3_candidate import testbed, FS_M3
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step60_multiview_blend import ap, logreg, predict
from pu_pairs.step62_multiview_candidate import L2, DEFS
from pu_pairs.build_allpairs_fold import fold_pools
TK = 'C:/Data/Projects/Money Making/work/poker/Claude Version/experiments/tail_keys_20260914/'
VT = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']
def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))
def sig(z): return 1 / (1 + np.exp(-z))
SX = pl.concat([pl.read_parquet(TK + f) for f in sorted(os.listdir(TK)) if f.startswith('sx_dev_fold') and f.endswith('.parquet')]).filter(pl.col('surface') == 'prefix60')
SXL = SX.with_columns([pl.Series(f'L_{v}', lg(SX[f'r_{v}'])) for v in VT])
agg = {}
for K in (16, 32):
    a = SXL.filter(pl.col('k') < K).group_by('pair_id', 'fold').agg([pl.col(f'L_{v}').mean() for v in VT])
    agg[K] = a.with_columns(pl.mean_horizontal([f'L_{v}' for v in VT]).alias('SXk'))
k0 = SX.filter(pl.col('k') == 0).select('pair_id', 'fold', pl.col('r_SHORT3').alias('k0_S3'), pl.col('r_ALL5').alias('k0_A5'))
d = testbed()
print('testbed cols', [c for c in d.columns if c in ('fold', 'fold_new')], d.height)
fcol = 'fold' if 'fold' in d.columns else 'fold_new'
f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2
out = {}
for dn, (posset, negset) in DEFS.items():
    c = d.filter(pl.col('cls').is_in(posset + negset))
    # candidate fold assignment: which fold's out-of-pool views match the stored single-draw views
    j = c.select('pair_id', 'r_SHORT3', 'r_ALL5').join(k0, on='pair_id', how='left')
    j = j.with_columns(((pl.col('k0_S3') - pl.col('r_SHORT3')).abs() + (pl.col('k0_A5') - pl.col('r_ALL5')).abs()).alias('dif')).sort('pair_id', 'dif').unique(subset=['pair_id'], keep='first', maintain_order=True)
    gate = float(j['dif'].max()); fm = j.select('pair_id', 'fold')
    c = c.join(fm, on='pair_id', how='left', suffix='_m')
    fc = 'fold_m' if 'fold_m' in c.columns else 'fold'
    for K in (16, 32):
        a = agg[K].rename({'fold': fc}) if fc != 'fold' else agg[K]
        c = c.join(a.select('pair_id', fc, *[pl.col(f'L_{v}').alias(f'L{K}_{v}') for v in VT], pl.col('SXk').alias(f'SX{K}')), on=['pair_id', fc], how='left')
    y = c['cls'].is_in(posset).to_numpy().astype(float); tab = c['table_id'].to_numpy()
    in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
    def cf(c2, fs):
        X = X_of(c2, fs, 'risk_prefix', 'R1_prefix', 'risk_noT'); pred = np.full(len(y), np.nan)
        m1 = logreg(X[in1], y[in1], L2); pred[in2 | insh] = predict(m1, X[in2 | insh]); m2 = logreg(X[in2], y[in2], L2); pred[in1] = predict(m2, X[in1])
        return {'pooled': ap(y, pred), 'test_f2sh': ap(y[in2 | insh], pred[in2 | insh]), 'test_f1': ap(y[in1], pred[in1])}
    r = {'n_pos': int(y.sum()), 'n': len(y), 'gate_k0_vs_stored_maxdiff': gate, 'null_SX32': int(c['SX32'].null_count())}
    r['M3'] = cf(c, FS_M3)
    for K in (16, 32):
        cx = c.with_columns(pl.Series('r_SHORT3', sig(c[f'L{K}_SHORT3'].to_numpy())), pl.Series('r_ALL5', sig(c[f'L{K}_ALL5'].to_numpy())))
        r[f'M3x{K}'] = cf(cx, FS_M3)
    cs = c.with_columns(pl.Series('r_SX', sig(c['SX32'].to_numpy())))
    r['M3s'] = cf(cs, [f for f in FS_M3 if f not in ('SHORT3', 'ALL5')] + ['SX'])
    r['frozen'] = {'pooled': ap(y, c['risk_prefix'].to_numpy())}
    out[dn] = r
    print(dn, json.dumps({k: ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in r.items()}), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
