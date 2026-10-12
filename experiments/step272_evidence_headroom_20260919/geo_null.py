"""STEP272-V: null calibration of the STEP272-U grid, and a fold-partition robustness read.

The +.00438 arm (MECH_fuse | SP | F1 | w.20) is the maximum over 384 grid cells, so the question is how
large a maximum the same grid produces from a score with NO information.  Null: permute the new score
WITHIN each pair (marginal and the decoder's magnitude distribution preserved, ordering destroyed), 24
replicates, same grid, recording (a) the max decoded delta per replicate and (b) the value at the chosen
cell.  p = fraction of replicates reaching the arm.
"""
import os, sys, json
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '2'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np, polars as pl
R = Path('C:/Data/Projects/Money Making/work/poker'); S = Path(os.environ['SCR']); CV = R / 'Claude Version'
OUT = CV / 'experiments/step272_evidence_headroom_20260919'
sys.path.insert(0, str(CV / 'src')); sys.path.insert(0, str(R / 'Codex Resume/src')); sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step5_evaluate import half
K = S165.K; CI = 'coordinated_isolation'
D = S165.load(); d, Q, QIX, dec = D['d'], D['Q'], D['QIX'], D['dec']
pidall = d['pair_id'].to_numpy()
truths = {kk[0]: set(gg['hand_id']) for kk, gg in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
key = d.select('pair_id', 'hand_id', 'table_id'); qids = sorted(truths)
fq = np.array([fams[q] for q in qids]); hv = np.array([half(ptab[q]) for q in qids])
qfam = Q['behavior_family'].to_numpy(); qpid = Q['pair_id'].to_numpy()
PROD = np.load(S / 'inc_p_PRODbare.npy')
SRC = {n: np.load(S / ('inc_p_%s.npy' % n)) for n in ('MECH_CTX', 'MECH', 'MECH_CTX_fuse', 'MECH_fuse')}
GRP = [np.flatnonzero(qpid == p) for p in np.unique(qpid)]
SCOPES = {'DTSP': ('directed_transfer', 'soft_play'), 'ALL': ('directed_transfer', 'soft_play', CI),
          'DT': ('directed_transfer',), 'SP': ('soft_play',)}


def dec5(p):
    s = np.zeros(len(pidall)); s[QIX] = 10.0 + p
    _, mm = summarize(key.with_columns(pl.Series('score', dec(s))), truths, fams)
    return np.array([mm[q] for q in qids])


def shift(p, F):
    q = np.clip(p, 1e-9, 1 - 1e-9); o = q / (1 - q) * F
    return o / (1 + o)


def fuse(pn, w, F, scope):
    m = np.isin(qfam, SCOPES[scope]); out = PROD.copy()
    out[m] = np.clip(PROD[m], 1e-9, 1.) ** (1 - w) * shift(pn, F)[m] ** w
    return out


D0 = dec5(PROD)
CELL = ('MECH_fuse', 'SP', 1., .20)
GRID = [(s, sc, F, w) for s in SRC for sc in ('DTSP', 'ALL', 'DT', 'SP') for F in (1., 3., 10., 30.) for w in (.10, .15, .20, .25, .30, .40)]
arm_cell = dec5(fuse(SRC[CELL[0]], CELL[3], CELL[2], CELL[1])).mean() - D0.mean()
arm_max = max(dec5(fuse(SRC[s], w, F, sc)).mean() - D0.mean() for s, sc, F, w in GRID)
print('ARM  chosen cell %+.6f   grid max %+.6f' % (arm_cell, arm_max), flush=True)
print('', flush=True)
mx = []; cl = []
for rep in range(24):
    rng = np.random.default_rng(9000 + rep)
    PERM = {}
    for n, v in SRC.items():
        q = v.copy()
        for g in GRP:
            q[g] = v[g][rng.permutation(len(g))]
        PERM[n] = q
    vals = {}
    for s, sc, F, w in GRID:
        vals[(s, sc, F, w)] = dec5(fuse(PERM[s], w, F, sc)).mean() - D0.mean()
    mx.append(max(vals.values())); cl.append(vals[CELL])
    print('rep %2d  null grid max %+.6f   null at the chosen cell %+.6f' % (rep, mx[-1], cl[-1]), flush=True)
mx = np.array(mx); cl = np.array(cl)
res = {'arm_cell': round(float(arm_cell), 6), 'arm_grid_max': round(float(arm_max), 6),
       'null_max_mean': round(float(mx.mean()), 6), 'null_max_sd': round(float(mx.std()), 6),
       'null_max_q95': round(float(np.quantile(mx, .95)), 6), 'null_max_max': round(float(mx.max()), 6),
       'p_null_max_ge_arm': round(float((mx >= arm_cell).mean()), 4),
       'null_cell_mean': round(float(cl.mean()), 6), 'null_cell_sd': round(float(cl.std()), 6),
       'p_null_cell_ge_arm': round(float((cl >= arm_cell).mean()), 4), 'replicates': len(mx)}
print('', flush=True)
print('null grid max: mean %+.6f sd %.6f q95 %+.6f max %+.6f   p(null max >= arm cell) = %.4f' % (
    mx.mean(), mx.std(), np.quantile(mx, .95), mx.max(), (mx >= arm_cell).mean()), flush=True)
print('null at the chosen cell: mean %+.6f sd %.6f   p(>= arm) = %.4f' % (cl.mean(), cl.std(), (cl >= arm_cell).mean()), flush=True)
json.dump(res, open(OUT / 'geo_null.json', 'w'), indent=2)
print('wrote geo_null.json', flush=True)
