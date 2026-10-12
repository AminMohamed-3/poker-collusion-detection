"""X2: un-pool the KEY.  A pre-specified extrapolation of the only pair-side effect the leaderboard has
CONFIRMED, chosen without any internal screening because X1 proved no internal instrument can be calibrated.

The production blend feeds the ten F10 members in as ONE column: KEY = unweighted mean of ten raw logits.
Nothing chose those weights -- they are an artefact of averaging unstandardised logits (W7: the implied
weights run .059 to .170 and are close to ANTI-correlated with usefulness).

STEP272 un-pooled exactly ONE member (ARCH2_MLP2, its own weight alongside KEY) and read **+.00020 public**.
That is the confirmed effect. STEP277 is its pre-specified continuation: give ALL TEN members their own
weight, i.e. replace KEY by its constituents.

Why this is the right class:
  * LINEAR -- no interaction, no hinge, no function of risk_noT, so it cannot exploit the risk_full
    circularity that made STEP273-275 read -.00101.
  * It ADDS INFORMATION -- nine columns currently collapsed into a mean become separately visible.
  * Transfer evidence: new measurement added linearly +0.51, re-weighting +0.19, form change -0.32.

Discipline, fixed before running:
  * l2 stays at the PRODUCTION value 10.0. No sweep, no tuning.
  * ONE arm. No family, no max-over-candidates, so there is no multiplicity to correct and no selection.
  * Internal numbers are reported for the record but are NOT a gate -- X1 showed they do not predict sign.
"""
import os, sys, json, time
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '4')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
SC = ('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/'
      'e91ed0b3-2ad5-45d4-b30c-115132b295b0/scratchpad/pair/')
sys.path.insert(0, C + 'src'); sys.path.insert(0, SC)
import numpy as np, polars as pl
from pu_pairs.step60_multiview_blend import ap, lg, logreg
from pu_pairs.step62_multiview_candidate import DEFS
from pu_pairs.step63_softR1_candidate import X_of
import harness as H

F10 = ['SXN3_SHORT3', 'SXN3_ALL5', 'SXN3_LIN30', 'SXN3_LIN300', 'SXN3_FAMMAX',
       'ARCH_XGB3', 'ARCH_ET', 'ARCH_MLP', 'ARCH2_MLP2', 'ARCH2_HGB5']
L2 = 10.0
t0 = time.time(); rec = {}

S = pl.concat([pl.read_parquet(C + 'experiments/pu_pairs_20260911/fold%d/scores.parquet' % f,
                               columns=['pair_id', 'shared_hands']) for f in (1, 2)]
              ).unique(subset=['pair_id'], keep='first')
V = pl.read_parquet(SC + 'V_arch3.parquet').filter(pl.col('surface') == 'prefix60').drop('surface')
d = H.load().join(V, on=['pair_id', 'fold'], how='left').join(
    S.select('pair_id', pl.col('shared_hands').alias('sh_full')), on='pair_id', how='left', validate='1:1')
in1, in2, insh = H.groups(d)
n = d.height
X0 = X_of(d, H.FS_M3, 'risk_prefix', 'R1_prefix', 'risk_noT')
M = d.select(F10).to_numpy()
KEY = M.mean(1)
M2 = d['ARCH2_MLP2'].to_numpy()
ELIG = d['sh_full'].to_numpy() >= 57
tab = d['table_id'].to_numpy()

certA = d['cls'].is_in(DEFS['A'][0] + DEFS['A'][1]).to_numpy(); yA = d['cls'].is_in(DEFS['A'][0]).to_numpy().astype(float)
certC = d['cls'].is_in(DEFS['C'][0] + DEFS['C'][1]).to_numpy(); yC = d['cls'].is_in(DEFS['C'][0]).to_numpy().astype(float)
DEFB = (['pos', 'undisclosed'], ['FP', 'undisclosed_R1', 'ambiguous', 'mid', 'neg'])
certB = d['cls'].is_in(DEFB[0] + DEFB[1]).to_numpy(); yB = d['cls'].is_in(DEFB[0]).to_numpy().astype(float)
ypos = (d['cls'] == 'pos').to_numpy().astype(float)
CE = {'A': (certA, yA), 'C': (certC, yC), 'B': (certB, yB)}

STATES = {
    'STEP270  X0+KEY            (13)': np.column_stack([X0, KEY]),
    'STEP272  X0+KEY+MLP2       (14)': np.column_stack([X0, KEY, M2]),
    'STEP277  X0+ten members    (22)': np.column_stack([X0, M]),
    'STEP277b X0+KEY+ten members(23)': np.column_stack([X0, KEY, M]),
}
P = {k: H.crossfold(v, yA, certA, in1, in2, insh, L2) for k, v in STATES.items()}

print('all states fitted at the PRODUCTION l2 = %g, no tuning\n' % L2, flush=True)
print('%-34s %9s %9s %9s   %s' % ('state', 'defA', 'defC', 'defB', '(eligible rows)'), flush=True)
out = {}
for k, p in P.items():
    r = {}
    for D in ('A', 'C', 'B'):
        cert, y = CE[D]
        s = cert & ELIG
        r[D] = float(ap(y[s], p[s]))
    out[k] = r
    print('%-34s %9.5f %9.5f %9.5f' % (k, r['A'], r['C'], r['B']), flush=True)
ref = out['STEP272  X0+KEY+MLP2       (14)']
print('\ndelta vs STEP272 (the current clean best, public .91352):', flush=True)
for k, r in out.items():
    if k.startswith('STEP272'):
        continue
    print('   %-34s dA %+.5f  dC %+.5f  dB %+.5f'
          % (k, r['A'] - ref['A'], r['C'] - ref['C'], r['B'] - ref['B']), flush=True)
rec['states'] = out

print('\nfitted member weights in STEP277 (standardised inputs, l2 = %g):' % L2, flush=True)
mk = logreg(np.column_stack([X0, M]), yA, L2)
w = np.asarray(mk[0] if isinstance(mk, tuple) else getattr(mk, 'coef_', mk)).ravel()
NAMES = list(H.FS_M3) + F10
imp = np.mean(np.abs(M - M.mean(0)) / (M.std(0) + 1e-12) * 0 + 1)  # placeholder to keep shapes obvious
for nm, wt in zip(NAMES, w[:len(NAMES)]):
    mark = '   <- pooled into KEY at weight %.3f' % (M.std(0)[F10.index(nm)] / M.std(0).sum()) if nm in F10 else ''
    print('   %-16s %+8.4f%s' % (nm, wt, mark), flush=True)
rec['weights'] = {nm: float(wt) for nm, wt in zip(NAMES, w[:len(NAMES)])}
rec['key_implied_weights'] = {c: float(s / M.std(0).sum()) for c, s in zip(F10, M.std(0))}

print('\npaired cluster bootstrap, STEP277 vs STEP272 (2000 table resamples):', flush=True)
boot = {}
for D in ('A', 'C'):
    cert, y = CE[D]
    s = np.where(cert & ELIG)[0]
    tabs = np.unique(tab[s]); by = {t: s[tab[s] == t] for t in tabs}
    rb = np.random.default_rng(5150); wv = []
    for _ in range(2000):
        pick = rb.choice(len(tabs), len(tabs), replace=True)
        ii = np.concatenate([by[tabs[j]] for j in pick])
        if y[ii].sum() < 2 or y[ii].sum() == len(ii):
            continue
        wv.append(ap(y[ii], P['STEP277  X0+ten members    (22)'][ii])
                  - ap(y[ii], P['STEP272  X0+KEY+MLP2       (14)'][ii]))
    wv = np.array(wv)
    boot[D] = {'P_gt0': float((wv > 0).mean()), 'mean': float(wv.mean()),
               'q05': float(np.quantile(wv, .05)), 'q95': float(np.quantile(wv, .95))}
    print('   def %s  P(delta>0) %.3f  mean %+.5f  [%+.5f, %+.5f]'
          % (D, boot[D]['P_gt0'], boot[D]['mean'], boot[D]['q05'], boot[D]['q95']), flush=True)
rec['bootstrap_277_vs_272'] = boot
rec['note'] = ('internal numbers are recorded, NOT used as a gate: X1 showed the internal instrument '
               'reproduces only 1 of 3 known public signs')
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'x2_unpool.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
