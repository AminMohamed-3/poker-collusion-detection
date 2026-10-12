"""X3: the LABEL-FREE version of the STEP272 fix, plus how big the block could be.

(A) W7 established that KEY's implied member weights ARE the members' standard deviations, because it
    averages UNSTANDARDISED logits: they run .037 to .161 and are close to anti-correlated with usefulness.
    STEP272 fixed this for one member by fitting a weight, and read +.00020.  The label-free fix for ALL
    ten is to standardise before averaging -- ZKEY = mean of z-scored members.  It chooses nothing from the
    labels, so it cannot exploit the def-A circularity at all.  Two arms only, no sweep:
        ZKEY   : KEY replaced by ZKEY
        BOTH   : KEY and ZKEY together (they span the same members, differently weighted)
    plus the production STEP272 reference.

(B) The block is 909 pairs out of 112,540.  Everything below it is ordered by the base file.  If the
    multiview frame covers more than 909 pairs, enlarging the block is a far bigger lever than any
    re-weighting -- so this simply measures how many pairs the frame actually has.
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
from pu_pairs.step60_multiview_blend import ap, lg
from pu_pairs.step62_multiview_candidate import DEFS, eval_frame
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
X0 = X_of(d, H.FS_M3, 'risk_prefix', 'R1_prefix', 'risk_noT')
M = d.select(F10).to_numpy()
KEY = M.mean(1)
ZKEY = ((M - M.mean(0)) / (M.std(0) + 1e-12)).mean(1)
M2 = d['ARCH2_MLP2'].to_numpy()
ELIG = d['sh_full'].to_numpy() >= 57
tab = d['table_id'].to_numpy()
certA = d['cls'].is_in(DEFS['A'][0] + DEFS['A'][1]).to_numpy(); yA = d['cls'].is_in(DEFS['A'][0]).to_numpy().astype(float)
certC = d['cls'].is_in(DEFS['C'][0] + DEFS['C'][1]).to_numpy(); yC = d['cls'].is_in(DEFS['C'][0]).to_numpy().astype(float)
DEFB = (['pos', 'undisclosed'], ['FP', 'undisclosed_R1', 'ambiguous', 'mid', 'neg'])
certB = d['cls'].is_in(DEFB[0] + DEFB[1]).to_numpy(); yB = d['cls'].is_in(DEFB[0]).to_numpy().astype(float)
CE = {'A': (certA, yA), 'C': (certC, yC), 'B': (certB, yB)}

ST = {'STEP272 KEY+MLP2 (production)': np.column_stack([X0, KEY, M2]),
      'ZKEY  (label-free restandardise)': np.column_stack([X0, ZKEY, M2]),
      'BOTH  KEY+ZKEY+MLP2': np.column_stack([X0, KEY, ZKEY, M2])}
P = {k: H.crossfold(v, yA, certA, in1, in2, insh, L2) for k, v in ST.items()}
print('(A) label-free KEY restandardisation, production l2 = %g, eligible rows\n' % L2, flush=True)
print('%-34s %9s %9s %9s' % ('arm', 'defA', 'defC', 'defB'), flush=True)
out = {}
for k, p in P.items():
    r = {D: float(ap(CE[D][1][CE[D][0] & ELIG], p[CE[D][0] & ELIG])) for D in ('A', 'C', 'B')}
    out[k] = r
    print('%-34s %9.5f %9.5f %9.5f' % (k, r['A'], r['C'], r['B']), flush=True)
ref = out['STEP272 KEY+MLP2 (production)']
print()
for k, r in out.items():
    if k.startswith('STEP272'):
        continue
    print('   %-34s dA %+.5f dC %+.5f dB %+.5f'
          % (k, r['A'] - ref['A'], r['C'] - ref['C'], r['B'] - ref['B']), flush=True)
rec['A_zkey'] = out
print('\n   corr(KEY, ZKEY) = %.4f' % np.corrcoef(KEY, ZKEY)[0, 1], flush=True)
rec['corr_key_zkey'] = float(np.corrcoef(KEY, ZKEY)[0, 1])

boot = {}
for k in ('ZKEY  (label-free restandardise)', 'BOTH  KEY+ZKEY+MLP2'):
    boot[k] = {}
    for D in ('A', 'C'):
        cert, y = CE[D]
        s = np.where(cert & ELIG)[0]
        tabs = np.unique(tab[s]); by = {t: s[tab[s] == t] for t in tabs}
        rb = np.random.default_rng(909090); wv = []
        for _ in range(2000):
            pick = rb.choice(len(tabs), len(tabs), replace=True)
            ii = np.concatenate([by[tabs[j]] for j in pick])
            if y[ii].sum() < 2 or y[ii].sum() == len(ii):
                continue
            wv.append(ap(y[ii], P[k][ii]) - ap(y[ii], P['STEP272 KEY+MLP2 (production)'][ii]))
        wv = np.array(wv)
        boot[k][D] = {'P_gt0': float((wv > 0).mean()), 'mean': float(wv.mean())}
    print('   %-34s bootstrap def A P %.3f (%+.5f)  def C P %.3f (%+.5f)'
          % (k, boot[k]['A']['P_gt0'], boot[k]['A']['mean'], boot[k]['C']['P_gt0'], boot[k]['C']['mean']), flush=True)
rec['A_bootstrap'] = boot

print('\n(B) how big is the multiview evaluation frame?', flush=True)
try:
    ef = eval_frame()
    rec['eval_frame_rows'] = int(ef.height)
    print('   eval_frame(): %d rows, %d columns' % (ef.height, len(ef.columns)), flush=True)
    if 'shared_hands' in ef.columns:
        sh = ef['shared_hands'].to_numpy()
        print('   shared_hands: min %d  median %d  max %d' % (sh.min(), int(np.median(sh)), sh.max()), flush=True)
except Exception as e:
    print('   eval_frame() failed: %r' % (e,), flush=True)
    rec['eval_frame_rows'] = None
print('   testbed (development block): %d rows' % d.height, flush=True)
rec['testbed_rows'] = int(d.height)
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'x3_zkey.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
