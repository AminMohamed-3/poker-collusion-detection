"""B3: the STEP272 block key M3xKM = M3x (12 inputs) + FINAL2 tail key (13th) + ARCH2_MLP2 (14th).

Same gates as B1: (G1) the 12-input fit reproduces receipt_step156a M3x_fit; (G2) its evaluation order
reproduces block_key_M3x.parquet; (G3) my per-draw rebuild of the evaluation FINAL2 columns reproduces
k_FINAL2 at 32 draws.  Fit rows = ALL def-A rows, exactly as production: g1_fitpop showed that restricting
the FIT to eligible rows is worse, so eligibility governs MEASUREMENT only.
"""
import os, sys, json, time
from pathlib import Path
os.environ['POLARS_MAX_THREADS'] = '4'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C = str(ROOT / 'Claude Version').replace('\\', '/') + '/'
SC = str(HERE / 'recovered_step272').replace('\\', '/') + '/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs.step64f_M3_candidate import final_fit, FS_M3
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step60_multiview_blend import logreg, predict, ap
from pu_pairs.step62_multiview_candidate import VD, eval_frame, DEFS, L2
from pu_pairs.step156a_C_block_M3x import sig, lg

TK = C + 'experiments/tail_keys_20260914/'
F10 = ['SXN3_SHORT3', 'SXN3_ALL5', 'SXN3_LIN30', 'SXN3_LIN300', 'SXN3_FAMMAX',
       'ARCH_XGB3', 'ARCH_ET', 'ARCH_MLP', 'ARCH2_MLP2', 'ARCH2_HGB5']
rec = {}
t0 = time.time()

# ---------- G1: the stored 12-input fit ----------
dx = pl.read_parquet(C + 'experiments/step258_mech_block_view_20260918/testbed32.parquet')
mx, infox = final_fit(dx, FS_M3)
stored = json.load(open(C + 'outputs_step156_20260915/receipt_step156a.json', encoding='utf-8'))['M3x_fit']
G1 = (all(abs(infox['weights_standardised'][v] - stored['weights_standardised'][v]) < 1e-12 for v in FS_M3)
      and infox['bias'] == stored['bias'] and infox['mu'] == stored['mu']
      and infox['sd'] == stored['sd'] and infox['AP_in_sample'] == stored['AP_in_sample'])
rec['G1_fit_reproduces_receipt'] = bool(G1)
print('G1', G1, flush=True)
assert G1

# ---------- G2: the evaluation 12-input score ----------
ev = eval_frame().with_columns(pl.col('R1').cast(pl.Int8).alias('R1i'))
ev = ev.join(pl.read_parquet(VD / 'tail_views_eval_frame.parquet'), on='pair_id', validate='1:1')
assert ev.height == 909
evv = pl.read_parquet(TK + 'eval_views_SX32.parquet', columns=['pair_id', 'L_SHORT3', 'L_ALL5'])
evx = ev.join(evv, on='pair_id', how='left', validate='1:1', maintain_order='left')
assert evx['L_SHORT3'].null_count() == 0
evx = evx.with_columns(pl.Series('r_SHORT3', sig(evx['L_SHORT3'])), pl.Series('r_ALL5', sig(evx['L_ALL5'])))
sx = predict(mx, X_of(evx, FS_M3, 'risk_v13', 'R1i', 'risk_noT'))
ref = pl.read_parquet(C + 'outputs_step156_20260915/block_key_M3x.parquet')
chk = evx.select('pair_id').with_columns(pl.Series('mine', sx)).join(ref, on='pair_id', how='left', validate='1:1')
G2 = float(np.abs(chk['mine'].to_numpy() - chk['score_M3x'].to_numpy()).max())
rec['G2_eval_M3x_max_abs_diff'] = G2
print('G2 max abs diff', G2, flush=True)
assert G2 < 1e-12


# ---------- G3: the evaluation type columns rebuilt from the per-draw files ----------
def build_key(kmax):
    S = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']
    a = pl.concat([pl.read_parquet(C + 'experiments/tail_keys_20260915_neg3/sxn3_eval_k%s.parquet' % s)
                   for s in ('0-7', '8-15', '16-23', '24-31')]).filter(pl.col('k') < kmax)
    A = (a.with_columns([pl.Series('L_' + c, lg(a['r_' + c])) for c in S])
         .group_by('pair_id').agg([pl.col('L_' + c).mean().alias('SXN3_' + c) for c in S]))
    b = pl.concat([pl.read_parquet(C + 'experiments/arch_views_20260915/arch_eval_neg1_k%s.parquet' % s)
                   for s in ('0-15', '16-31')]).filter(pl.col('k') < kmax)
    B = (b.with_columns([pl.Series('L_' + c, lg(b['r_' + c])) for c in ['XGB3', 'ET', 'MLP']])
         .group_by('pair_id').agg([pl.col('L_' + c).mean().alias('ARCH_' + c) for c in ['XGB3', 'ET', 'MLP']]))
    c2 = pl.concat([pl.read_parquet(C + 'experiments/arch2_views_20260915/arch2_eval_k%s.parquet' % s)
                    for s in ('0-15', '16-31')]).filter(pl.col('k') < kmax)
    C2 = (c2.with_columns([pl.Series('L_' + c, lg(c2['r_' + c])) for c in ['MLP2', 'HGB5']])
          .group_by('pair_id').agg([pl.col('L_' + c).mean().alias('ARCH2_' + c) for c in ['MLP2', 'HGB5']]))
    j = A.join(B, on='pair_id', validate='1:1').join(C2, on='pair_id', validate='1:1')
    return j.with_columns(pl.mean_horizontal(F10).alias('KEY'))


k32 = build_key(32)
k16 = build_key(16)
ref32 = pl.read_parquet(C + 'outputs_step163_20260915/eval_views_FINAL2_32.parquet', columns=['pair_id', 'k_FINAL2'])
mm = k32.join(ref32, on='pair_id', validate='1:1')
G3 = float(np.abs(mm['KEY'].to_numpy() - mm['k_FINAL2'].to_numpy()).max())
rec['G3_eval_key32_reproduces_k_FINAL2'] = G3
print('G3 max abs diff', G3, flush=True)
assert G3 < 1e-9

# ---------- the 14-input fit on the development block ----------
V = pl.read_parquet(SC + 'V_arch3.parquet').filter(pl.col('surface') == 'prefix60').drop('surface')
dxk = dx.join(V, on=['pair_id', 'fold'], how='left')
assert dxk['ARCH2_MLP2'].null_count() == 0
devkey = dxk.select(F10).to_numpy().mean(1)
devm2 = dxk['ARCH2_MLP2'].to_numpy()

posset, negset = DEFS['A']
sel = dxk['cls'].is_in(posset + negset).to_numpy()
c = dxk.filter(pl.Series(sel))
y = c['cls'].is_in(posset).to_numpy().astype(float)
Xc = np.column_stack([X_of(c, FS_M3, 'risk_prefix', 'R1_prefix', 'risk_noT'), devkey[sel], devm2[sel]])
mk = logreg(Xc, y, L2)
NAMES = FS_M3 + ['KEY', 'MLP2']
rec['new_fit'] = {'features': NAMES, 'l2': L2, 'n_rows': int(len(y)), 'n_pos': int(y.sum()),
                  'weights_standardised': {v: round(float(w), 4) for v, w in zip(NAMES, mk[0])},
                  'bias': round(float(mk[1]), 4), 'AP_in_sample': round(ap(y, predict(mk, Xc)), 5),
                  'AP_in_sample_M3x': stored['AP_in_sample']}
print('new fit weights', rec['new_fit']['weights_standardised'], flush=True)
print('in-sample AP: M3xKM %.5f vs M3x %s' % (rec['new_fit']['AP_in_sample'], stored['AP_in_sample']), flush=True)

# ---------- the evaluation block score ----------
for tag, kk in (('16', k16), ('32', k32)):
    e2 = evx.join(kk.select('pair_id', 'KEY', 'ARCH2_MLP2'), on='pair_id', how='left',
                  validate='1:1', maintain_order='left')
    assert e2['KEY'].null_count() == 0 and e2['ARCH2_MLP2'].null_count() == 0
    sxk = predict(mk, np.column_stack([X_of(e2, FS_M3, 'risk_v13', 'R1i', 'risk_noT'),
                                       e2['KEY'].to_numpy(), e2['ARCH2_MLP2'].to_numpy()]))
    o_old = np.argsort(-sx, kind='stable')
    o_new = np.argsort(-sxk, kind='stable')
    r_old = np.empty(909, int); r_old[o_old] = np.arange(909)
    r_new = np.empty(909, int); r_new[o_new] = np.arange(909)
    mv = r_new - r_old
    rec['eval_' + tag] = {'spearman_vs_M3x': float(np.corrcoef(r_old, r_new)[0, 1]),
                          'mean_abs_rank_move': float(np.abs(mv).mean()),
                          'max_abs_move': int(np.abs(mv).max()),
                          'moved_gt50': int((np.abs(mv) > 50).sum())}
    r = rec['eval_' + tag]
    print('eval %s: spearman %.4f mean|move| %.1f max %d moved>50 %d'
          % (tag, r['spearman_vs_M3x'], r['mean_abs_rank_move'], r['max_abs_move'], r['moved_gt50']), flush=True)
    pl.DataFrame({'pair_id': e2['pair_id'], 'score_M3xKM': sxk,
                  'score_M3x': sx}).write_parquet(SC + 'block_key_M3xKM_' + tag + '.parquet')

a16 = pl.read_parquet(SC + 'block_key_M3xKM_16.parquet')
a32 = pl.read_parquet(SC + 'block_key_M3xKM_32.parquet')
r1 = np.argsort(np.argsort(-a16['score_M3xKM'].to_numpy()))
r2 = np.argsort(np.argsort(-a32['score_M3xKM'].to_numpy()))
rec['key16_vs_key32'] = {'spearman': float(np.corrcoef(r1, r2)[0, 1]), 'mean_abs_move': float(np.abs(r1 - r2).mean())}
print('16 vs 32 draw order: spearman %.5f mean|move| %.2f'
      % (rec['key16_vs_key32']['spearman'], rec['key16_vs_key32']['mean_abs_move']), flush=True)

p270 = pl.read_parquet(SC + 'block_key_M3xK_16.parquet').rename({'score_M3xK': 's270'})
j = a16.join(p270.select('pair_id', 's270'), on='pair_id', validate='1:1')
ra = np.argsort(np.argsort(-j['score_M3xKM'].to_numpy()))
rb = np.argsort(np.argsort(-j['s270'].to_numpy()))
rec['vs_step270'] = {'spearman': float(np.corrcoef(ra, rb)[0, 1]), 'mean_abs_move': float(np.abs(ra - rb).mean()),
                     'max_abs_move': int(np.abs(ra - rb).max())}
print('vs STEP270 order: spearman %.4f mean|move| %.1f max %d'
      % (rec['vs_step270']['spearman'], rec['vs_step270']['mean_abs_move'], rec['vs_step270']['max_abs_move']), flush=True)
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'b3_key.json', 'w'), indent=1)
print('OK', rec['seconds'], 's', flush=True)
