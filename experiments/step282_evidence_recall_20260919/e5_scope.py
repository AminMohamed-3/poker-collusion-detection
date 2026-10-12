"""E5: SCOPE.  Only 16,060 of the 112,540 submitted pairs carry an evidence list.  Neither session has
touched this.  A true target OUTSIDE the scope contributes ZERO to the evidence term no matter how good the
decoder is, so scope is a hard multiplier on everything the evidence session is optimising.

Two questions, both cheap:
 (a) STRUCTURE -- in the shipped file, which pairs actually carry evidence, and where do they sit in the risk
     ordering?  If scope is "the top N by risk" then coverage is a pure function of the pair ranking.
 (b) COVERAGE -- what fraction of TRUE targets would such a scope miss?  Estimated on development, where real
     annotator labels exist, at the same scope FRACTION as the evaluation file (16,060 / 112,540 = 14.27 %).

If coverage is ~100 % the lane is closed and the evidence session loses nothing.  If it is materially below
100 %, every point they win inside the decoder is being multiplied by less than one, and widening scope is a
cheaper lever than anything in the decoder.
"""
import os, sys, json, time
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.dont_write_bytecode = True
P = 'C:/Data/Projects/Money Making/work/poker/'
SC = ('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/'
      'e91ed0b3-2ad5-45d4-b30c-115132b295b0/scratchpad/evid/')
import numpy as np, polars as pl

SUB = P + 'Claude Version/outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv'
t0 = time.time(); rec = {}

sub = pl.read_csv(SUB, infer_schema=False)
n = sub.height
h1 = sub['evidence_hand_1']
has = (h1.is_not_null() & (h1 != '') & (h1 != 'null')).to_numpy()
print('submission rows %d ; rows carrying evidence_hand_1: %d (%.4f)' % (n, has.sum(), has.mean()), flush=True)
rec['rows'] = int(n); rec['with_evidence'] = int(has.sum())
for i in (2, 3, 4, 5):
    c = sub['evidence_hand_%d' % i]
    k = (c.is_not_null() & (c != '') & (c != 'null')).to_numpy()
    print('   slot %d filled: %d' % (i, k.sum()), flush=True)
    rec['slot_%d' % i] = int(k.sum())

risk = sub['risk_score'].cast(pl.Float64).to_numpy()
order = np.argsort(-risk, kind='mergesort')
rank = np.empty(n, dtype=np.int64)
rank[order] = np.arange(n)
rk_has = rank[has]
print('\nrisk-rank of the pairs carrying evidence: min %d  max %d  median %d'
      % (rk_has.min(), rk_has.max(), int(np.median(rk_has))), flush=True)
topN = int(has.sum())
inside = int((rank[has] < topN).sum())
print('   is scope exactly "top %d by risk"?  %d of %d shipped lists are inside that prefix (%.4f)'
      % (topN, inside, topN, inside / topN), flush=True)
rec['scope_is_risk_prefix'] = float(inside / topN)
rec['scope_rank_max'] = int(rk_has.max())
fam = sub['predicted_behavior'].to_numpy()
print('   family split of pairs WITH evidence:', flush=True)
for f in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
    print('      %-24s %6d of %6d (%.4f)'
          % (f, int((has & (fam == f)).sum()), int((fam == f).sum()),
             float((has & (fam == f)).sum() / max((fam == f).sum(), 1))), flush=True)
    rec['fam_' + f] = float((has & (fam == f)).sum() / max((fam == f).sum(), 1))

# ---------------- (b) coverage of TRUE targets at the same scope fraction, on development
frac = has.sum() / n
print('\nscope fraction = %.5f' % frac, flush=True)
rec['scope_fraction'] = float(frac)
sc = pl.concat([pl.read_parquet(P + 'Claude Version/experiments/pu_pairs_20260911/fold%d/scores.parquet' % f,
                                columns=['pair_id', 'label', 'behavior_family', 'risk'])
                for f in (1, 2)]).unique(subset=['pair_id'], keep='first')
print('development pairs with a risk score: %d' % sc.height, flush=True)
lab = pl.read_csv(P + 'data/development_labels.csv').select(['pair_id', 'label', 'behavior_family'])
dv = sc.drop('label', 'behavior_family').join(lab, on='pair_id', how='left')
dr = dv['risk'].to_numpy()
dl = (dv['label'].fill_null(-1) == 1).to_numpy()
dfam = dv['behavior_family'].fill_null('none').to_numpy()
m = int(round(frac * dv.height))
o = np.argsort(-dr, kind='mergesort')
drank = np.empty(dv.height, dtype=np.int64); drank[o] = np.arange(dv.height)
print('equivalent development scope: top %d of %d by risk' % (m, dv.height), flush=True)
print('   true targets on development: %d' % int(dl.sum()), flush=True)
cov = float((dl & (drank < m)).sum() / max(dl.sum(), 1))
print('\n   >>> TRUE-TARGET COVERAGE at the production scope fraction: %.4f' % cov, flush=True)
rec['dev_target_coverage'] = cov
rec['dev_targets'] = int(dl.sum())
print('       targets missed by scope: %d of %d' % (int(dl.sum() - (dl & (drank < m)).sum()), int(dl.sum())), flush=True)
for f in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
    sel = dl & (dfam == f)
    if sel.sum():
        c = float((sel & (drank < m)).sum() / sel.sum())
        print('       %-24s coverage %.4f  (%d targets)' % (f, c, int(sel.sum())), flush=True)
        rec['dev_cov_' + f] = c
print('\n   coverage at other scope sizes:', flush=True)
for mult in (.5, 1.0, 1.5, 2.0, 3.0, 5.0):
    mm = int(round(frac * mult * dv.height))
    c = float((dl & (drank < mm)).sum() / max(dl.sum(), 1))
    print('      scope x%-4g (top %6d) coverage %.4f' % (mult, mm, c), flush=True)
    rec['cov_x%g' % mult] = c
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'e5_scope.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
