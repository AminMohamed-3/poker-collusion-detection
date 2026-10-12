"""E6: verify the encoder-lane session's COVERAGE finding (19.233): "5,788 pairs in the current top-16,060
by risk still carry v13 evidence lists".

This sits in my lane (scope/coverage) and, unlike everything else measured tonight, it is not a modelling
idea -- it is a possible STALENESS BUG.  The evidence decoder re-ranks the top 16,060 pairs BY RISK.  Risk has
been re-ordered many times since those lists were produced (STEP169 boundary merge, STEP251, STEP259b,
STEP270, STEP272).  A pair that rose into the top 16,060 AFTER its lists were computed would still be
carrying whatever evidence it had before -- the v13-era lists.

If true this is worth real points, because the current evidence stack beat v13 in PUBLIC, not just on the
audit: STEP167 (LP evidence) read +.00258 and STEP259b (fusion) +.00114.  Those gains were never delivered to
a pair still holding a v13 list.

Measured here, read-only, on the shipped files:
  1. how many of the current top-16,060-by-risk pairs carry evidence IDENTICAL to the v13 file
  2. the same for pairs OUTSIDE the top 16,060, as a control -- they are expected to be ~all v13
  3. where the stale pairs sit in the risk ordering, and their predicted_behavior split
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

CUR = P + 'Claude Version/outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv'
V13 = P + 'outputs_v13_coverage/submission_conditional_coverage.csv'
EV = ['evidence_hand_%d' % i for i in (1, 2, 3, 4, 5)]
SCOPE = 16060
t0 = time.time(); rec = {'scope': SCOPE}

cur = pl.read_csv(CUR, infer_schema=False)
v13 = pl.read_csv(V13, infer_schema=False)
print('current %d rows, v13 %d rows' % (cur.height, v13.height), flush=True)
rec['cur_rows'] = int(cur.height); rec['v13_rows'] = int(v13.height)

j = cur.select(['pair_id', 'risk_score', 'predicted_behavior'] + EV).join(
    v13.select(['pair_id'] + EV), on='pair_id', how='left', suffix='_v13', validate='1:1')
print('joined %d (missing from v13: %d)' % (j.height, int(j['evidence_hand_1_v13'].is_null().sum())), flush=True)

same = np.ones(j.height, dtype=bool)
for c in EV:
    same &= (j[c] == j[c + '_v13']).fill_null(False).to_numpy()
risk = j['risk_score'].cast(pl.Float64).to_numpy()
order = np.argsort(-risk, kind='mergesort')
rank = np.empty(j.height, dtype=np.int64); rank[order] = np.arange(j.height)
inside = rank < SCOPE

print('\nlists IDENTICAL to v13:', flush=True)
print('   inside  top %-6d : %6d of %6d  (%.4f)' % (SCOPE, int(same[inside].sum()), int(inside.sum()),
                                                    same[inside].mean()), flush=True)
print('   outside top %-6d : %6d of %6d  (%.4f)' % (SCOPE, int(same[~inside].sum()), int((~inside).sum()),
                                                    same[~inside].mean()), flush=True)
rec['stale_inside'] = int(same[inside].sum())
rec['stale_outside_frac'] = float(same[~inside].mean())
print('\n   >>> encoder session reported 5,788 stale inside the scope; I measure %d -- %s'
      % (int(same[inside].sum()),
         'CONFIRMED' if abs(int(same[inside].sum()) - 5788) <= 50 else 'DIFFERENT, flag it'), flush=True)
rec['confirms_5788'] = bool(abs(int(same[inside].sum()) - 5788) <= 50)

st = same & inside
if st.sum():
    r = rank[st]
    print('\n   stale pairs inside scope: risk-rank min %d  median %d  max %d'
          % (r.min(), int(np.median(r)), r.max()), flush=True)
    print('   how many in the TOP 1000 by risk: %d ; top 100: %d' % (int((r < 1000).sum()), int((r < 100).sum())),
          flush=True)
    rec['stale_rank'] = {'min': int(r.min()), 'median': int(np.median(r)), 'max': int(r.max()),
                         'top1000': int((r < 1000).sum()), 'top100': int((r < 100).sum())}
    fam = j['predicted_behavior'].to_numpy()
    for f in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
        print('      %-24s %5d stale' % (f, int((st & (fam == f)).sum())), flush=True)
        rec['stale_' + f] = int((st & (fam == f)).sum())

# decile profile: is staleness concentrated where risk moved most?
print('\n   staleness by risk decile inside the scope:', flush=True)
for dcl in range(10):
    lo, hi = dcl * SCOPE // 10, (dcl + 1) * SCOPE // 10
    m = inside & (rank >= lo) & (rank < hi)
    print('      ranks %6d-%6d : %.4f stale' % (lo, hi, same[m].mean()), flush=True)
    rec['decile_%d' % dcl] = float(same[m].mean())
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'e6_stale.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
