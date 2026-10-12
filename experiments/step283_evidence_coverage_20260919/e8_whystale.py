"""E8: is the staleness a SKIPPED-SCOPE bug, or just AGREEMENT?

E6/E7: 5,855 of the current top-16,060-by-risk pairs carry v13-identical evidence, monotone in risk rank
(0.8 % at the top of the scope, 67.3 % at the bottom), worth ~5.8 expected true targets = +.0002 to +.0009
composite with NO audit-transfer discount.

Two explanations, and they imply completely different actions:
  (A) SKIPPED SCOPE.  The lists were computed when the scope was the top 16,060 under an OLDER risk ordering.
      Pairs that have since RISEN into the scope were never scored, so they still hold v13 lists.
      -> a real bug, and re-scoring exactly those pairs is a clean, buildable gain.
  (B) AGREEMENT.  The current decoder was applied but produced the same top-5 as v13 for these pairs.
      Low-risk pairs have weaker, shorter candidate sets, so agreement rising toward the bottom of the scope
      would be expected and there is nothing to fix.
The monotone decile profile is consistent with BOTH, so it does not discriminate.

Decisive test: STEP167 built the current evidence on the STEP169 file, so its scope was the top 16,060 under
STEP169's risk.  Under (A) the stale set should be almost exactly {pairs outside STEP169's top 16,060} INTER
{pairs inside the current top 16,060}.  Under (B) there is no reason for that to line up.
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
OLD = P + 'Claude Version/outputs_step169_20260915/submission_step169.csv'
EV = ['evidence_hand_%d' % i for i in (1, 2, 3, 4, 5)]
SCOPE = 16060
t0 = time.time(); rec = {}


def ranks_of(path):
    f = pl.read_csv(path, infer_schema=False).select(['pair_id', 'risk_score'])
    r = f['risk_score'].cast(pl.Float64).to_numpy()
    o = np.argsort(-r, kind='mergesort')
    rk = np.empty(len(r), np.int64); rk[o] = np.arange(len(r))
    return dict(zip(f['pair_id'].to_list(), rk.tolist()))


cur = pl.read_csv(CUR, infer_schema=False)
v13 = pl.read_csv(V13, infer_schema=False)
j = cur.select(['pair_id', 'risk_score'] + EV).join(v13.select(['pair_id'] + EV), on='pair_id',
                                                    how='left', suffix='_v13', validate='1:1')
same = np.ones(j.height, bool)
for c in EV:
    same &= (j[c] == j[c + '_v13']).fill_null(False).to_numpy()
r = j['risk_score'].cast(pl.Float64).to_numpy()
o = np.argsort(-r, kind='mergesort'); rank = np.empty(j.height, np.int64); rank[o] = np.arange(j.height)
inside = rank < SCOPE
stale = same & inside
pids = j['pair_id'].to_list()
print('current scope %d, stale inside %d' % (int(inside.sum()), int(stale.sum())), flush=True)

oldr = ranks_of(OLD)
orank = np.array([oldr.get(p, 10 ** 9) for p in pids])
old_inside = orank < SCOPE
print('STEP169 scope recovered for %d of %d pairs' % (int((orank < 10 ** 9).sum()), j.height), flush=True)

newly = inside & ~old_inside          # risen into scope since STEP169
print('\npairs that RISisEN into the scope since STEP169: %d' % int(newly.sum()), flush=True)
rec['newly_in_scope'] = int(newly.sum())
rec['stale'] = int(stale.sum())

a = int((stale & newly).sum())
b = int((stale & ~newly).sum())
c_ = int((~stale & newly).sum())
print('\n            | stale      | fresh', flush=True)
print('  risen     | %6d     | %6d' % (a, c_), flush=True)
print('  always in | %6d     | %6d' % (b, int((~stale & ~newly & inside).sum())), flush=True)
rec['crosstab'] = {'risen_stale': a, 'risen_fresh': c_, 'always_stale': b,
                   'always_fresh': int((~stale & ~newly & inside).sum())}
if newly.sum():
    print('\n  P(stale | risen into scope)      = %.4f' % (a / newly.sum()), flush=True)
    rec['P_stale_given_risen'] = float(a / newly.sum())
if (inside & ~newly).sum():
    print('  P(stale | in scope all along)    = %.4f' % (b / (inside & ~newly).sum()), flush=True)
    rec['P_stale_given_always'] = float(b / (inside & ~newly).sum())

print('\n  >>> VERDICT: %s' % (
    'SKIPPED SCOPE (A) -- staleness tracks having risen into scope'
    if rec.get('P_stale_given_risen', 0) > 3 * max(rec.get('P_stale_given_always', 1e-9), 1e-9)
    else 'NOT explained by scope movement -- agreement (B) is the better explanation, nothing to fix'), flush=True)
rec['verdict_skipped_scope'] = bool(rec.get('P_stale_given_risen', 0) >
                                    3 * max(rec.get('P_stale_given_always', 1e-9), 1e-9))

# control: outside the current scope entirely, what fraction match v13?
print('\n  control -- outside the current scope: %.4f match v13 (expected ~1 if never re-scored)'
      % same[~inside].mean(), flush=True)
rec['outside_match'] = float(same[~inside].mean())
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'e8_whystale.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
