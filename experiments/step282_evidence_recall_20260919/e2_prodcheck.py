"""E2: check the E1 pool conclusion against the REAL production output, not the audit incumbent.

E1 found that widening the candidate pool from 20 to ALL changes map5 by exactly 0.000000 and moves zero of
162 lists, because stage 2 never promotes a stage-1 rank>20 hand into its own top 5.  But E1 scored the audit
frame's `incumbent` column, whose raw map5 is .609180, while the evidence session reports production raw
.686353 -- so `incumbent` is NOT the current production stack (consistent with their note that the audit REF
decodes .682658 against production's .733740).

The conclusion therefore has to be re-tested on production's ACTUAL chosen hands, which are sitting in the
submission file as evidence_hand_1..5.  For the 162 audit pairs:

    what stage-1 rank does each hand production actually shipped occupy?

If they are all <= 20 the pool is non-binding for production too and the axis is closed for real.  If some
exceed 20, production is already reaching past the top 20 and E1's conclusion is an artefact of the incumbent.
Read-only: nothing is written except this receipt.
"""
import os, sys, json, time
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.dont_write_bytecode = True
P = 'C:/Data/Projects/Money Making/work/poker/'
SC = ('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/'
      'e91ed0b3-2ad5-45d4-b30c-115132b295b0/scratchpad/evid/')
sys.path.insert(0, P + 'Claude Version/src')
sys.path.insert(0, P + 'Codex Resume/src')
import numpy as np, polars as pl
import audit_common as ac

SUB = P + 'Claude Version/outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv'
t0 = time.time(); rec = {}
d, shared, rcols, saved = ac.load_current()
aud = set(d['pair_id'].unique().to_list())
print('audit pairs: %d' % len(aud), flush=True)

sub = pl.read_csv(SUB, infer_schema=False)
ev = sub.filter(pl.col('pair_id').is_in(list(aud)))
print('audit pairs present in the submission: %d' % ev.height, flush=True)
rec['audit_pairs_in_submission'] = int(ev.height)

long = ev.select(['pair_id'] + ['evidence_hand_%d' % i for i in (1, 2, 3, 4, 5)]).unpivot(
    index='pair_id', variable_name='slot', value_name='hand_id')
long = long.with_columns(pl.col('slot').str.replace('evidence_hand_', '').cast(pl.Int64))
print('production evidence rows for audit pairs: %d' % long.height, flush=True)

ranks = d.select('pair_id', 'hand_id', 'oof_rank', 'relevance', 'behavior_family')
j = long.join(ranks, on=['pair_id', 'hand_id'], how='left')
miss = int(j['oof_rank'].is_null().sum())
print('shipped hands not found in the audit candidate frame at all: %d' % miss, flush=True)
rec['shipped_not_in_frame'] = miss

r = j.filter(pl.col('oof_rank').is_not_null())['oof_rank'].to_numpy()
if len(r):
    print('\nstage-1 rank of the hands production ACTUALLY SHIPPED (%d hands):' % len(r), flush=True)
    for K in (5, 10, 20, 25, 30, 40, 60, 10 ** 9):
        lab = 'ALL' if K > 10 ** 8 else str(K)
        print('   within stage-1 top %-4s : %5d of %d  (%.4f)'
              % (lab, int((r <= K).sum()), len(r), (r <= K).mean()), flush=True)
    rec['shipped_rank'] = {'n': int(len(r)), 'max': float(r.max()), 'median': float(np.median(r)),
                           'within20': float((r <= 20).mean()), 'within40': float((r <= 40).mean())}
    over = j.filter(pl.col('oof_rank') > 20)
    print('\n   hands shipped from BEYOND stage-1 rank 20: %d' % over.height, flush=True)
    rec['shipped_beyond_20'] = int(over.height)
    if over.height:
        print('   their ranks: %s' % sorted(over['oof_rank'].to_numpy().astype(int).tolist())[:40], flush=True)
        print('   how many of those are TRUE: %d' % int((over['relevance'] == 1).sum()), flush=True)
        print('   by family: %s' % over.group_by('behavior_family').len().to_dicts(), flush=True)
        rec['shipped_beyond_20_true'] = int((over['relevance'] == 1).sum())
    print('\n   >>> VERDICT: %s' % ('pool is NON-BINDING for production too -- E1 stands, axis CLOSED'
                                    if over.height == 0 else
                                    'production DOES reach past rank 20 -- E1 was an incumbent artefact'), flush=True)
    rec['verdict_pool_nonbinding'] = bool(over.height == 0)

# how good are production's shipped lists on the audit truth, for reference
tj = j.with_columns(pl.col('relevance').fill_null(0))
hit = tj.group_by('pair_id').agg(pl.col('relevance').sum().alias('hits'))
print('\nproduction shipped-list hits among the 5 slots: mean %.4f of 5' % hit['hits'].mean(), flush=True)
rec['mean_hits_of_5'] = float(hit['hits'].mean())
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'e2_prodcheck.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
