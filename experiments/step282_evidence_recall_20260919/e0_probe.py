"""E0: load the 162-query audit once and (i) independently confirm the evidence session's three baseline
numbers, (ii) map what varying the POOL SIZE would actually cost.

The evidence session reports, from its own harness load tonight:
    production evidence decoded MAP@5   .733740
    production evidence raw MAP@5       .686353
    stage-1 TOP-20 recall               .9710
and asked for a second pair of eyes, because its 8:1 ranking-vs-recall split -- and therefore my whole lane --
rests on that recall figure.

Production takes the top 20 candidates per pair by the stage-1 score.  Top-20 is an inherited CHOICE, not a
constraint: the audit frame holds every scored candidate, so recall@K is computable for any K with no
rescoring at all.  This probe only measures; it changes nothing.
"""
import os, sys, json, time
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'                      # match the evidence session's footprint exactly
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.dont_write_bytecode = True
P = 'C:/Data/Projects/Money Making/work/poker/'
SC = ('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/'
      'e91ed0b3-2ad5-45d4-b30c-115132b295b0/scratchpad/evid/')
sys.path.insert(0, P + 'Claude Version/src')
sys.path.insert(0, P + 'Codex Resume/src')
import numpy as np, polars as pl
import audit_common as ac

t0 = time.time(); rec = {}
d, shared, rcols, saved = ac.load_current()
print('audit frame: %d rows, %d pairs' % (d.height, d['pair_id'].n_unique()), flush=True)
print('saved columns: %s' % saved.columns, flush=True)
print('d columns (non-feature): %s' % [c for c in d.columns if c not in rcols][:40], flush=True)
rec['rows'] = int(d.height); rec['pairs'] = int(d['pair_id'].n_unique())
rec['n_rcols'] = len(rcols); rec['n_shared'] = len(shared)

per = d.group_by('pair_id').len().sort('len')
print('\ncandidates per pair: min %d  q25 %d  median %d  q75 %d  max %d  mean %.1f'
      % (per['len'].min(), per['len'].quantile(.25), per['len'].median(),
         per['len'].quantile(.75), per['len'].max(), per['len'].mean()), flush=True)
rec['per_pair'] = {'min': int(per['len'].min()), 'median': int(per['len'].median()),
                   'max': int(per['len'].max()), 'mean': float(per['len'].mean())}

tr = d.filter(pl.col('relevance') == 1)
print('truth hands: %d over %d pairs (mean %.3f per pair)'
      % (tr.height, tr['pair_id'].n_unique(), tr.height / d['pair_id'].n_unique()), flush=True)
rec['truth_hands'] = int(tr.height)
print('family split of the 162 queries:', flush=True)
fam = d.select('pair_id', 'behavior_family').unique().group_by('behavior_family').len().sort('len', descending=True)
for r in fam.iter_rows():
    print('   %-24s %d' % r, flush=True)
rec['families'] = {r[0]: int(r[1]) for r in fam.iter_rows()}

# ---- recall@K on the stage-1 ordering, which is exactly what the pool selection uses
print('\nrecall@K on the stage-1 ordering (oof_rank), and the share of pairs whose pool is that big:', flush=True)
rk = d['oof_rank'].to_numpy()
rel = (d['relevance'] == 1).to_numpy()
pid = d['pair_id'].to_numpy()
npairs = d['pair_id'].n_unique()
sizes = per['len'].to_numpy()
rec['recall'] = {}
print('  %5s %10s %12s %14s' % ('K', 'recall@K', 'truth lost', 'pairs >= K'), flush=True)
for K in (5, 10, 20, 25, 30, 40, 60, 80, 120, 10 ** 9):
    got = (rel & (rk <= K)).sum()
    lab = 'ALL' if K > 10 ** 8 else str(K)
    rec['recall'][lab] = float(got / rel.sum())
    print('  %5s %10.4f %12d %14.3f'
          % (lab, got / rel.sum(), int(rel.sum() - got), float((sizes >= min(K, sizes.max())).mean())), flush=True)

print('\n  >>> CONFIRMATION vs the evidence session: top-20 recall = %.4f (they report .9710) -- %s'
      % (rec['recall']['20'], 'MATCH' if abs(rec['recall']['20'] - .9710) < 5e-4 else 'MISMATCH, flag it'), flush=True)
rec['confirms_recall_9710'] = bool(abs(rec['recall']['20'] - .9710) < 5e-4)

# ---- where the unreachable truth sits: is it deep, or just outside 20?
lost = d.filter(rel & (pl.col('oof_rank') > 20))
if lost.height:
    lr = lost['oof_rank'].to_numpy()
    print('\nthe %d truth hands outside the top 20 sit at stage-1 rank: %s%s'
          % (lost.height, sorted(lr.astype(int))[:30], ' ...' if lost.height > 30 else ''), flush=True)
    print('   of them, %d are within rank 40 and %d within rank 60'
          % (int((lr <= 40).sum()), int((lr <= 60).sum())), flush=True)
    rec['lost_ranks'] = {'n': int(lost.height), 'within40': int((lr <= 40).sum()), 'within60': int((lr <= 60).sum()),
                         'median_rank': float(np.median(lr)), 'max_rank': float(lr.max())}
    print('   they belong to families: %s'
          % lost.group_by('behavior_family').len().sort('len', descending=True).to_dicts(), flush=True)
    rec['lost_by_family'] = {r['behavior_family']: int(r['len'])
                             for r in lost.group_by('behavior_family').len().to_dicts()}
    pl_lost = lost['pair_id'].n_unique()
    print('   spread over %d distinct pairs of 162' % pl_lost, flush=True)
    rec['lost_pairs'] = int(pl_lost)

rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'e0_probe.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
