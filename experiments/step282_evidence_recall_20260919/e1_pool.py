"""E1: POOL SIZE.  Production takes the top 20 candidates per pair by stage-1.  That 20 is an inherited
choice; the audit frame holds every scored candidate (median 110 per pair, min 57), so K can be varied with
NO REFIT and NO RESCORING -- only the selection changes.

That matters for resolvability: the evidence session's trap 1 (an arm that refits cannot be signed on 162
queries, partition spread ~ +/-.004) does not apply to a pure re-selection on frozen scores.  There is no
fold noise here because nothing is fitted.

Confirmed in E0 against their independent load: top-20 recall .9710 EXACT match.  Recall by K:
    20 -> .9710 (23 truth hands lost)   25 -> .9786   30 -> .9861   40 -> .9962 (3 lost)   80 -> 1.0000

Measured here, all on production's own frozen stage-2 column (`incumbent`), never a reproduction:
    RAW      map5 from ranking the pool by `incumbent` and taking the top 5
    ORACLE   the same, but truth hands inside the pool are floated to the top -- the ceiling K buys
    per-family, because CI is 43 of the 162 queries and is my slice
Bootstrap by table pool via audit_common.boot, paired against the K=20 reference.
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
from evidence_aug.step65_structural_evidence import summarize

t0 = time.time(); rec = {}
d, shared, rcols, saved = ac.load_current()
n = d.height
pid = d['pair_id'].to_numpy()
rk = d['oof_rank'].to_numpy().astype(float)
rel = (d['relevance'] == 1).to_numpy()
inc = saved['incumbent'].to_numpy().astype(float)
key = d.select('pair_id', 'hand_id', 'table_id')
truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
pools = sorted(set(ptab.values()))
print('rows %d  pairs %d  truth %d' % (n, len(truths), int(rel.sum())), flush=True)


def m5(score):
    s, m = summarize(key.with_columns(pl.Series('score', score)), truths, fams)
    return s, m


NEG = -1e18


def pooled(K):
    v = inc.copy()
    v[rk > K] = NEG
    return v


# ---------------- what does production's raw .686353 actually correspond to?
s_all, _ = m5(inc)
s_20, m20 = m5(pooled(20))
print('\nidentifying production raw MAP@5 = .686353 (evidence session):', flush=True)
print('   incumbent over ALL candidates : %.6f' % s_all['map5'], flush=True)
print('   incumbent over the TOP 20     : %.6f   <- %s'
      % (s_20['map5'], 'THIS ONE' if abs(s_20['map5'] - .686353) < 5e-5 else ''), flush=True)
rec['raw_all'] = s_all['map5']; rec['raw_top20'] = s_20['map5']
rec['matches_686353'] = bool(abs(s_20['map5'] - .686353) < 5e-5 or abs(s_all['map5'] - .686353) < 5e-5)

# ---------------- the sweep
KS = (10, 15, 20, 25, 30, 40, 60, 80, 120, 10 ** 9)
print('\nPOOL SIZE SWEEP on production stage-2 (`incumbent`), no refit:\n', flush=True)
print('%6s %10s %10s   %9s %9s %9s   %10s'
      % ('K', 'map5', 'd vs K20', 'DT', 'SP', 'CI', 'recall@K'), flush=True)
out = {}
maps = {}
for K in KS:
    v = pooled(K)
    s, m = m5(v)
    maps[K] = m
    lab = 'ALL' if K > 10 ** 8 else str(K)
    r = float((rel & (rk <= K)).sum() / rel.sum())
    out[lab] = {'map5': s['map5'], 'd': s['map5'] - s_20['map5'], 'by_family': s['by_family'], 'recall': r}
    print('%6s %10.6f %+10.6f   %9.5f %9.5f %9.5f   %10.4f'
          % (lab, s['map5'], s['map5'] - s_20['map5'], s['by_family'].get('directed_transfer', float('nan')),
             s['by_family'].get('soft_play', float('nan')),
             s['by_family'].get('coordinated_isolation', float('nan')), r), flush=True)
rec['sweep'] = out

# ---------------- the ceiling each K buys: float in-pool truths to the top
print('\nORACLE ceiling per K (truth inside the pool floated to the top):', flush=True)
orc = {}
for K in (20, 25, 30, 40, 80):
    v = pooled(K)
    v = np.where((rk <= K) & rel, 1e17, v)
    s, _ = m5(v)
    orc[str(K)] = s['map5']
    print('   K=%-4d oracle map5 %.6f   (headroom above the real K=%d: %+.6f)'
          % (K, s['map5'], K, s['map5'] - out[str(K)]['map5']), flush=True)
rec['oracle'] = orc

# ---------------- paired bootstrap of the best K against K=20
best = max([k for k in out if k != '20'], key=lambda z: out[z]['map5'])
print('\npaired bootstrap by table pool, K=%s vs K=20:' % best, flush=True)
bk = 10 ** 9 if best == 'ALL' else int(best)
c = ac.boot(maps[20], maps[bk], ptab, pools)
c['by_family'] = {f: float(np.mean([maps[bk][p] - maps[20][p] for p in maps[20] if fams[p] == f]))
                  for f in ('directed_transfer', 'soft_play', 'coordinated_isolation')}
print('   mean_delta %+.6f  ci95 [%+.6f, %+.6f]' % (c['mean_delta'], c['ci95'][0], c['ci95'][1]), flush=True)
print('   by family: %s' % {k: round(v, 5) for k, v in c['by_family'].items()}, flush=True)
rec['boot_best_vs_20'] = {'K': best, 'mean_delta': float(c['mean_delta']),
                          'ci95': [float(c['ci95'][0]), float(c['ci95'][1])], 'by_family': c['by_family']}
np_ch = int(sum(1 for p in maps[20] if abs(maps[bk][p] - maps[20][p]) > 1e-12))
print('   pairs whose list changes at all: %d of %d' % (np_ch, len(maps[20])), flush=True)
rec['pairs_changed'] = np_ch
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'e1_pool.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
