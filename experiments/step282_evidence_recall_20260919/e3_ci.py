"""E3: coordinated_isolation.  My slice, handed over by the evidence session: CI rides UNFUSED
(FUSED_FAMS = directed_transfer, soft_play only), every DT/ALL-scope intervention they ran HURT it, every
SP-scope one left it exactly unchanged, and the harness drops `rcols` for CI in stage 2.

E1 showed CI is also the WORST family on the audit incumbent: CI .55643 against DT .62058 and SP .63998.
43 of the 162 queries are CI and summarize() takes an unweighted mean over pairs, so CI is 26.5 % of the
metric sitting ~.08 below the other two.

Diagnosed here WITHOUT ANY REFIT, so the evidence session's partition trap cannot apply:
  1. per-family recall@20 and per-family ORACLE -- is CI short on recall, or on ranking?
  2. where CI's truth hands actually sit in each available ordering
  3. label-free reorderings of the SAME pool: chronology, reverse chronology, rank-mean of stage-2 with
     chronology.  Their 19.230(b) found CI's generator order is nearly pure chronology (Kendall .9609 for CI
     against .743 DT / .712 SP), and STEP148 already shipped a CI time-order rule, so this asks whether the
     production stage-2 ordering is actually beating time order for CI at all.
Every arm is a re-ordering of frozen scores: no fit, no fold noise, and the comparison is exact.
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

FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
t0 = time.time(); rec = {}
d, shared, rcols, saved = ac.load_current()
pid = d['pair_id'].to_numpy()
rk = d['oof_rank'].to_numpy().astype(float)
rel = (d['relevance'] == 1).to_numpy()
inc = saved['incumbent'].to_numpy().astype(float)
chrono = d['chrono_rank'].to_numpy().astype(float)
cfrac = d['chrono_frac'].to_numpy().astype(float)
key = d.select('pair_id', 'hand_id', 'table_id')
truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
pools = sorted(set(ptab.values()))
famrow = d['behavior_family'].to_numpy()
NEG = -1e18
POOL = rk <= 20


def m5(score):
    s, m = summarize(key.with_columns(pl.Series('score', score)), truths, fams)
    return s, m


def pooled(v):
    out = np.where(POOL, v, NEG)
    return out


# ---------------------------------------------------------------- 1. recall vs ranking, per family
print('per-family diagnosis (pool = production top 20):\n', flush=True)
base_s, base_m = m5(pooled(inc))
orc = pooled(np.where(POOL & rel, 1e17, inc))
orc_s, _ = m5(orc)
print('%-24s %8s %9s %9s %9s %10s' % ('family', 'queries', 'recall20', 'actual', 'oracle', 'rank gap'), flush=True)
diag = {}
for f in FAMS:
    fm = famrow == f
    r20 = float((rel & fm & POOL).sum() / (rel & fm).sum())
    a, o = base_s['by_family'][f], orc_s['by_family'][f]
    diag[f] = {'queries': int(len(set(pid[fm]))), 'recall20': r20, 'actual': a, 'oracle': o, 'gap': o - a}
    print('%-24s %8d %9.4f %9.5f %9.5f %10.5f'
          % (f, diag[f]['queries'], r20, a, o, o - a), flush=True)
rec['per_family'] = diag

# ---------------------------------------------------------------- 2. where CI truth sits
ci = famrow == 'coordinated_isolation'
print('\nCI truth hands inside the pool: stage-1 rank and incumbent rank', flush=True)
sub = d.filter(pl.Series(ci & rel & POOL))
if sub.height:
    print('   n=%d  stage-1 rank: median %.1f  mean %.1f'
          % (sub.height, float(sub['oof_rank'].median()), float(sub['oof_rank'].mean())), flush=True)
    rec['ci_truth_stage1_rank_median'] = float(sub['oof_rank'].median())

# ---------------------------------------------------------------- 3. label-free reorderings, CI only
def rank_within(v, descending=True):
    f = pl.DataFrame({'pair_id': pid, 'v': v})
    e = pl.col('v').rank('average', descending=descending).over('pair_id')
    return f.select(e.alias('r'))['r'].to_numpy().astype(float)


r_inc = rank_within(inc)
r_chr_new = rank_within(chrono)            # latest hands first
r_chr_old = rank_within(-chrono)           # earliest hands first
ARMS = {
    'PROD incumbent': inc,
    'CHRONO newest-first': -r_chr_new,
    'CHRONO oldest-first': -r_chr_old,
    'rankmean(inc, newest)': -(r_inc + r_chr_new) / 2.0,
    'rankmean(inc, oldest)': -(r_inc + r_chr_old) / 2.0,
    'rankmean(inc,inc,newest)': -(2 * r_inc + r_chr_new) / 3.0,
}
print('\nlabel-free reorderings of the SAME top-20 pool, applied to CI ONLY'
      ' (other families keep production):\n', flush=True)
print('%-28s %10s %10s   %9s %9s %9s' % ('arm (CI only)', 'map5', 'd vs prod', 'DT', 'SP', 'CI'), flush=True)
res = {}
maps = {}
for nm, v in ARMS.items():
    mix = np.where(ci, v, inc)             # CI rows use the arm, everything else production
    # keep the two orderings on a common scale by ranking within pair before pooling
    s, m = m5(pooled(rank_within(mix) * -1.0))
    res[nm] = {'map5': s['map5'], 'd': s['map5'] - base_s['map5'], 'by_family': s['by_family']}
    maps[nm] = m
    print('%-28s %10.6f %+10.6f   %9.5f %9.5f %9.5f'
          % (nm, s['map5'], s['map5'] - base_s['map5'], s['by_family']['directed_transfer'],
             s['by_family']['soft_play'], s['by_family']['coordinated_isolation']), flush=True)
rec['ci_arms'] = res
rec['base'] = {'map5': base_s['map5'], 'by_family': base_s['by_family']}

best = max(res, key=lambda z: res[z]['map5'])
if res[best]['d'] > 0:
    print('\npaired bootstrap by table pool, %s vs production:' % best, flush=True)
    c = ac.boot(base_m, maps[best], ptab, pools)
    cf = {f: float(np.mean([maps[best][p] - base_m[p] for p in base_m if fams[p] == f])) for f in FAMS}
    print('   mean_delta %+.6f  ci95 [%+.6f, %+.6f]' % (c['mean_delta'], c['ci95'][0], c['ci95'][1]), flush=True)
    print('   by family %s' % {k: round(v, 5) for k, v in cf.items()}, flush=True)
    ch = int(sum(1 for p in base_m if abs(maps[best][p] - base_m[p]) > 1e-12))
    print('   CI pairs whose list changes: %d of %d' % (ch, diag['coordinated_isolation']['queries']), flush=True)
    rec['best'] = {'arm': best, 'mean_delta': float(c['mean_delta']),
                   'ci95': [float(c['ci95'][0]), float(c['ci95'][1])], 'by_family': cf, 'pairs_changed': ch}
else:
    print('\nno CI reordering beats production; best was %s at %+.6f' % (best, res[best]['d']), flush=True)
    rec['best'] = None
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'e3_ci.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
