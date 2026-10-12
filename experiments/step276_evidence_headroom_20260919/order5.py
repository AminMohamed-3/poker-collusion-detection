"""STEP276-AL: how much MAP@5 does production lose to the ORDER of the five it already selected?

MAP@5 with exactly 5 relevant and 5 predicted: if all five are right AP = 1 whatever the order.  If only k of
five are right the order matters a lot -- 4 hits placed first give AP .8000, the same four placed last give
.5433.  So every partially-correct list carries up to ~.26 of AP that depends purely on ordering, and production
orders its five by p*q descending, which is an estimate of P(cited), not of the generator's own rank.

Nobody in this project has measured that ceiling.  This is a pure diagnostic: SET HELD FIXED, order varied.
  ORACLE_ORDER  hits first  -> the ceiling reachable without ever changing which hands are listed
  ANTI_ORDER    hits last   -> the floor, for scale
  BY_P          production's own order (gate: must equal production's decoded MAP@5)
  RANDOM        mean over 200 random orders of the same five, for the midpoint
"""
import os, sys, json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ[k] = '2'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np, polars as pl
R = Path('C:/Data/Projects/Money Making/work/poker'); S = Path(os.environ['SCR']); CV = R/'Claude Version'
OUT = CV/'experiments/step276_evidence_headroom_20260919'
sys.path.insert(0, str(CV/'src')); sys.path.insert(0, str(R/'Codex Resume/src')); sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step104_C_typed_decoder_build import chronology, align, less_than_five
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step5_evaluate import half
K = S165.K; CI = 'coordinated_isolation'
D = S165.load(); d, Q, QIX, dec = D['d'], D['Q'], D['QIX'], D['dec']
pid = d['pair_id'].to_numpy(); rel = d['relevance'].to_numpy().astype(int)
fams = dict(d.select('pair_id','behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id','table_id').unique().iter_rows())
truths = {kk[0]: set(gg['hand_id']) for kk, gg in d.filter(pl.col('relevance')==1).group_by('pair_id')}
qids = sorted(truths); key = d.select('pair_id','hand_id','table_id')
fq = np.array([fams[q] for q in qids]); hv = np.array([half(ptab[q]) for q in qids])
famall = np.array([fams[p] for p in pid]); top = D['top']
v = pl.read_parquet(CV/'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(K)
fr = chronology(d.select(K+['table_id']).with_columns(pl.Series('score', v['S_star'].to_numpy())))
t1all = align(fr, pl.read_parquet(CV/'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
cr = fr['chrono_rank'].to_numpy()
PRI = (~(t1all & (famall != CI))).astype(np.int64)
P = np.load(S/'inc_p_PRODbare.npy'); pf = np.zeros(len(pid)); pf[QIX] = P
GT = {pid[g[0]]: g for g in (np.flatnonzero((pid==p)&top) for p in np.unique(pid))}
# production's decoded score, then its selected five IN ORDER
sc = np.zeros(len(pid))
for q, g in GT.items():
    p = np.clip(pf[g], 0., 1.)
    o = np.lexsort((cr[g], PRI[g])); qq = np.empty(len(g)); qq[o] = less_than_five(p[o])
    sc[g] = 10.0 + p*qq + 1e-9*p
SEL = {q: GT[q][np.argsort(-sc[GT[q]], kind='stable')[:5]] for q in qids}

NREL = {q: len(truths[q]) for q in qids}

def ap5(hits, q):
    h = 0; s = 0.0
    for i, r in enumerate(hits, 1):
        if r: h += 1; s += h/i
    return s/min(5, NREL[q])

_, mprod = summarize(key.with_columns(pl.Series('score', sc)), truths, fams)
byp = np.array([ap5(rel[SEL[q]], q) for q in qids])
prod = np.array([mprod[q] for q in qids])
print('GATE  my AP@5 on production order vs summarize(): max abs diff %.2e   mean %.6f' % (
    np.abs(byp-prod).max(), prod.mean()), flush=True)
nh = np.array([int(rel[SEL[q]].sum()) for q in qids])
nrl = np.array([NREL[q] for q in qids])
sat = np.array([nh[j] >= min(5, nrl[j]) for j in range(len(qids))])
print('   cited-hand counts per pair (n_relevant): %s' % np.bincount(nrl, minlength=6).tolist(), flush=True)
print('   lists already SATURATED (every cited hand is in the five): %d of 162' % int(sat.sum()), flush=True)
print('', flush=True)
print('hits among production\'s five, distribution over 162 lists: %s  (mean %.3f)' % (
    np.bincount(nh, minlength=6).tolist(), nh.mean()), flush=True)
print('   lists where order CANNOT matter (saturated or empty): %d of 162' % int((sat|(nh==0)).sum()), flush=True)
orc = np.array([ap5(sorted(rel[SEL[q]], reverse=True), q) for q in qids])
ant = np.array([ap5(sorted(rel[SEL[q]]), q) for q in qids])
rng = np.random.default_rng(276)
rnd = np.array([np.mean([ap5(rng.permutation(rel[SEL[q]]), q) for _ in range(200)]) for q in qids])
print('', flush=True)
print('%-14s %10s %+11s   %9s %9s   %9s %9s %9s' % ('arm','MAP@5','d vs prod','h0','h1','DT','SP','CI'), flush=True)
res = {}
for nm, a in (('BY_P (prod)', byp), ('ORACLE_ORDER', orc), ('RANDOM_ORDER', rnd), ('ANTI_ORDER', ant)):
    dl = a - byp
    res[nm] = {'map5': round(float(a.mean()),6), 'd': round(float(dl.mean()),6),
               'DT': round(float(dl[fq=='directed_transfer'].mean()),6),
               'SP': round(float(dl[fq=='soft_play'].mean()),6), 'CI': round(float(dl[fq==CI].mean()),6)}
    print('%-14s %10.6f %+11.6f   %+9.5f %+9.5f   %+9.5f %+9.5f %+9.5f' % (
        nm, a.mean(), dl.mean(), dl[hv==0].mean(), dl[hv==1].mean(),
        dl[fq=='directed_transfer'].mean(), dl[fq=='soft_play'].mean(), dl[fq==CI].mean()), flush=True)
# where does production place its misses?
posm = []
for q in qids:
    r = rel[SEL[q]]
    if 0 < r.sum() < min(5, NREL[q]):
        posm += [i+1 for i, x in enumerate(r) if not x]
posm = np.array(posm)
print('', flush=True)
print('positions of production\'s MISSES in partially-correct lists (%d misses over %d lists):' % (
    len(posm), int(((nh>0)&~sat).sum())), flush=True)
print('   slot 1..5 counts %s   -- uniform would be %s' % (
    np.bincount(posm, minlength=6)[1:].tolist(), [round(len(posm)/5,1)]*5), flush=True)
json.dump({'n_lists': 162, 'hits_dist': np.bincount(nh, minlength=6).tolist(), 'arms': res,
           'miss_slot_counts': np.bincount(posm, minlength=6)[1:].tolist(),
           'note': 'set held fixed to production selection; only the order of the five varies'},
          open(OUT/'order5_ceiling.json','w'), indent=2)
print('', flush=True)
print('wrote order5_ceiling.json', flush=True)
