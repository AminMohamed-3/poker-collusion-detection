"""STEP277-A: recover the GENERATOR's rule for choosing the five cited hands.

New objective, new substrate.  Not a model of citation -- the RULE.  If the generator computes a per-hand
quantity and takes the top five, then some observable reproduces the cited SET exactly, and its order reproduces
`evidence_rank`.  That is a different question from everything in 19.234, and it is the only one whose answer is
worth the full ORACLE_TRUTH +.053, because a recovered rule returns the truth rather than a fraction of it.

Substrate: all 372 labelled development pairs (2.3x the 162-query audit), their FULL candidate universe from the
STEP254 mechanical tensor (~120 hands per pair, 662 frozen per-hand columns), and real chronology from
hands.parquet `started_at`.  Nothing here is fitted, so there is no fold noise and no selection to pay for until
a column is chosen -- and choosing one is exactly what the exact-match criteria below are for: an exact set match
on 372 pairs cannot be a coincidence, and a near-match is not evidence of anything.

Probes, cheapest and sharpest first:
  A  is `evidence_rank` simply CHRONOLOGICAL order among the cited hands?  If yes the rank carries no extra
     information and the whole rule lives in the predicate that picks the set.
  B  where do the cited hands sit in the pair's own chronology?  (are they the first five?  the last five?)
  C  for each of 662 columns x both directions: does top-k by that column REPRODUCE the cited set exactly?
     Reported as exact-set rate over pairs and as recall, per family and overall.
  D  for each column: does its order over the cited hands reproduce `evidence_rank` exactly?
"""
import os, sys, json, time
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '2'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np, polars as pl
R = Path('C:/Data/Projects/Money Making/work/poker'); S = Path(os.environ['SCR']); CV = R / 'Claude Version'
OUT = CV / 'experiments/step277_generator_rule_20260919'; OUT.mkdir(parents=True, exist_ok=True)
MP = CV / 'experiments/step254_mechanical_tensor_20260917/step254_features.parquet'

ev = pl.read_csv(R / 'data/development_evidence.csv')
hands = pl.read_parquet(R / 'data/hands.parquet', columns=['hand_id', 'table_id', 'started_at', 'phase'])
print('evidence: %d rows, %d pairs' % (ev.height, ev['pair_id'].n_unique()), flush=True)
print('hand phases: %s' % hands.group_by('phase').len().sort('phase').to_dicts(), flush=True)
epairs = ev['pair_id'].unique().to_list()
mech = pl.scan_parquet(MP).filter(pl.col('pair_id').is_in(epairs)).collect()
print('candidate universe for those pairs: %d rows, %d pairs  (mean %.1f hands/pair)' % (
    mech.height, mech['pair_id'].n_unique(), mech.height / max(mech['pair_id'].n_unique(), 1)), flush=True)
miss = set(epairs) - set(mech['pair_id'].to_list())
if miss:
    print('WARNING: %d evidence pairs absent from the tensor -- dropped' % len(miss), flush=True)
mech = mech.join(hands, on='hand_id', how='left')
# are cited hands actually inside the candidate universe?
key2 = set(zip(mech['pair_id'].to_list(), mech['hand_id'].to_list()))
inuni = sum(1 for p, h in zip(ev['pair_id'], ev['hand_id']) if (p, h) in key2)
print('cited (pair,hand) rows present in the tensor: %d of %d' % (inuni, ev.height), flush=True)
mech = mech.join(ev.select('pair_id', 'hand_id', 'evidence_rank'), on=['pair_id', 'hand_id'], how='left')
fam = dict(ev.select('pair_id', 'behavior_family').unique().iter_rows())
mech = mech.with_columns(pl.col('pair_id').replace_strict(fam, default=None).alias('family')).sort(['pair_id', 'started_at'])
pid = mech['pair_id'].to_numpy(); rk = mech['evidence_rank'].to_numpy()
cited = ~np.isnan(rk.astype(float)) if rk.dtype.kind == 'f' else (rk != None)
cited = np.array([x is not None and not (isinstance(x, float) and np.isnan(x)) for x in mech['evidence_rank'].to_list()])
famv = mech['family'].to_numpy(); sa = mech['started_at'].to_numpy()
uq, first = np.unique(pid, return_index=True)
order = np.argsort(first); uq = uq[order]; first = first[order]
GRP = np.split(np.arange(len(pid)), np.sort(first)[1:])
GRP = [g for g in GRP if cited[g].sum() >= 3]
print('usable pairs (>= 3 cited hands inside the universe): %d' % len(GRP), flush=True)
FAMS = sorted(set(famv[g[0]] for g in GRP))
print('', flush=True)

# ---- A. is evidence_rank chronological order among the cited hands?
exact = 0; spear = []
for g in GRP:
    c = g[cited[g]]
    r = np.array([mech['evidence_rank'][int(i)] for i in c], dtype=float)
    t = sa[c].astype('datetime64[ns]').astype('int64') if sa.dtype.kind == 'M' else np.asarray(sa[c], dtype='int64')
    o = np.argsort(t, kind='stable')
    exact += int(np.array_equal(r[o], np.sort(r)))
    if len(r) > 1:
        a = np.argsort(np.argsort(t, kind='stable')); b = np.argsort(np.argsort(r, kind='stable'))
        spear.append(np.corrcoef(a, b)[0, 1])
print('A. evidence_rank == chronological order among the cited hands: %d of %d pairs (%.1f%%)' % (
    exact, len(GRP), 100 * exact / len(GRP)), flush=True)
print('   mean within-pair Spearman(rank, started_at) = %+.4f' % float(np.mean(spear)), flush=True)

# ---- B. where do the cited hands sit in the pair's own chronology?
pcts = []; firstk = 0; lastk = 0
for g in GRP:
    n = len(g); c = cited[g]
    t = sa[g].astype('datetime64[ns]').astype('int64') if sa.dtype.kind == 'M' else np.asarray(sa[g], dtype='int64')
    rr = np.argsort(np.argsort(t, kind='stable')) / max(n - 1, 1)
    pcts += list(rr[c]); k = int(c.sum())
    firstk += int(rr[c].max() <= (k - 1) / max(n - 1, 1) + 1e-12)
    lastk += int(rr[c].min() >= (n - k) / max(n - 1, 1) - 1e-12)
pcts = np.array(pcts)
print('', flush=True)
print('B. chronological percentile of cited hands within the pair: mean %.4f  p10 %.3f  p50 %.3f  p90 %.3f' % (
    pcts.mean(), *np.percentile(pcts, [10, 50, 90])), flush=True)
print('   pairs whose cited hands are EXACTLY the chronologically first k: %d   the last k: %d   (of %d)' % (
    firstk, lastk, len(GRP)), flush=True)

# ---- C/D. every mechanical column as a candidate generator score
cols = [c for c in mech.columns if c.startswith(('t1_', 't2_', 't3_', 't4_', 't5_'))]
print('', flush=True)
print('C. sweeping %d mechanical columns x 2 directions for an EXACT set reproduction' % len(cols), flush=True)
M = mech.select(cols).to_numpy().astype(float)
M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
RKF = np.array([np.nan if v is None else float(v) for v in mech['evidence_rank'].to_list()])
res = []
t0 = time.perf_counter()
for j, cn in enumerate(cols):
    x = M[:, j]
    for sgn in (1.0, -1.0):
        exact_set = 0; rec = 0.; den = 0.; exact_ord = 0; nord = 0
        perfam = {f: [0, 0] for f in FAMS}
        for g in GRP:
            c = cited[g]; k = int(c.sum()); v = sgn * x[g]
            topk = np.argsort(-v, kind='stable')[:k]
            hit = int(c[topk].sum())
            rec += hit; den += k
            ok = int(hit == k); exact_set += ok
            f = famv[g[0]]; perfam[f][0] += ok; perfam[f][1] += 1
            ci = g[c]; rr = RKF[ci]
            if len(rr) > 1:
                oo = np.argsort(-sgn * x[ci], kind='stable')
                exact_ord += int(np.array_equal(rr[oo], np.sort(rr))); nord += 1
        res.append({'col': cn, 'sign': '+' if sgn > 0 else '-', 'exact_set': exact_set,
                    'exact_set_rate': exact_set / len(GRP), 'recall': rec / den,
                    'exact_order': exact_ord, 'exact_order_rate': exact_ord / max(nord, 1),
                    'perfam': {f: perfam[f][0] for f in FAMS}})
    if j and j % 150 == 0:
        print('   %d/%d  (%.0fs)' % (j, len(cols), time.perf_counter() - t0), flush=True)
res.sort(key=lambda r: (-r['exact_set'], -r['recall']))
print('', flush=True)
print('%-40s %4s %9s %9s %9s   %s' % ('column', 'sgn', 'exactSet', 'rate', 'recall@k', 'per family'), flush=True)
for r in res[:15]:
    print('%-40s %4s %9d %9.4f %9.4f   %s' % (r['col'][:40], r['sign'], r['exact_set'], r['exact_set_rate'],
                                              r['recall'], r['perfam']), flush=True)
byrec = sorted(res, key=lambda r: -r['recall'])
print('', flush=True)
print('best by RECALL@k (a soft rule may still be the right lead):', flush=True)
for r in byrec[:8]:
    print('%-40s %4s   recall %.4f   exactSet %d   exactOrder %d' % (
        r['col'][:40], r['sign'], r['recall'], r['exact_set'], r['exact_order']), flush=True)
byord = sorted(res, key=lambda r: -r['exact_order'])
print('', flush=True)
print('D. best by EXACT reproduction of evidence_rank ORDER over the cited hands:', flush=True)
for r in byord[:8]:
    print('%-40s %4s   exactOrder %d of %d (%.4f)   recall %.4f' % (
        r['col'][:40], r['sign'], r['exact_order'], len(GRP), r['exact_order_rate'], r['recall']), flush=True)
json.dump({'n_pairs': len(GRP), 'n_cols': len(cols), 'families': FAMS,
           'A_rank_is_chrono_pairs': exact, 'A_rank_chrono_spearman': round(float(np.mean(spear)), 4),
           'B_cited_chrono_pct_mean': round(float(pcts.mean()), 4), 'B_exactly_first_k': firstk,
           'B_exactly_last_k': lastk, 'top_by_exact_set': res[:25], 'top_by_recall': byrec[:15],
           'top_by_exact_order': byord[:15]},
          open(OUT / 'gen_rule_sweep.json', 'w'), indent=2)
print('', flush=True)
print('wrote gen_rule_sweep.json  (%.0fs)' % (time.perf_counter() - t0), flush=True)
