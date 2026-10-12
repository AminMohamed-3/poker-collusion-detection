"""STEP277-B: does any single observable CUT THE CANDIDATE UNIVERSE?  The coverage-depth probe.

STEP277-A found the generator's rule is family-specific and named the mechanistically right quantities:
t5_transfer_bb_A2B / t4_flow_bb_A2B reproduce the cited set exactly for 8 pairs, ALL directed_transfer;
t1_ap_call_aggressor_* for 11 pairs, ALL soft_play; coordinated_isolation gets 0 from every column.  Exact-set
rates are low (<= 3%), so no single column IS the rule -- but that is the wrong question to stop on.

The right question is DEPTH.  Production's stage 1 needs the top 20 of ~121 candidates to reach .97 recall.  If
some column covers all five cited hands within its own top 8, the candidate universe collapses by 15x and the
generator's predicate is found -- that would beat production's recall outright, which is the only thing that can
convert into score.  So: for every column, per family, the within-pair rank a hand needs before all cited hands
are included.

  depth_p50 / p90     over pairs, the max within-pair rank among the cited hands (lower is better; 5 is perfect)
  recall@8, recall@20 per hand, directly comparable to production's stage-1 .9710 at 20
  support_gt0         how many hands per pair have the column nonzero -- a nonzero-predicate is the cheapest
                      possible rule, so it is measured explicitly rather than inferred

Baseline carried in every table: production's own stage-1 ordering on the audit needs 20 for .9710.
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
hands = pl.read_parquet(R / 'data/hands.parquet', columns=['hand_id', 'started_at'])
epairs = ev['pair_id'].unique().to_list()
mech = pl.scan_parquet(MP).filter(pl.col('pair_id').is_in(epairs)).collect().join(hands, on='hand_id', how='left')
mech = mech.join(ev.select('pair_id', 'hand_id', 'evidence_rank'), on=['pair_id', 'hand_id'], how='left')
fam = dict(ev.select('pair_id', 'behavior_family').unique().iter_rows())
mech = mech.with_columns(pl.col('pair_id').replace_strict(fam, default=None).alias('family')).sort(['pair_id', 'started_at'])
pid = mech['pair_id'].to_numpy()
cited = np.array([v is not None for v in mech['evidence_rank'].to_list()])
famv = mech['family'].to_numpy()
_, first = np.unique(pid, return_index=True)
GRP = np.split(np.arange(len(pid)), np.sort(first)[1:])
FAMS = sorted(set(famv[g[0]] for g in GRP))
GF = {f: [g for g in GRP if famv[g[0]] == f] for f in FAMS}
print('pairs %d   %s' % (len(GRP), {f: len(GF[f]) for f in FAMS}), flush=True)
cols = [c for c in mech.columns if c.startswith(('t1_', 't2_', 't3_', 't4_', 't5_'))]
M = np.nan_to_num(mech.select(cols).to_numpy().astype(float), nan=0., posinf=0., neginf=0.)
print('columns %d   rows %d' % (len(cols), M.shape[0]), flush=True)


def probe(x, groups):
    """within-pair descending rank of cited hands; returns depth per pair and rank per cited hand"""
    dep = []; rks = []; sup = []
    for g in groups:
        v = x[g]; c = cited[g]
        o = np.argsort(-v, kind='stable')
        rr = np.empty(len(g), dtype=int); rr[o] = np.arange(1, len(g) + 1)
        rc = rr[c]
        dep.append(int(rc.max())); rks += list(rc); sup.append(int((v > 0).sum()))
    return np.array(dep), np.array(rks), np.array(sup)


t0 = time.perf_counter(); rows = []
for j, cn in enumerate(cols):
    for sgn in (1.0, -1.0):
        x = sgn * M[:, j]
        rec = {'col': cn, 'sign': '+' if sgn > 0 else '-'}
        ok = True
        for f in FAMS:
            dep, rks, sup = probe(x, GF[f])
            rec[f] = {'depth_p50': int(np.percentile(dep, 50)), 'depth_p90': int(np.percentile(dep, 90)),
                      'rec8': round(float((rks <= 8).mean()), 4), 'rec20': round(float((rks <= 20).mean()), 4),
                      'sup_med': int(np.median(sup))}
        rec['rec20_all'] = round(float(np.mean([rec[f]['rec20'] for f in FAMS])), 4)
        rec['rec8_all'] = round(float(np.mean([rec[f]['rec8'] for f in FAMS])), 4)
        rows.append(rec)
    if j and j % 200 == 0:
        print('   %d/%d (%.0fs)' % (j, len(cols), time.perf_counter() - t0), flush=True)
print('', flush=True)
print('=== best single column PER FAMILY, by recall@20 (production stage-1 = .9710 at 20) ===', flush=True)
best = {}
for f in FAMS:
    rr = sorted(rows, key=lambda r: -r[f]['rec20'])
    best[f] = rr[:6]
    print('', flush=True)
    print('%s   (%d pairs)' % (f.upper(), len(GF[f])), flush=True)
    print('  %-38s %4s %8s %8s %9s %9s %8s' % ('column', 'sgn', 'rec@8', 'rec@20', 'depth_p50', 'depth_p90', 'sup_med'), flush=True)
    for r in rr[:6]:
        d = r[f]
        print('  %-38s %4s %8.4f %8.4f %9d %9d %8d' % (r['col'][:38], r['sign'], d['rec8'], d['rec20'],
                                                       d['depth_p50'], d['depth_p90'], d['sup_med']), flush=True)
print('', flush=True)
print('=== best column OVERALL by mean recall@20 across families ===', flush=True)
ro = sorted(rows, key=lambda r: -r['rec20_all'])[:8]
print('  %-38s %4s %9s %9s' % ('column', 'sgn', 'rec8_all', 'rec20_all'), flush=True)
for r in ro:
    print('  %-38s %4s %9.4f %9.4f' % (r['col'][:38], r['sign'], r['rec8_all'], r['rec20_all']), flush=True)
json.dump({'n_pairs': len(GRP), 'per_family_pairs': {f: len(GF[f]) for f in FAMS},
           'prod_stage1_recall20': 0.9710, 'best_per_family': best, 'best_overall': ro},
          open(OUT / 'gen_depth.json', 'w'), indent=2)
print('', flush=True)
print('wrote gen_depth.json  (%.0fs)' % (time.perf_counter() - t0), flush=True)
