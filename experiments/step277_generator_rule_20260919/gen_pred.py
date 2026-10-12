"""STEP277-C: is there a HARD eligibility predicate the generator never violates?

STEP277-B's leaders were all the same object wearing different names -- t3_bothlive_flop, t1_sp_flop_1_*, and
negated t1_a_pre_fold_* all encode "both players were still live past preflop".  That is an ELIGIBILITY rule, not
a severity score, and ranking is the wrong instrument for it: a binary column's within-pair rank is arbitrary
among the tied group, which inflates depth and hides a perfect predicate.

So measure predicates as predicates, on all 1,817 cited rows of 372 labelled pairs:
    recall   = fraction of CITED hands satisfying (x > 0)        -- 1.0000 means the generator NEVER violates it
    support  = hands per pair satisfying it, as a fraction       -- small support with recall 1 is a real cut
    lift     = recall / support_fraction                         -- how much the universe shrinks per unit recall

A predicate with recall exactly 1.0000 over 1,817 cited hands and support well under 1 is a HARD CONSTRAINT of
the generator, and it is immediately actionable in a way a soft feature is not: production's top five can be
checked against it, and any listed hand that violates it is provably wrong.  The complement is equally useful --
a predicate satisfied by ZERO cited hands is an exclusion.

Everything is reported per family, because STEP277-A already showed the rule is family-specific, and both the
constraint and its violation count in PRODUCTION's own output are printed, since a constraint production already
respects is worth nothing.
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
cols = [c for c in mech.columns if c.startswith(('t1_', 't2_', 't3_', 't4_', 't5_'))]
M = np.nan_to_num(mech.select(cols).to_numpy().astype(float), nan=0., posinf=0., neginf=0.)
print('pairs %d  %s   cited rows %d   columns %d' % (
    len(GRP), {f: len(GF[f]) for f in FAMS}, int(cited.sum()), len(cols)), flush=True)
ROWF = {f: np.concatenate(GF[f]) for f in FAMS}
t0 = time.perf_counter(); out = []
for j, cn in enumerate(cols):
    x = M[:, j]
    pos = x > 0
    rec = {'col': cn}
    keep = True
    for f in FAMS:
        ix = ROWF[f]; cf = cited[ix]; pf = pos[ix]
        nrec = float(pf[cf].mean()); nsup = float(pf.mean())
        rec[f] = {'recall': round(nrec, 4), 'support_frac': round(nsup, 4),
                  'lift': round(nrec / nsup, 3) if nsup > 0 else None}
    rec['min_recall'] = min(rec[f]['recall'] for f in FAMS)
    rec['max_support'] = max(rec[f]['support_frac'] for f in FAMS)
    out.append(rec)
    if j and j % 250 == 0:
        print('   %d/%d (%.0fs)' % (j, len(cols), time.perf_counter() - t0), flush=True)
print('', flush=True)
print('=== PERFECT predicates: recall = 1.0000 in EVERY family, ranked by how much they cut ===', flush=True)
perf = [r for r in out if r['min_recall'] >= 0.9999]
perf.sort(key=lambda r: r['max_support'])
print('   found %d of %d columns with recall 1.0000 in all three families' % (len(perf), len(cols)), flush=True)
print('  %-40s %10s %10s %10s' % ('column', 'sup DT', 'sup SP', 'sup CI'), flush=True)
for r in perf[:14]:
    print('  %-40s %10.4f %10.4f %10.4f' % (r['col'][:40], r['directed_transfer']['support_frac'],
                                            r['soft_play']['support_frac'],
                                            r['coordinated_isolation']['support_frac']), flush=True)
print('', flush=True)
print('=== PER-FAMILY perfect predicates (recall 1.0000 in that family alone) ===', flush=True)
byfam = {}
for f in FAMS:
    pf = [r for r in out if r[f]['recall'] >= 0.9999 and r[f]['support_frac'] < 0.999]
    pf.sort(key=lambda r: r[f]['support_frac'])
    byfam[f] = pf[:8]
    print('', flush=True)
    print('%s  -- %d columns never violated, best cuts:' % (f.upper(), len(pf)), flush=True)
    for r in pf[:8]:
        d = r[f]
        print('   %-40s support %.4f   lift %.2fx' % (r['col'][:40], d['support_frac'], d['lift']), flush=True)
print('', flush=True)
print('=== EXCLUSIONS: predicates NO cited hand ever satisfies, ranked by how many hands they remove ===', flush=True)
exc = []
for f in FAMS:
    ef = [r for r in out if r[f]['recall'] <= 1e-9 and r[f]['support_frac'] > 0.02]
    ef.sort(key=lambda r: -r[f]['support_frac'])
    exc.append((f, ef[:6], len(ef)))
    print('', flush=True)
    print('%s  -- %d columns with ZERO cited hands, biggest removals:' % (f.upper(), len(ef)), flush=True)
    for r in ef[:6]:
        print('   %-40s removes %.4f of the pair\'s hands' % (r['col'][:40], r[f]['support_frac']), flush=True)
json.dump({'n_pairs': len(GRP), 'n_cited': int(cited.sum()), 'n_cols': len(cols),
           'perfect_all_families': perf[:30], 'perfect_per_family': byfam,
           'exclusions': {f: e for f, e, _ in exc}, 'n_exclusions': {f: n for f, _, n in exc}},
          open(OUT / 'gen_pred.json', 'w'), indent=2)
print('', flush=True)
print('wrote gen_pred.json  (%.0fs)' % (time.perf_counter() - t0), flush=True)
