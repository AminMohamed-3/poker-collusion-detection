"""STEP277-G: is the SOFT-PLAY gain from the out-of-sample induced rule real, paid for honestly?

STEP277-F trained a per-family citation rule on the 210 labelled pairs OUTSIDE the audit and blended it into
production's p as p^(1-w) * rule^w.  Alone it loses (AUC .9057 vs .9334, decoded -.0785).  Blended, within-top-20
AUC rises monotonically to .9410 at w = .30 and decoded is negative overall -- but SOFT PLAY is positive at every
single weight, rising to +.02808 at w = .30, while DT and CI go negative.  SP is the weakest family in the whole
stack (.685337) and it is the one place an out-of-sample rule helps.

Why this is not automatically the usual trap.  The RULE is fitted on pairs disjoint from the audit, so the thing
being measured is not a refit of the evaluation set.  What still has to be paid for is the choice of (scope,
weight) -- 28 cells -- and today's lesson says a smooth monotone curve is exactly what one lucky draw looks like,
so the curve's smoothness earns nothing here.

  HONEST      pick (scope, w) on half the audit's table pools, score on the other half, 400 times, w = 0 in the
              grid as the control
  PRESPEC     SP at w = .30 evaluated on its own: both halves, per-pool bootstrap, and the count of lists it
              actually changes -- because a gain resting on two lists is not a gain (the gamma_CI lesson)
  FAMSPLIT    the three per-family winners applied together, to see whether they add or fight

The rule is cached to disk so this and any follow-up read the identical fitted object.
"""
import os, sys, json, time
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '2'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
R = Path('C:/Data/Projects/Money Making/work/poker'); S = Path(os.environ['SCR']); CV = R / 'Claude Version'
OUT = CV / 'experiments/step277_generator_rule_20260919'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CV / 'src')); sys.path.insert(0, str(R / 'Codex Resume/src')); sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step104_C_typed_decoder_build import chronology, align, less_than_five
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step5_evaluate import half
K = S165.K; CI = 'coordinated_isolation'
MP = CV / 'experiments/step254_mechanical_tensor_20260917/step254_features.parquet'
ev = pl.read_csv(R / 'data/development_evidence.csv')
fam372 = dict(ev.select('pair_id', 'behavior_family').unique().iter_rows())
D = S165.load(); d = D['d']; QIX = D['QIX']; top = D['top']
apairs = set(d['pair_id'].unique().to_list())
hpairs = [p for p in fam372 if p not in apairs]
pid = d['pair_id'].to_numpy(); rel = d['relevance'].to_numpy().astype(int)
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
truths = {kk[0]: set(gg['hand_id']) for kk, gg in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
famall = np.array([fams[p] for p in pid]); key = d.select('pair_id', 'hand_id', 'table_id')
FAMS = sorted(set(fam372.values()))
am = d.select(K).join(pl.scan_parquet(MP).collect(), on=K, how='left')
cols = [c for c in am.columns if c.startswith(('t1_', 't2_', 't3_', 't4_', 't5_'))]
Xte = np.nan_to_num(am.select(cols).to_numpy().astype(np.float32), nan=0., posinf=0., neginf=0.)
v = pl.read_parquet(CV / 'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(K)
fr = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', v['S_star'].to_numpy())))
t1all = align(fr, pl.read_parquet(CV / 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
cr = fr['chrono_rank'].to_numpy()
PRI = (~(t1all & (famall != CI))).astype(np.int64)
P = np.load(S / 'inc_p_PRODbare.npy'); pf = np.zeros(len(pid)); pf[QIX] = P
qall = sorted(set(pid[top]))
GT = {p: np.flatnonzero((pid == p) & top) for p in qall}
NREL = {q: len(truths[q]) for q in qall}
CACHE = OUT / 'induced_rule_audit.npy'
if CACHE.exists():
    rule = np.load(CACHE); print('loaded cached induced rule', flush=True)
else:
    hands = pl.read_parquet(R / 'data/hands.parquet', columns=['hand_id', 'started_at'])
    tr = pl.scan_parquet(MP).filter(pl.col('pair_id').is_in(hpairs)).collect().join(hands, on='hand_id', how='left')
    tr = tr.join(ev.select('pair_id', 'hand_id', 'evidence_rank'), on=['pair_id', 'hand_id'], how='left')
    tr = tr.with_columns(pl.col('pair_id').replace_strict(fam372, default=None).alias('family')).sort(['pair_id', 'started_at'])
    Xtr = np.nan_to_num(tr.select(cols).to_numpy().astype(np.float32), nan=0., posinf=0., neginf=0.)
    ytr = np.array([x is not None for x in tr['evidence_rank'].to_list()]).astype(int)
    ftr = tr['family'].to_numpy()
    PAR = dict(objective='binary', n_estimators=400, num_leaves=15, max_depth=4, min_child_samples=30,
               learning_rate=.05, subsample=.8, subsample_freq=1, colsample_bytree=.5, n_jobs=2, verbosity=-1)
    rule = np.zeros(len(pid)); t0 = time.perf_counter()
    for f in FAMS:
        mtr = ftr == f
        ix = np.concatenate([GT[q] for q in qall if fams[q] == f])
        acc = np.zeros(len(ix))
        for sd in (11, 23, 37, 51, 67, 79, 93, 107):
            acc += lgb.LGBMClassifier(random_state=sd, **PAR).fit(Xtr[mtr], ytr[mtr]).predict_proba(Xte[ix])[:, 1]
        rule[ix] = acc / 8
        print('  fitted %-24s (%.0fs)' % (f, time.perf_counter() - t0), flush=True)
    np.save(CACHE, rule)


def ap5(hits, q):
    h = 0; s = 0.
    for i, r in enumerate(hits, 1):
        if r:
            h += 1; s += h / i
    return s / min(5, NREL[q])


def blend(scope, w):
    pu = pf.copy()
    if w > 0:
        sel = np.ones(len(pid), bool) if scope == 'ALL' else (famall == scope)
        pu[sel] = np.clip(pf[sel], 1e-9, 1.) ** (1 - w) * np.clip(rule[sel], 1e-9, 1.) ** w
    return pu


def run(pu):
    sc = np.zeros(len(pid)); m = []; sel5 = {}
    for q in qall:
        g = GT[q]; p = np.clip(pu[g], 0., 1.)
        o = np.lexsort((cr[g], PRI[g])); wv = np.empty(len(g)); wv[o] = less_than_five(p[o])
        sc[g] = 10.0 + p * wv + 1e-9 * p
        s5 = g[np.argsort(-sc[g], kind='stable')[:5]]
        sel5[q] = tuple(s5); m.append(ap5(rel[s5], q))
    return np.array(m), sel5


D0, SEL0 = run(pf)
_, mprod = summarize(key.with_columns(pl.Series('score', np.where(top, 10.0 + pf, 0.0))), truths, fams)
print('', flush=True)
print('GATE  production decoded %.6f   (summarize %.6f)' % (D0.mean(), np.mean([mprod[q] for q in qall])), flush=True)
WS = (0.0, .05, .1, .15, .2, .3, .5)
SCOPES = ['ALL'] + FAMS
CELL = [(s, w) for s in SCOPES for w in WS]
A = np.empty((len(CELL), len(qall))); SELS = {}
for k, (s, w) in enumerate(CELL):
    m, sl = run(blend(s, w)); A[k] = m; SELS[(s, w)] = sl
base = A[CELL.index(('ALL', 0.0))]
tabq = np.array([ptab[q] for q in qall]); fqa = np.array([fams[q] for q in qall])
hva = np.array([half(ptab[q]) for q in qall])
pa = np.array(sorted(set(tabq))); rr = np.random.default_rng(277)
held = []; picks = []
for _ in range(200):
    pr = rr.permutation(len(pa))
    Am = np.isin(tabq, pa[pr[:len(pa) // 2]]); Bm = ~Am
    for trm, tem in ((Am, Bm), (Bm, Am)):
        kk = int(np.argmax(A[:, trm].mean(1)))
        held.append(float(A[kk, tem].mean() - base[tem].mean())); picks.append(CELL[kk])
held = np.array(held)
from collections import Counter
cnt = Counter(picks)
print('', flush=True)
print('HONEST split-half over %d (scope, weight) cells: held-out d_dec %+.6f   positive %d/400' % (
    len(CELL), held.mean(), int((held > 0).sum())), flush=True)
print('   modal picks: %s' % [(s, w, n) for (s, w), n in cnt.most_common(3)], flush=True)
rngb = np.random.default_rng(272); pools = sorted(set(ptab.values()))
qbp = {pp: [i for i, q in enumerate(qall) if ptab[q] == pp] for pp in pools}
IDX = [np.concatenate([qbp[pools[i]] for i in rngb.choice(len(pools), len(pools), replace=True)]) for _ in range(4000)]
print('', flush=True)
print('%-26s %9s %11s %8s %7s %7s   %9s %9s %9s' % (
    'arm', 'MAP@5', 'd_dec', 'P(>0)', 'chg', 'chgSet', 'h0', 'h1', 'famdelta'), flush=True)
res = {'honest_heldout': round(float(held.mean()), 6), 'honest_pos_400': int((held > 0).sum()),
       'modal': [[s, w, n] for (s, w), n in cnt.most_common(3)], 'arms': {}}
for s, w in [('soft_play', .3), ('soft_play', .2), ('soft_play', .5), ('ALL', .15)]:
    m = A[CELL.index((s, w))]; dl = m - base
    bs = np.array([dl[ix].mean() for ix in IDX])
    sl = SELS[(s, w)]
    chg = sum(1 for q in qall if sl[q] != SEL0[q])
    chgset = sum(1 for q in qall if set(sl[q]) != set(SEL0[q]))
    fd = dl[fqa == s].mean() if s != 'ALL' else dl.mean()
    res['arms']['%s|w=%.2f' % (s, w)] = {
        'map5': round(float(m.mean()), 6), 'd_dec': round(float(dl.mean()), 6),
        'P_gt_0': round(float((bs > 0).mean()), 4), 'lists_changed': chg, 'lists_set_changed': chgset,
        'h0': round(float(dl[hva == 0].mean()), 6), 'h1': round(float(dl[hva == 1].mean()), 6),
        'famdelta': round(float(fd), 6)}
    r = res['arms']['%s|w=%.2f' % (s, w)]
    print('%-26s %9.6f %+11.6f %8.4f %7d %7d   %+9.5f %+9.5f   %+9.5f' % (
        '%s w=%.2f' % (s[:14], w), r['map5'], r['d_dec'], r['P_gt_0'], chg, chgset, r['h0'], r['h1'], fd), flush=True)
# per-family winner applied jointly
pu = pf.copy()
for f in FAMS:
    kk = max(WS, key=lambda w: A[CELL.index((f, w))][fqa == f].mean())
    if kk > 0:
        sel = famall == f
        pu[sel] = np.clip(pf[sel], 1e-9, 1.) ** (1 - kk) * np.clip(rule[sel], 1e-9, 1.) ** kk
    res.setdefault('famsplit_w', {})[f] = kk
m, sl = run(pu); dl = m - base
bs = np.array([dl[ix].mean() for ix in IDX])
print('', flush=True)
print('FAMSPLIT (per-family best w = %s, chosen IN-SAMPLE): decoded %.6f  d_dec %+.6f  P(>0) %.4f' % (
    res['famsplit_w'], m.mean(), dl.mean(), (bs > 0).mean()), flush=True)
res['famsplit'] = {'map5': round(float(m.mean()), 6), 'd_dec': round(float(dl.mean()), 6),
                   'P_gt_0': round(float((bs > 0).mean()), 4)}
json.dump(res, open(OUT / 'gen_honest.json', 'w'), indent=2)
print('', flush=True)
print('wrote gen_honest.json', flush=True)
