"""STEP277-F: induce the generator's severity function from the labels, OUT OF SAMPLE, and race it against
production.  The honest end of the generator-rule lane.

What the lane has established so far.  The hard constraints are real and recoverable (0 violations of the main
ones over 771 audit truth hands, .9957 coverage for the full conjunction derived out of sample) but production
already respects them in its output, so they are worth +0.000000; and removing provably-uncitable hands from the
censoring accumulation reads -.000988, real information (its size-matched random control reads -.016848) but
still worse than leaving them in.

What is NOT yet tested is the continuous part.  No SINGLE mechanical column reproduces the cited set (<= 3% exact,
recall@k <= .356), but the generator's severity function need not be a single column.  So: fit it, per family, on
the 662 frozen mechanical columns over the 210 labelled pairs OUTSIDE the audit -- their FULL candidate universe,
~121 hands each, not a top-20 pool -- and evaluate on the audit's 162, which no part of the fit has seen.

That split is what makes this the cleanest instrument in the whole session: train and test pairs are disjoint by
construction, so the +-.004 fold-assignment noise that closed the refit lanes cannot enter, and nothing is
selected on the audit.

Note on the metric, because it changes what is worth fitting: `evidence_rank` does NOT enter MAP@5 at all -- only
SET membership does, plus our own ordering of the hits.  So the target here is citation, not rank.

Race, all on the audit's 162:
  within-top-20 AUC        against production's .9334
  hits in the top five     against production's 3.858 of 5
  decoded MAP@5            rule alone, and blended with production's p as p^(1-w) * rule^w, w on a small grid
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
print('train pairs (outside the audit) %d   test pairs (the audit) %d' % (len(hpairs), len(apairs)), flush=True)
hands = pl.read_parquet(R / 'data/hands.parquet', columns=['hand_id', 'started_at'])
tr = pl.scan_parquet(MP).filter(pl.col('pair_id').is_in(hpairs)).collect().join(hands, on='hand_id', how='left')
tr = tr.join(ev.select('pair_id', 'hand_id', 'evidence_rank'), on=['pair_id', 'hand_id'], how='left')
tr = tr.with_columns(pl.col('pair_id').replace_strict(fam372, default=None).alias('family')).sort(['pair_id', 'started_at'])
cols = [c for c in tr.columns if c.startswith(('t1_', 't2_', 't3_', 't4_', 't5_'))]
Xtr = np.nan_to_num(tr.select(cols).to_numpy().astype(np.float32), nan=0., posinf=0., neginf=0.)
ytr = np.array([v is not None for v in tr['evidence_rank'].to_list()]).astype(int)
ftr = tr['family'].to_numpy(); ptr = tr['pair_id'].to_numpy()
FAMS = sorted(set(fam372.values()))
print('train rows %d  positives %d  columns %d' % (len(ytr), int(ytr.sum()), len(cols)), flush=True)
pid = d['pair_id'].to_numpy(); rel = d['relevance'].to_numpy().astype(int)
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
truths = {kk[0]: set(gg['hand_id']) for kk, gg in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
famall = np.array([fams[p] for p in pid]); key = d.select('pair_id', 'hand_id', 'table_id')
am = d.select(K).join(pl.scan_parquet(MP).collect(), on=K, how='left')
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
# ---- fit per family, bagged over seeds (huge-ensemble directive)
PAR = dict(objective='binary', n_estimators=400, num_leaves=15, max_depth=4, min_child_samples=30,
           learning_rate=.05, subsample=.8, subsample_freq=1, colsample_bytree=.5, n_jobs=2, verbosity=-1)
SEEDS = (11, 23, 37, 51, 67, 79, 93, 107)
rule = np.zeros(len(pid))
t0 = time.perf_counter()
for f in FAMS:
    mtr = ftr == f
    rows_te = [q for q in qall if fams[q] == f]
    ix = np.concatenate([GT[q] for q in rows_te])
    acc = np.zeros(len(ix))
    for sd in SEEDS:
        m = lgb.LGBMClassifier(random_state=sd, **PAR).fit(Xtr[mtr], ytr[mtr])
        acc += m.predict_proba(Xte[ix])[:, 1]
    rule[ix] = acc / len(SEEDS)
    print('  %-24s trained on %d rows (%d pos) x %d seeds -> %d audit rows  (%.0fs)' % (
        f, int(mtr.sum()), int(ytr[mtr].sum()), len(SEEDS), len(ix), time.perf_counter() - t0), flush=True)


def wauc(sig):
    num = den = 0.
    for q in qall:
        g = GT[q]; x = sig[g].astype(float); y = rel[g]
        pos = x[y == 1]; neg = x[y == 0]
        if not len(pos) or not len(neg):
            continue
        num += (pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()
        den += len(pos) * len(neg)
    return num / den


def ap5(hits, q):
    h = 0; s = 0.
    for i, r in enumerate(hits, 1):
        if r:
            h += 1; s += h / i
    return s / min(5, NREL[q])


def dec_and_score(p_use):
    sc = np.zeros(len(pid)); m = []
    for q in qall:
        g = GT[q]; p = np.clip(p_use[g], 0., 1.)
        o = np.lexsort((cr[g], PRI[g])); w = np.empty(len(g)); w[o] = less_than_five(p[o])
        sc[g] = 10.0 + p * w + 1e-9 * p
        s5 = g[np.argsort(-sc[g], kind='stable')[:5]]
        m.append(ap5(rel[s5], q))
    return sc, np.array(m)


sc0, D0 = dec_and_score(pf)
_, mprod = summarize(key.with_columns(pl.Series('score', sc0)), truths, fams)
assert np.allclose(D0, np.array([mprod[q] for q in qall])), 'GATE'
hits0 = np.mean([rel[GT[q][np.argsort(-sc0[GT[q]], kind='stable')[:5]]].sum() for q in qall])
print('', flush=True)
print('GATE ok.  production: within-top-20 AUC %.4f   hits %.3f/5   decoded %.6f' % (
    wauc(pf), hits0, D0.mean()), flush=True)
ra = wauc(rule)
scr, Dr = dec_and_score(rule)
hitsr = np.mean([rel[GT[q][np.argsort(-scr[GT[q]], kind='stable')[:5]]].sum() for q in qall])
print('induced rule   : within-top-20 AUC %.4f   hits %.3f/5   decoded %.6f   (%+.6f)' % (
    ra, hitsr, Dr.mean(), Dr.mean() - D0.mean()), flush=True)
fqa = np.array([fams[q] for q in qall]); hva = np.array([half(ptab[q]) for q in qall])
rngb = np.random.default_rng(272); pools = sorted(set(ptab.values()))
qbp = {pp: [i for i, q in enumerate(qall) if ptab[q] == pp] for pp in pools}
IDX = [np.concatenate([qbp[pools[i]] for i in rngb.choice(len(pools), len(pools), replace=True)]) for _ in range(4000)]
print('', flush=True)
print('%-14s %9s %9s %11s %8s   %9s %9s   %9s %9s %9s' % (
    'blend w', 'AUC20', 'MAP@5', 'd_dec', 'P(>0)', 'h0', 'h1', 'DT', 'SP', 'CI'), flush=True)
res = {'prod': {'auc20': round(float(wauc(pf)), 4), 'hits': round(float(hits0), 3), 'dec': round(float(D0.mean()), 6)},
       'rule': {'auc20': round(float(ra), 4), 'hits': round(float(hitsr), 3), 'dec': round(float(Dr.mean()), 6)},
       'blends': {}}
for w in (0.0, .05, .1, .15, .2, .3, .5):
    pu = np.clip(pf, 1e-9, 1.) ** (1 - w) * np.clip(rule, 1e-9, 1.) ** w if w > 0 else pf
    scb, m = dec_and_score(pu)
    dl = m - D0; bs = np.array([dl[ix].mean() for ix in IDX])
    res['blends']['w=%.2f' % w] = {'auc20': round(float(wauc(pu)), 4), 'map5': round(float(m.mean()), 6),
                                   'd_dec': round(float(dl.mean()), 6), 'P_gt_0': round(float((bs > 0).mean()), 4),
                                   'DT': round(float(dl[fqa == 'directed_transfer'].mean()), 6),
                                   'SP': round(float(dl[fqa == 'soft_play'].mean()), 6),
                                   'CI': round(float(dl[fqa == CI].mean()), 6)}
    r = res['blends']['w=%.2f' % w]
    print('%-14.2f %9.4f %9.6f %+11.6f %8.4f   %+9.5f %+9.5f   %+9.5f %+9.5f %+9.5f' % (
        w, r['auc20'], r['map5'], r['d_dec'], r['P_gt_0'], dl[hva == 0].mean(), dl[hva == 1].mean(),
        r['DT'], r['SP'], r['CI']), flush=True)
json.dump(res, open(OUT / 'gen_induce.json', 'w'), indent=2)
print('', flush=True)
print('wrote gen_induce.json  (%.0fs)' % (time.perf_counter() - t0), flush=True)
