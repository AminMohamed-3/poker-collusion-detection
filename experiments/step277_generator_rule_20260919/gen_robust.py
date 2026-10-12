"""STEP277-H: is the soft-play gain the LABELS, or just the blend?  The control battery.

STEP277-G: an induced citation rule, fitted on the 210 labelled pairs OUTSIDE the audit and blended into
production's p as p^(1-w)*rule^w, gives SOFT PLAY +.02809 (whole audit +.008148, P .9595, 39 lists changed, 16
with a different set), and an honest split-half over 28 (scope, weight) cells pays for the selection and still
returns +.003039 with 307/400 positive.  That is the first arm all night to survive its own honest test.

Before it can be recommended it has to survive the controls, and one of them is decisive:

  PERM      train on citation labels PERMUTED WITHIN EACH PAIR.  The rule then carries no information about which
            hand is cited, but the blend still shrinks p toward a second score.  If PERM also helps soft play, the
            gain is the blend's arithmetic, not the generator's labels, and the lane is dead.
  HP x5     five hyperparameter settings.  A gain that exists only at one depth is a fitting artefact.
  SUBSET    ten random 70% subsets of the 210 training pairs.  A gain that depends on the training draw is noise.
  SHUF_FAM  train soft play's rule on the OTHER families' pairs -- wrong-family information, another null.

Reported on soft play's own 47 audit queries at w = .30, the modal honest pick, plus the whole-audit delta.
Nothing here is selected: w and scope are fixed at what STEP277-G's honest split-half already chose.
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
K = S165.K; CI = 'coordinated_isolation'; SP = 'soft_play'; W = .30
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
fqa = np.array([fams[q] for q in qall]); hva = np.array([half(ptab[q]) for q in qall])
hands = pl.read_parquet(R / 'data/hands.parquet', columns=['hand_id', 'started_at'])
tr = pl.scan_parquet(MP).filter(pl.col('pair_id').is_in(hpairs)).collect().join(hands, on='hand_id', how='left')
tr = tr.join(ev.select('pair_id', 'hand_id', 'evidence_rank'), on=['pair_id', 'hand_id'], how='left')
tr = tr.with_columns(pl.col('pair_id').replace_strict(fam372, default=None).alias('family')).sort(['pair_id', 'started_at'])
Xtr = np.nan_to_num(tr.select(cols).to_numpy().astype(np.float32), nan=0., posinf=0., neginf=0.)
ytr = np.array([x is not None for x in tr['evidence_rank'].to_list()]).astype(int)
ftr = tr['family'].to_numpy(); ptr = tr['pair_id'].to_numpy()
SPTR = np.flatnonzero(ftr == SP); SPPAIRS = np.array(sorted(set(ptr[SPTR])))
OTHTR = np.flatnonzero(ftr != SP)
IXSP = np.concatenate([GT[q] for q in qall if fams[q] == SP])
print('soft play: train rows %d (%d pos, %d pairs)   audit rows %d over %d queries' % (
    len(SPTR), int(ytr[SPTR].sum()), len(SPPAIRS), len(IXSP), int((fqa == SP).sum())), flush=True)


def ap5(hits, q):
    h = 0; s = 0.
    for i, r in enumerate(hits, 1):
        if r:
            h += 1; s += h / i
    return s / min(5, NREL[q])


def run(pu):
    sc = np.zeros(len(pid)); m = []; sel = {}
    for q in qall:
        g = GT[q]; p = np.clip(pu[g], 0., 1.)
        o = np.lexsort((cr[g], PRI[g])); wv = np.empty(len(g)); wv[o] = less_than_five(p[o])
        sc[g] = 10.0 + p * wv + 1e-9 * p
        s5 = g[np.argsort(-sc[g], kind='stable')[:5]]
        sel[q] = tuple(s5); m.append(ap5(rel[s5], q))
    return np.array(m), sel


D0, SEL0 = run(pf)
_, mprod = summarize(key.with_columns(pl.Series('score', np.array([10.0 + pf[i] if top[i] else 0.0 for i in range(len(pf))]))), truths, fams)
sc0 = np.zeros(len(pid))
for q in qall:
    g = GT[q]; p = np.clip(pf[g], 0., 1.)
    o = np.lexsort((cr[g], PRI[g])); wv = np.empty(len(g)); wv[o] = less_than_five(p[o])
    sc0[g] = 10.0 + p * wv + 1e-9 * p
_, mchk = summarize(key.with_columns(pl.Series('score', sc0)), truths, fams)
assert np.allclose(D0, np.array([mchk[q] for q in qall])), 'GATE: AP@5 must equal summarize on the decoded score'
print('GATE ok (decoded vs summarize exact).  production decoded %.6f', flush=True)
print('   production decoded %.6f   soft-play subset %.6f' % (D0.mean(), D0[fqa == SP].mean()), flush=True)
rngb = np.random.default_rng(272); pools = sorted(set(ptab.values()))
qbp = {pp: [i for i, q in enumerate(qall) if ptab[q] == pp] for pp in pools}
IDX = [np.concatenate([qbp[pools[i]] for i in rngb.choice(len(pools), len(pools), replace=True)]) for _ in range(4000)]
BASEHP = dict(objective='binary', n_estimators=400, num_leaves=15, max_depth=4, min_child_samples=30,
              learning_rate=.05, subsample=.8, subsample_freq=1, colsample_bytree=.5, n_jobs=2, verbosity=-1)


def fit_rule(rows, y, hp, seeds=(11, 23, 37)):
    acc = np.zeros(len(IXSP))
    for sd in seeds:
        acc += lgb.LGBMClassifier(random_state=sd, **hp).fit(Xtr[rows], y[rows]).predict_proba(Xte[IXSP])[:, 1]
    return acc / len(seeds)


def evaluate(rulesp):
    pu = pf.copy()
    pu[IXSP] = np.clip(pf[IXSP], 1e-9, 1.) ** (1 - W) * np.clip(rulesp, 1e-9, 1.) ** W
    m, sel = run(pu); dl = m - D0
    bs = np.array([dl[ix].mean() for ix in IDX])
    return {'d_all': float(dl.mean()), 'd_sp': float(dl[fqa == SP].mean()), 'P': float((bs > 0).mean()),
            'h0': float(dl[hva == 0].mean()), 'h1': float(dl[hva == 1].mean()),
            'chg': sum(1 for q in qall if sel[q] != SEL0[q]),
            'chgset': sum(1 for q in qall if set(sel[q]) != set(SEL0[q]))}


t0 = time.perf_counter(); res = {}
print('', flush=True)
print('%-30s %11s %11s %8s %9s %9s %6s %6s' % ('arm', 'd_all', 'd_softplay', 'P(>0)', 'h0', 'h1', 'chg', 'set'), flush=True)


def show(nm, r):
    print('%-30s %+11.6f %+11.6f %8.4f %+9.5f %+9.5f %6d %6d' % (
        nm, r['d_all'], r['d_sp'], r['P'], r['h0'], r['h1'], r['chg'], r['chgset']), flush=True)
    res[nm] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}


show('REAL (3 seeds)', evaluate(fit_rule(SPTR, ytr, BASEHP)))
# ---- PERM: labels permuted within each training pair
yp = ytr.copy(); rp = np.random.default_rng(401)
for p in SPPAIRS:
    ii = SPTR[ptr[SPTR] == p]
    yp[ii] = rp.permutation(ytr[ii])
show('PERM within-pair labels', evaluate(fit_rule(SPTR, yp, BASEHP)))
show('SHUF_FAM (other families)', evaluate(fit_rule(OTHTR, ytr, BASEHP)))
print('', flush=True)
HPS = [('leaves7 depth3', dict(BASEHP, num_leaves=7, max_depth=3)),
       ('leaves31 depth6', dict(BASEHP, num_leaves=31, max_depth=6)),
       ('lr.02 n800', dict(BASEHP, learning_rate=.02, n_estimators=800)),
       ('lr.10 n200', dict(BASEHP, learning_rate=.10, n_estimators=200)),
       ('colsample.2', dict(BASEHP, colsample_bytree=.2))]
for nm, hp in HPS:
    show('HP ' + nm, evaluate(fit_rule(SPTR, ytr, hp)))
print('', flush=True)
sub = []
rs = np.random.default_rng(402)
for it in range(10):
    keep = set(rs.choice(SPPAIRS, int(.7 * len(SPPAIRS)), replace=False).tolist())
    rows = SPTR[np.isin(ptr[SPTR], list(keep))]
    r = evaluate(fit_rule(rows, ytr, BASEHP, seeds=(11, 23)))
    sub.append(r['d_sp'])
sub = np.array(sub)
print('SUBSET 70%% of training pairs, 10 draws: d_softplay mean %+.6f  sd %.6f  min %+.6f  positive %d/10' % (
    sub.mean(), sub.std(), sub.min(), int((sub > 0).sum())), flush=True)
res['SUBSET'] = {'mean': round(float(sub.mean()), 6), 'sd': round(float(sub.std()), 6),
                 'min': round(float(sub.min()), 6), 'positive': int((sub > 0).sum()), 'n': 10}
json.dump({'w': W, 'prod_dec': float(D0.mean()), 'prod_sp': float(D0[fqa == SP].mean()), 'arms': res},
          open(OUT / 'gen_robust.json', 'w'), indent=2)
print('', flush=True)
print('wrote gen_robust.json  (%.0fs)' % (time.perf_counter() - t0), flush=True)
