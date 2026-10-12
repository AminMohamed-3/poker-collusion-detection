"""STEP133 v2 (poker-a8; session now listed as poker-1b): capacity learning curves projected to production scale.
Registration: PREDECLARE_step133v2_capacity_curve.md. Loader and S* recipe copied read-only from poker-09's replay_v1_step124.py;
A1 = poker-11's run130.py A1_noview recipe; rs members from registration129.json. Audit look only: no held read, nothing kept."""
REG_SHA = '835c28fac73b646daa49a5f18d81dd4faa4394894b49b25047f02179e3cb7f4b'
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
sys.path.insert(0, ROOT + 'Claude Version/src')
sys.path.insert(0, ROOT + 'Codex Resume/src')
import json, time, hashlib
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.model_selection import GroupKFold
import audit_common as ac
from evidence_aug.step104_C_typed_decoder_build import chronology, decode, align
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step65b_family_stage2 import boot
CVS, CR = ROOT + 'Claude Version/', ROOT + 'Codex Resume/'
OUT = CVS + 'experiments/learning_curve_a8_20260913/'
E130, E129 = CVS + 'experiments/typed_attributes_increment_C_20260913/', CVS + 'experiments/stage2_ensemble_C_20260913/'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
REG = sha(OUT + 'PREDECLARE_step133v2_capacity_curve.md')
assert REG == REG_SHA, 'registration changed'
PIN = {E130 + 'registration130.json': 'f1c11e24bf61dfc0309e91cf11e2e48e60682b6186003a8e780ba4b60e6643a6',
       E130 + 'audit_step130_scores.parquet': '398885a85ff379f53abadca969f71bf74ec152211120b2b986ed360eb6060dfe',
       E129 + 'registration129.json': '24ee7766f45f1e9f7cb37e45075f12cdd0589e9e53e31e23b47027b20648afe6',
       E129 + 'audit_step129_scores.parquet': '1b294e78202284239437626fcf840a7a92625ddbce67549fc497ed5ae39166e6'}
for _p, _h in PIN.items():
    assert sha(_p) == _h, _p
K, CI, FAM = ['pair_id', 'hand_id'], 'coordinated_isolation', ac.FAMILIES
PHIS, SEEDS, FIT = (.30, .45, .60, .75, .90), range(5), (.45, .60, .75, .90)
T0 = time.time()
log = lambda *a: print(*a, f'{time.time() - T0:.0f}s', flush=True)
prot0 = ac.check_protected()
d, shared, rcols, saved = ac.load_current()
top = (d['oof_rank'] <= 20).to_numpy()
cs = pl.read_parquet(CR + 'research/contestable_state_20260912/features.parquet').sort(K)
cs_cols = [c for c in cs.columns if c not in K]
assert len(cs_cols) == 29
pf = pl.read_parquet(CR + 'research/retrospective_strength_20260912/prefix.parquet').sort(K)
pf_cols = json.load(open(CR + 'research/retrospective_strength_20260912/freeze.json', encoding='utf-8'))['features']
assert pf.columns == K + pf_cols
Q = d.filter(pl.col('oof_rank') <= 20).join(cs, on=K, validate='1:1').join(pf, on=K, validate='1:1').sort(K)
QIX = np.flatnonzero(top)
assert d[QIX].select(K).equals(Q.select(K))
BL = json.load(open(E130 + 'registration130.json', encoding='utf-8'))['block']
AP = CVS + BL['source']
assert sha(AP) == BL['source_sha256'] and len(BL['block52']) == 52
A = Q.select(K).join(pl.read_parquet(AP).select(K + BL['block52'] + ['T1']), on=K, how='left', validate='1:1', maintain_order='left')
assert A.select(K).equals(Q.select(K))
T1A = A['T1'].cast(pl.Float64).to_numpy()
EXA = np.hstack([A.select([pl.col(c).cast(pl.Float64) for c in BL['block52']]).to_numpy(), T1A[:, None]])
MEM = json.load(open(E129 + 'registration129.json', encoding='utf-8'))['members'][:30]
assert [m['name'] for m in MEM] == ['rs%02d' % i for i in range(30)] and all(m['learner'] == 'lgbm' for m in MEM)
FOLDS = {}
for f in FAM:
    ix = np.flatnonzero((Q['behavior_family'] == f).to_numpy())
    q = Q[ix]; y = q['relevance'].to_numpy().astype(int); g = q['table_id'].to_numpy()
    cols = shared + (rcols if f != CI else []) + cs_cols + pf_cols
    FOLDS[f] = (ix, q['pair_id'].to_numpy(), y, g, q.select(cols).to_numpy().astype(float), list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), y, g)))


def run(phi, seed, arms, rs_all=False):
    P = {a: d['oof'].to_numpy().copy() - 1e6 for a in arms}
    M30, ntr = (np.zeros((30, Q.height)) if rs_all else None), {f: [] for f in FAM}
    for fi, f in enumerate(FAM):
        ix, pid, y, g, z, folds = FOLDS[f]
        za = np.hstack([z, EXA[ix]])
        for k, (tr, te) in enumerate(folds):
            if phi < 1:
                tabs = np.unique(g[tr])
                keep = np.random.default_rng([133, seed, fi, k]).choice(tabs, max(2, int(round(phi * len(tabs)))), replace=False)
                tr = tr[np.isin(g[tr], keep)]
            ntr[f].append(int(len(np.unique(pid[tr]))))
            w = QIX[ix[te]]
            P['S'][w] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(z[tr], y[tr]).predict_proba(z[te])[:, 1] + 10
            P['A1'][w] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(za[tr], y[tr]).predict_proba(za[te])[:, 1] + 10
            for mi in range(30 if rs_all else int('rs' in arms)):
                pr = lgb.LGBMClassifier(**MEM[mi]['params']).fit(z[tr], y[tr]).predict_proba(z[te])[:, 1]
                if mi == 0 and 'rs' in arms:
                    P['rs'][w] = pr + 10
                if rs_all:
                    M30[mi, ix[te]] += pr
    return P, ntr, M30


truths = {k_[0]: set(g_['hand_id']) for k_, g_ in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
pools = sorted(set(ptab.values()))
key = d.select('pair_id', 'hand_id', 'table_id')
QORD = sorted(summarize(key.with_columns(pl.Series('score', d['oof'].to_numpy())), truths, fams)[1])
assert len(QORD) == 162
famq = np.array([fams[q_] for q_ in QORD])
yv = d['relevance'].to_numpy()[top]


def logloss(p):
    v = np.clip(p[top] - 10, 1e-6, 1 - 1e-6)
    return float(-np.mean(yv * np.log(v) + (1 - yv) * np.log(1 - v)))


def ev(preds):
    S, M, out = {}, {}, {}
    for a, p in preds.items():
        S[a], M[a] = summarize(key.with_columns(pl.Series('score', p)), truths, fams)
        v = np.array([M[a][q_] for q_ in QORD])
        out[a] = {'map5': float(S[a]['map5']), 'map5_minus_ap_mean': float(S[a]['map5'] - v.mean()), 'fam': {f: float(v[famq == f].mean()) for f in FAM}, '_ap': v}
    for a in preds:
        if a != 'S':
            c = boot(M['S'], M[a], ptab, pools)
            out[a].update(delta=float(c['mean_delta']), ci95=[float(x) for x in c['ci95']], delta_fam={f: out[a]['fam'][f] - out['S']['fam'][f] for f in FAM})
    return out


APS = {}


def record(phi, seed, P, ntr):
    D = {a: dec(P[a]) for a in P}
    row = {'phi': phi, 'seed': seed, 'n': ntr, 'n_mean': float(np.mean([v for f in FAM for v in ntr[f]])), 'n_fam': {f: float(np.mean(ntr[f])) for f in FAM},
           'raw': ev(P), 'dec': ev(D), 'logloss': {a: logloss(P[a]) for a in P}}
    APS[(phi, seed)] = {kk: {a: row[kk][a].pop('_ap') for a in P} for kk in ('raw', 'dec')}
    return row


sc = pl.read_parquet(CVS + 'experiments/codex_ideas_sweep_C_20260913/scores-greedy-step1-prefix_strength.parquet').sort(K)
assert sc.select(K).equals(d.select(K))
jn = lambda p_, c_: d.select(K).join(pl.read_parquet(p_).select(K + [c_]), on=K, how='left', validate='1:1', maintain_order='left')[c_].to_numpy()
P1, N1, M30 = run(1.0, 0, ('S', 'A1', 'rs'), rs_all=True)
SS = P1['S']
G = {'a_S_star_exact': bool(np.array_equal(SS, sc['S_plus_X'].to_numpy()))}
RES = {'registration_sha256': REG, 'pins': PIN, 'gates': G, 'runs': []}
dump = lambda: json.dump(RES, open(OUT + 'step133_results.json', 'w', encoding='utf-8'), indent=1)
log('phi 1.0 fitted', G)
if not G['a_S_star_exact']:
    dump(); sys.exit('ABORT: gate a (S* replay)')
fr0 = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', SS)))
assert fr0.select(K).equals(d.select(K))
t1 = align(fr0, pl.read_parquet(CVS + 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
dec = lambda p: decode(fr0.with_columns(pl.Series('score', p)), t1)
G['b_T1_equal'] = bool(np.array_equal(T1A, np.asarray(t1)[QIX].astype(float)))
G['b_A1_exact'] = bool(np.array_equal(P1['A1'], jn(E130 + 'audit_step130_scores.parquet', 'A1_noview')))
if not (G['b_T1_equal'] and G['b_A1_exact']):
    dump(); sys.exit('ABORT: gate b (A1 replay or T1)')
r30 = jn(E129 + 'audit_step129_scores.parquet', 'E_rs30_noview')
p30 = SS.copy(); p30[QIX] = M30.mean(axis=0) + 10
G['c_rs30_exact'], G['c_rs30_maxabs'] = bool(np.array_equal(p30, r30)), float(np.abs(p30 - r30).max())
ARMS = ('S', 'A1', 'rs') if G['c_rs30_exact'] else ('S', 'A1')
if 'rs' not in ARMS:
    del P1['rs']
G['d_protected_unchanged_at_gates'] = bool(ac.check_protected() == prot0)
dump(); log('gates', G)
RES['full'] = record(1.0, 0, P1, N1)
x30 = {kk: ev({'S': fn(SS), 'rs30': fn(p30)})['rs30'] for kk, fn in (('raw', lambda p: p), ('dec', dec))}
RES['full']['rs30_descriptive'] = {kk: {k_: v_ for k_, v_ in x30[kk].items() if k_ != '_ap'} for kk in x30}
dump(); log('full', {kk: {a: round(RES['full'][kk][a]['map5'], 4) for a in ARMS} for kk in ('raw', 'dec')})
for phi in PHIS:
    for s in SEEDS:
        P, ntr, _ = run(phi, s, ARMS)
        r = record(phi, s, P, ntr); RES['runs'].append(r); dump()
        log('phi', phi, 'seed', s, 'n', round(r['n_mean'], 1), {kk: {a: round(r[kk][a]['map5'], 4) for a in ARMS} for kk in ('raw', 'dec')})


LAB = [(1.0, 0)] + [(x, s) for x in PHIS for s in SEEDS]
RI = {lab: i for i, lab in enumerate(LAB)}
NMEAN = np.array([RES['full']['n_mean']] + [r['n_mean'] for r in RES['runs']])
NFAM = {f: np.array([RES['full']['n_fam'][f]] + [r['n_fam'][f] for r in RES['runs']]) for f in FAM}
V = {kk: {a: np.stack([APS[lab][kk][a] for lab in LAB]) for a in ARMS} for kk in ('raw', 'dec')}
np.savez(OUT + 'step133_query_ap.npz', qord=np.array(QORD), labels=np.array(LAB), **{kk + '_' + a: V[kk][a] for kk in V for a in ARMS})
SER = {kk: dict([('S', V[kk]['S'])] + [(a + '-S', V[kk][a] - V[kk]['S']) for a in ARMS if a != 'S']) for kk in V}
TI = np.unique(np.array([ptab[q_] for q_ in QORD]), return_inverse=True)[1]
NT = int(TI.max()) + 1
fitj = json.load(open(CVS + 'outputs_stage2_r382_20260911/fit.json', encoding='utf-8'))['stage2']
NPROD = {f: int(fitj['r382/' + f]['training_pairs']) for f in FAM}
DF = {f: float(np.log2(NPROD[f] / RES['full']['n_fam'][f])) for f in FAM}
NQ = {f: int((famq == f).sum()) for f in FAM}
DBAR = float(sum(NQ[f] * DF[f] for f in FAM) / sum(NQ.values()))


def slope(Y, w, sel, nm=NMEAN):
    m = Y @ w / w.sum()
    ii = {x: [RI[(x, s)] for s in sel[x]] for x in FIT}
    xs = [np.log2(nm[0])] + [np.log2(nm[ii[x]].mean()) for x in FIT]
    ys = [m[0]] + [m[ii[x]].mean() for x in FIT]
    return float(np.polyfit(xs, ys, 1)[0]), float(m[0])


ALL5, ONE = {x: list(range(5)) for x in PHIS}, np.ones(len(QORD))
rng, rng2 = np.random.default_rng(1330), np.random.default_rng(1331)
DRAWS = [(np.bincount(rng.integers(0, NT, NT), minlength=NT)[TI].astype(float), {x: rng.integers(0, 5, 5) for x in FIT}) for _ in range(2000)]
SDRAWS = [{x: rng2.integers(0, 5, 5) for x in FIT} for _ in range(2000)]
qq = lambda a_: [float(np.quantile(a_, .05)), float(np.quantile(a_, .95))]
CUR = {}
for kk in SER:
    CUR[kk] = {}
    for nm_, Y in SER[kk].items():
        dl = nm_ != 'S'
        s0, d1 = slope(Y, ONE, ALL5)
        bj = np.array([slope(Y, w, sel) for w, sel in DRAWS])
        bs = np.array([slope(Y, ONE, sel)[0] for sel in SDRAWS])
        pm = Y.mean(axis=1)
        e = {'slope': s0, 'at_phi1': d1, 'slope_q05_q95_joint': qq(bj[:, 0]), 'slope_q05_q95_seeds': qq(bs),
             'proj_production': d1 * dl + s0 * DBAR, 'proj_q05_q95': qq(bj[:, 1] * dl + bj[:, 0] * DBAR),
             'points': [{'phi': x, 'n_mean': float(NMEAN[[RI[(x, s)] for s in SEEDS]].mean()), 'mean': float(pm[[RI[(x, s)] for s in SEEDS]].mean()),
                         'sd': float(pm[[RI[(x, s)] for s in SEEDS]].std(ddof=1))} for x in PHIS], 'by_family': {}}
        e['composite'], e['composite_x057'] = .2 * e['proj_production'], .57 * .2 * e['proj_production']
        for f in FAM:
            sf, f1 = slope(Y, (famq == f).astype(float), ALL5, NFAM[f])
            e['by_family'][f] = {'slope': sf, 'at_phi1': f1, 'proj_production': f1 * dl + sf * DF[f]}
        CUR[kk][nm_] = e
    log('curve', kk, {nm_: [round(CUR[kk][nm_]['slope'], 4), round(CUR[kk][nm_]['proj_production'], 4)] for nm_ in CUR[kk]})


READ = {}
for a in ARMS:
    if a != 'S':
        e = CUR['dec'][a + '-S']
        READ[a] = 'UNDERSTATED' if (e['slope_q05_q95_joint'][0] > 0 and e['proj_production'] >= .005) else ('NOT_UNDERSTATED' if e['proj_q05_q95'][1] < .005 else 'INCONCLUSIVE')
from scipy.optimize import curve_fit
for kk in CUR:
    e = CUR[kk]['S']
    try:
        xs = np.array([NMEAN[0]] + [p_['n_mean'] for p_ in e['points']]); ys = np.array([e['at_phi1']] + [p_['mean'] for p_ in e['points']])
        (a_, b_, c_), _ = curve_fit(lambda n, a, b, c: a - b * n ** (-c), xs, ys, p0=[ys.max() + .02, 1., .5], bounds=([0, 0, 0], [1, 100, 5]), maxfev=20000)
        e['saturating'] = {'a': float(a_), 'b': float(b_), 'c': float(c_), 'gain_audit_to_production': float(b_ * (NMEAN[0] ** -c_ - (NMEAN[0] * 2 ** DBAR) ** -c_))}
    except Exception as ex:
        e['saturating'] = 'not identifiable: ' + type(ex).__name__
RES.update(curve=CUR, reading=READ, arms=list(ARMS), production_training_queries=NPROD, audit_training_queries_phi1=RES['full']['n_fam'], doublings_by_family=DF,
           doublings_weighted=DBAR, queries_by_family=NQ, bootstrap={'joint_rng': 1330, 'seeds_only_rng': 1331, 'draws': 2000, 'tables': NT},
           protected_unchanged=bool(ac.check_protected() == prot0), seconds=round(time.time() - T0))
dump()
log('READING', READ, 'D', round(DBAR, 3), 'DF', DF)
for kk in CUR:
    log(kk, json.dumps({nm_: {k_: v_ for k_, v_ in e.items() if k_ not in ('points', 'by_family')} for nm_, e in CUR[kk].items()}))
