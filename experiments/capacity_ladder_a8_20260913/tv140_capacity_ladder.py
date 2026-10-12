"""STEP140 (poker-1b, formerly poker-a8): capacity ladder for S* on the STEP133 learning curve, projected to production scale.
Registration: PREDECLARE_step140_capacity_ladder.md. Loader, folds, subsamples, curve and bootstraps copied from tv138_v1_scale_curve.py
(itself from replay_v1_step124.py and tv133_curve.py). Audit look only: no held read, nothing kept."""
REG_SHA = '0a66cc4a71d7e235e0a4d53269d8cb9b1e1445ce342edc49553ef5156df75f3a'
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
CVS, CR = ROOT + 'Claude Version/', ROOT + 'Codex Resume/'
OUT = CVS + 'experiments/capacity_ladder_a8_20260913/'
E133 = CVS + 'experiments/learning_curve_a8_20260913/'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
REG = sha(OUT + 'PREDECLARE_step140_capacity_ladder.md')
assert REG == REG_SHA, 'registration changed'
assert sha(E133 + 'step133_results.json') == '39ffd0415d151dcd2f3fd6febcf64a6eb117aa4043bcceebbc23ec00fb8fcba1'
K, CI, FAM = ['pair_id', 'hand_id'], 'coordinated_isolation', ac.FAMILIES
PHIS, SEEDS, FIT = (.30, .45, .60, .75, .90), range(5), (.45, .60, .75, .90)
BASE = dict(ac.CLF)
assert BASE == {'n_estimators': 150, 'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 40, 'learning_rate': 0.05, 'n_jobs': 2, 'verbosity': -1}
ARMS = {'S': {}, 'C2': dict(num_leaves=15, max_depth=4, min_child_samples=20, n_estimators=300),
        'C4': dict(num_leaves=31, max_depth=5, min_child_samples=10, n_estimators=600),
        'CH': dict(num_leaves=3, max_depth=2, min_child_samples=80, n_estimators=150)}
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


def columns(f):
    return shared + (rcols if f != CI else []) + cs_cols + pf_cols


FOLDS = {}
for f in FAM:
    ix = np.flatnonzero((Q['behavior_family'] == f).to_numpy())
    q = Q[ix]; y = q['relevance'].to_numpy().astype(int); g = q['table_id'].to_numpy()
    FOLDS[f] = (ix, q['pair_id'].to_numpy(), y, g, q.select(columns(f)).to_numpy().astype(float), list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), y, g)))
truths = {k_[0]: set(g_['hand_id']) for k_, g_ in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
key = d.select('pair_id', 'hand_id', 'table_id')
QORD = sorted(summarize(key.with_columns(pl.Series('score', d['oof'].to_numpy())), truths, fams)[1])
assert len(QORD) == 162
famq = np.array([fams[q_] for q_ in QORD])
log('loaded', Q.height, {f: len(columns(f)) for f in FAM})


def fold_scores(phi, seed):
    P = {a: d['oof'].to_numpy().copy() - 1e6 for a in ARMS}
    ntr = {f: [] for f in FAM}
    for fi, f in enumerate(FAM):
        ix, pid, y, g, z, folds = FOLDS[f]
        for k, (tr, te) in enumerate(folds):
            if phi < 1:
                tabs = np.unique(g[tr])
                keep = np.random.default_rng([133, seed, fi, k]).choice(tabs, max(2, int(round(phi * len(tabs)))), replace=False)
                tr = tr[np.isin(g[tr], keep)]
            ntr[f].append(int(len(np.unique(pid[tr]))))
            for a in ARMS:
                m = lgb.LGBMClassifier(**{**BASE, **ARMS[a]}, random_state=11).fit(z[tr], y[tr])
                P[a][QIX[ix[te]]] = m.predict_proba(z[te])[:, 1] + 10
    return P, ntr


sc = pl.read_parquet(CVS + 'experiments/codex_ideas_sweep_C_20260913/scores-greedy-step1-prefix_strength.parquet').sort(K)
assert sc.select(K).equals(d.select(K))
LAB = [(1.0, 0)] + [(x, s) for x in PHIS for s in SEEDS]
J133 = json.load(open(E133 + 'step133_results.json', encoding='utf-8'))
R133 = {(r_['phi'], r_['seed']): r_ for r_ in J133['runs']}
R133[(1.0, 0)] = J133['full']
G = {'b_S_phi_replays': {}}
RES = {'registration_sha256': REG, 'arms': {a: {**BASE, **ARMS[a]} for a in ARMS}, 'gates': G, 'runs': []}
dump = lambda: json.dump(RES, open(OUT + 'results_step140.json', 'w', encoding='utf-8'), indent=1)
APS, dec = {}, None


def aps(P):
    out = {}
    for kind, fn in (('raw', lambda p: p), ('dec', dec)):
        out[kind] = {}
        for a, p in P.items():
            S_, M_ = summarize(key.with_columns(pl.Series('score', fn(p))), truths, fams)
            out[kind][a] = (float(S_['map5']), np.array([M_[q_] for q_ in QORD]))
    return out


for phi, seed in LAB:
    P, ntr = fold_scores(phi, seed)
    if phi == 1.0:
        G['a_S_star_exact'] = bool(np.array_equal(P['S'], sc['S_plus_X'].to_numpy()))
        if not G['a_S_star_exact']:
            dump(); sys.exit('ABORT: gate a (S* replay)')
        fr0 = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', P['S'])))
        assert fr0.select(K).equals(d.select(K))
        t1 = align(fr0, pl.read_parquet(CVS + 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
        dec = lambda p: decode(fr0.with_columns(pl.Series('score', p)), t1)
    m = aps(P)
    r_ = R133[(phi, seed)]
    n_mean = float(np.mean([v for f in FAM for v in ntr[f]]))
    ok = bool(abs(m['raw']['S'][0] - r_['raw']['S']['map5']) < 1e-12 and abs(m['dec']['S'][0] - r_['dec']['S']['map5']) < 1e-12 and abs(n_mean - r_['n_mean']) < 1e-9)
    G['b_S_phi_replays'][f'{phi}_{seed}'] = ok
    APS[(phi, seed)] = {kind: {a: m[kind][a][1] for a in m[kind]} for kind in m}
    RES['runs'].append({'phi': phi, 'seed': seed, 'n_mean': n_mean, 'n_fam': {f: float(np.mean(ntr[f])) for f in FAM},
                        'map5': {kind: {a: m[kind][a][0] for a in m[kind]} for kind in m}})
    dump(); log(phi, seed, ok, round(n_mean, 1), {kind: {a: round(m[kind][a][0], 4) for a in m[kind]} for kind in m})
    if not ok:
        sys.exit('ABORT: gate b (S_phi replay)')


NMEAN = np.array([r_['n_mean'] for r_ in RES['runs']])
NFAM = {f: np.array([r_['n_fam'][f] for r_ in RES['runs']]) for f in FAM}
RI = {lab: i for i, lab in enumerate(LAB)}
V = {kind: {a: np.stack([APS[lab][kind][a] for lab in LAB]) for a in ARMS} for kind in ('raw', 'dec')}
np.savez(OUT + 'query_ap_step140.npz', qord=np.array(QORD), labels=np.array(LAB), **{kind + '_' + a: V[kind][a] for kind in V for a in V[kind]})
SER = {kind: dict([('S', V[kind]['S'])] + [(a + '-S', V[kind][a] - V[kind]['S']) for a in ARMS if a != 'S']) for kind in V}
TI = np.unique(np.array([ptab[q_] for q_ in QORD]), return_inverse=True)[1]
NT = int(TI.max()) + 1
fitj = json.load(open(CVS + 'outputs_stage2_r382_20260911/fit.json', encoding='utf-8'))['stage2']
NPROD = {f: int(fitj['r382/' + f]['training_pairs']) for f in FAM}
DF = {f: float(np.log2(NPROD[f] / RES['runs'][0]['n_fam'][f])) for f in FAM}
NQ = {f: int((famq == f).sum()) for f in FAM}
DBAR = float(sum(NQ[f] * DF[f] for f in FAM) / sum(NQ.values()))
assert abs(DBAR - 1.507) < 5e-4
NPRODM = float(NMEAN[0] * 2 ** DBAR)


def slope(Y, w, sel, nm=NMEAN):
    m = Y @ w / w.sum()
    ii = {x: [RI[(x, s)] for s in sel[x]] for x in FIT}
    xs = [np.log2(nm[0])] + [np.log2(nm[ii[x]].mean()) for x in FIT]
    ys = [m[0]] + [m[ii[x]].mean() for x in FIT]
    return float(np.polyfit(xs, ys, 1)[0]), float(m[0])


def invfit(Y, w, sel, nm=NMEAN, nprod=NPRODM):
    m = Y @ w / w.sum()
    ii = {x: [RI[(x, s)] for s in sel[x]] for x in PHIS}
    xs = [1 / nm[0]] + [1 / nm[ii[x]].mean() for x in PHIS]
    ys = [m[0]] + [m[ii[x]].mean() for x in PHIS]
    b, a = np.polyfit(xs, ys, 1)
    return float(a + b / nprod), float(a), float(-b)


ALL5, ONE = {x: list(range(5)) for x in PHIS}, np.ones(len(QORD))
rng, rng2, rng3 = np.random.default_rng(1330), np.random.default_rng(1331), np.random.default_rng(1390)
DRAWS = [(np.bincount(rng.integers(0, NT, NT), minlength=NT)[TI].astype(float), {x: rng.integers(0, 5, 5) for x in FIT}) for _ in range(2000)]
SDRAWS = [{x: rng2.integers(0, 5, 5) for x in FIT} for _ in range(2000)]
IDRAWS = [(np.bincount(rng3.integers(0, NT, NT), minlength=NT)[TI].astype(float), {x: rng3.integers(0, 5, 5) for x in PHIS}) for _ in range(2000)]
qq = lambda a_: [float(np.quantile(a_, .05)), float(np.quantile(a_, .95))]


def readout(Y, dl):
    s0, d1 = slope(Y, ONE, ALL5)
    bj = np.array([slope(Y, w, sel) for w, sel in DRAWS])
    bs = np.array([slope(Y, ONE, sel)[0] for sel in SDRAWS])
    pi, g0, p0 = invfit(Y, ONE, ALL5)
    bi = np.array([invfit(Y, w, sel)[0] for w, sel in IDRAWS])
    pm = Y.mean(axis=1)
    e = {'slope': s0, 'at_phi1': d1, 'slope_q05_q95_joint': qq(bj[:, 0]), 'slope_q05_q95_seeds': qq(bs),
         'P_log': d1 * dl + s0 * DBAR, 'P_log_q05_q95': qq(bj[:, 1] * dl + bj[:, 0] * DBAR),
         'P_inv': pi, 'inv_g': g0, 'inv_p': p0, 'P_inv_q05_q95': qq(bi),
         'points': [{'phi': x, 'n_mean': float(NMEAN[[RI[(x, s)] for s in SEEDS]].mean()), 'mean': float(pm[[RI[(x, s)] for s in SEEDS]].mean()),
                     'sd': float(pm[[RI[(x, s)] for s in SEEDS]].std(ddof=1))} for x in PHIS], 'by_family': {}}
    for f in FAM:
        wf = (famq == f).astype(float)
        sf, f1 = slope(Y, wf, ALL5, NFAM[f])
        e['by_family'][f] = {'slope': sf, 'at_phi1': f1, 'P_log': f1 * dl + sf * DF[f], 'P_inv': invfit(Y, wf, ALL5, NFAM[f], NPROD[f])[0]}
    return e


rule = lambda e: 'GAINS_AT_SCALE' if (e['P_inv_q05_q95'][0] > 0 and e['P_inv'] >= .005 and e['P_log'] >= .005) else ('NO_GAIN' if e['P_inv_q05_q95'][1] < .005 else 'INCONCLUSIVE')
r4 = lambda v_: round(v_, 4) if isinstance(v_, float) else [round(u, 4) for u in v_]
CUR = {kind: {nm_: readout(Y, nm_ != 'S') for nm_, Y in SER[kind].items()} for kind in SER}
for kind in CUR:
    log('curve', kind, {nm_: {k_: r4(e[k_]) for k_ in ('slope', 'at_phi1', 'P_log', 'P_log_q05_q95', 'P_inv', 'P_inv_q05_q95')} for nm_, e in CUR[kind].items()})
reading = rule(CUR['dec']['C2-S'])
descriptive = {kind + '_' + nm_: rule(CUR[kind][nm_]) for kind in CUR for nm_ in CUR[kind] if nm_ != 'S' and (kind, nm_) != ('dec', 'C2-S')}
G['c_protected_unchanged'] = bool(ac.check_protected() == prot0)
RES.update(curve=CUR, reading=reading, descriptive_readings=descriptive, production_training_queries=NPROD, doublings_by_family=DF, dbar=DBAR,
           n_prod_pooled=NPRODM, runtime_s=round(time.time() - T0))
dump()
e = CUR['dec']['C2-S']
log('READING', reading, {k_: r4(e[k_]) for k_ in ('slope', 'slope_q05_q95_joint', 'at_phi1', 'P_log', 'P_log_q05_q95', 'P_inv', 'inv_g', 'inv_p', 'P_inv_q05_q95')})
log('points C2-S dec', [(p_['phi'], round(p_['mean'], 4)) for p_ in e['points']], 'by_family', {f: {k_: round(v_, 4) for k_, v_ in e['by_family'][f].items()} for f in FAM})
log('descriptive', descriptive, 'gates', {k_: v_ for k_, v_ in G.items() if k_ != 'b_S_phi_replays'}, all(G['b_S_phi_replays'].values()))
