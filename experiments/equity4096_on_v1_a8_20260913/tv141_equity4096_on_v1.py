"""STEP141 (poker-1b, formerly poker-a8): 4096-runout preflop equity inside S* and v1, on the STEP133 learning curve.
Registration: PREDECLARE_step141_equity4096_on_v1.md. Loader, folds, views, pads, the v1 recipe, curve and bootstraps copied from
tv138_v1_scale_curve.py. Codex's 4096 inputs are read-only. Audit look only: no held read, nothing kept."""
REG_SHA = 'c3e16a9109867410228566fa796cebb503c2ec889e353438bff03b6b5c32afe2'
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
from evidence_aug.frame import context
from evidence_aug.step104_C_typed_decoder_build import chronology, decode, align
from evidence_aug.step65_structural_evidence import summarize
CVS, CR = ROOT + 'Claude Version/', ROOT + 'Codex Resume/'
OUT = CVS + 'experiments/equity4096_on_v1_a8_20260913/'
E133 = CVS + 'experiments/learning_curve_a8_20260913/'
E124 = CVS + 'experiments/view_averaging_incumbent_C_20260913/'
P4 = CR + 'research/preflop_equity_precision_20260912/'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
REG = sha(OUT + 'PREDECLARE_step141_equity4096_on_v1.md')
assert REG == REG_SHA, 'registration changed'
assert sha(E133 + 'step133_results.json') == '39ffd0415d151dcd2f3fd6febcf64a6eb117aa4043bcceebbc23ec00fb8fcba1'
assert sha(E124 + 'audit_v1_scores.parquet') == '1f157a9d15ccfb51030c46bf9969004fc24d0f27faef63e8ec480c2b3576525a'
assert sha(P4 + 'features4096.parquet') == '054c15fa51b09a259e8ecd3be4dc9e63080925a0193ffa32a73b6aee57039529'
assert sha(P4 + 'baseline256-cached.parquet') == '393c4da3e3d9827399ba0be1d1f07a43bf32d06c51c667d68411c6f3195f6206'
K, CI, FAM = ['pair_id', 'hand_id'], 'coordinated_isolation', ac.FAMILIES
PHIS, SEEDS, FIT = (.30, .45, .60, .75, .90), range(5), (.45, .60, .75, .90)
POOLF = ['chrono_rank', 'chrono_frac', 'n_hands', 'oof_rank']
T0 = time.time()
log = lambda *a: print(*a, f'{time.time() - T0:.0f}s', flush=True)
prot0 = ac.check_protected()
d, shared, rcols, saved = ac.load_current()
top = (d['oof_rank'] <= 20).to_numpy()
ctxc = [c for c in rcols if c.startswith('context_')]
src = [c[8:] for c in ctxc]
assert all(s in d.columns for s in src) and len(ctxc) == 136 and set(POOLF) <= set(shared)
cs = pl.read_parquet(CR + 'research/contestable_state_20260912/features.parquet').sort(K)
cs_cols = [c for c in cs.columns if c not in K]
assert len(cs_cols) == 29
pf = pl.read_parquet(CR + 'research/retrospective_strength_20260912/prefix.parquet').sort(K)
pf_cols = json.load(open(CR + 'research/retrospective_strength_20260912/freeze.json', encoding='utf-8'))['features']
assert pf.columns == K + pf_cols
Q = d.filter(pl.col('oof_rank') <= 20).join(cs, on=K, validate='1:1').join(pf, on=K, validate='1:1').sort(K)
QIX = np.flatnonzero(top)
assert d[QIX].select(K).equals(Q.select(K))
f4, b256 = pl.read_parquet(P4 + 'features4096.parquet').sort(K), pl.read_parquet(P4 + 'baseline256-cached.parquet').sort(K)
EQ = [c for c in f4.columns if c not in K]
assert len(EQ) == 24 and all(c in rcols and c not in shared + cs_cols + pf_cols + ctxc for c in EQ)
assert f4.select(K).equals(Q.select(K)) and b256.select(K).equals(Q.select(K)) and b256.columns == f4.columns
PARITY = max(float((Q[c].cast(pl.Float64) - b256[c].cast(pl.Float64)).abs().max()) for c in EQ)
assert PARITY == 0.0, 'eq_* values differ from the 256 baseline'
Q4 = Q.with_columns([f4[c].cast(Q[c].dtype).alias(c) for c in EQ])


def columns(f):
    return shared + (rcols if f != CI else []) + cs_cols + pf_cols


POOL = d.select(K + ['relevance', 'chrono_rank', 'oof', 'oof_rank'] + src).with_columns(pl.col('oof_rank').le(20).alias('top0'))
FOLDS, EQIX = {}, {}
for f in FAM:
    ix = np.flatnonzero((Q['behavior_family'] == f).to_numpy())
    q = Q[ix]; y = q['relevance'].to_numpy().astype(int); g = q['table_id'].to_numpy()
    zS, zE = q.select(columns(f)).to_numpy().astype(float), Q4[ix].select(columns(f)).to_numpy().astype(float)
    EQIX[f] = [columns(f).index(c) for c in EQ if c in columns(f)]
    other = np.setdiff1d(np.arange(zS.shape[1]), EQIX[f])
    assert np.array_equal(zS[:, other], zE[:, other]) and (len(EQIX[f]) == (0 if f == CI else 24))
    FOLDS[f] = (ix, q['pair_id'].to_numpy(), y, g, zS, zE, list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), y, g)))
truths = {k_[0]: set(g_['hand_id']) for k_, g_ in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
key = d.select('pair_id', 'hand_id', 'table_id')
QORD = sorted(summarize(key.with_columns(pl.Series('score', d['oof'].to_numpy())), truths, fams)[1])
assert len(QORD) == 162
famq = np.array([fams[q_] for q_ in QORD])
log('loaded', Q.height, 'eq parity', PARITY, {f: len(EQIX[f]) for f in FAM})


def views(u, m, seed):
    top0, parts = POOL['top0'].to_numpy(), []
    for c_ in range(m):
        r = np.random.default_rng([seed, c_]).random(POOL.height)
        parts.append(POOL.filter(pl.Series(~((r < u) & ~top0))).with_columns(pl.lit(c_).alias('copy')))
    a = pl.concat(parts).with_columns(pl.concat_str([pl.col('pair_id'), pl.col('copy').cast(pl.String)], separator='#').alias('pc'))
    a = a.with_columns((pl.col('chrono_rank').rank('ordinal').over('pc') - 1).cast(pl.Float64).alias('chrono_rank'),
                       pl.len().over('pc').cast(pl.Float64).alias('n_hands'),
                       pl.col('oof').rank('ordinal', descending=True).over('pc').cast(pl.Float64).alias('oof_rank'))
    a = a.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac'))
    a = context(a.rename({'pair_id': 'true_pair'}).rename({'pc': 'pair_id'}), src).drop('pair_id').rename({'true_pair': 'pair_id'})
    a = a.filter(pl.col('top0')).select(K + ['copy'] + POOLF + ctxc)
    keep = [c for c in Q.columns if c not in POOLF + ctxc]
    a = a.join(Q.select(keep), on=K, how='left', validate='m:1')
    out = []
    for c_ in range(m):
        v = a.filter(pl.col('copy') == c_).sort(K); assert v.select(K).equals(Q.select(K)); out.append(v)
    return out



CMAX = float(POOL['chrono_rank'].max()) + 1


def pads(u, m, seed, train):
    rel, top0, parts = POOL['relevance'].to_numpy(), POOL['top0'].to_numpy(), []
    for c_ in range(m):
        r = np.random.default_rng([seed, c_]).random(POOL.height)
        pick = (r < u) & ~top0
        if train:
            pick = pick & (rel == 0)
        dupl = POOL.filter(pl.Series(pick)).with_columns((pl.col('chrono_rank') + CMAX * 10).alias('chrono_rank'), pl.lit(False).alias('top0'))
        parts.append(pl.concat([POOL, dupl]).with_columns(pl.lit(c_).alias('copy')))
    a = pl.concat(parts).with_columns(pl.concat_str([pl.col('pair_id'), pl.col('copy').cast(pl.String)], separator='#').alias('pc'))
    a = a.with_columns((pl.col('chrono_rank').rank('ordinal').over('pc') - 1).cast(pl.Float64).alias('chrono_rank'),
                       pl.len().over('pc').cast(pl.Float64).alias('n_hands'),
                       pl.col('oof').rank('ordinal', descending=True).over('pc').cast(pl.Float64).alias('oof_rank'))
    a = a.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac'))
    a = context(a.rename({'pair_id': 'true_pair'}).rename({'pc': 'pair_id'}), src).drop('pair_id').rename({'true_pair': 'pair_id'})
    a = a.filter(pl.col('top0')).select(K + ['copy'] + POOLF + ctxc)
    keep = [c for c in Q.columns if c not in POOLF + ctxc]
    a = a.join(Q.select(keep), on=K, how='left', validate='m:1')
    out = []
    for c_ in range(m):
        v = a.filter(pl.col('copy') == c_).sort(K); assert v.select(K).equals(Q.select(K))
        assert np.array_equal(v['oof_rank'].to_numpy(), Q['oof_rank'].to_numpy()); out.append(v)
    return out


v0 = pads(0., 1, 0, False)[0]
assert max(float((v0[c].cast(pl.Float64) - Q[c].cast(pl.Float64)).abs().max()) for c in POOLF + ctxc) < 1e-5
print('u=0 replay ok', round(time.time() - T0), flush=True)


def zmat(v):
    return {f: v[FOLDS[f][0]].select(columns(f)).to_numpy().astype(float) for f in FAM}


ZV = []
for s in (1241, 2241):
    ZV += [zmat(v) for v in views(.5, 10, s)]
NTHIN = len(ZV)
for s in (1242, 2242):
    ZV += [zmat(v) for v in pads(.5, 10, s, False)]
assert (NTHIN, len(ZV)) == (20, 40)
log('views built', len(ZV))
ZVE = []
for zv in ZV:
    e_ = {}
    for f in FAM:
        zS, zE = FOLDS[f][4], FOLDS[f][5]
        assert np.array_equal(zv[f][:, EQIX[f]], zS[:, EQIX[f]]), 'view raw eq_* differ from Q'
        if EQIX[f]:
            m_ = zv[f].copy(); m_[:, EQIX[f]] = zE[:, EQIX[f]]; e_[f] = m_
        else:
            e_[f] = zv[f]
    ZVE.append(e_)
log('E views built', len(ZVE))


def fold_models(phi, seed):
    M, ntr = {}, {f: [] for f in FAM}
    for fi, f in enumerate(FAM):
        ix, pid, y, g, zS, zE, folds = FOLDS[f]
        for k, (tr, te) in enumerate(folds):
            if phi < 1:
                tabs = np.unique(g[tr])
                keep = np.random.default_rng([133, seed, fi, k]).choice(tabs, max(2, int(round(phi * len(tabs)))), replace=False)
                tr = tr[np.isin(g[tr], keep)]
            ntr[f].append(int(len(np.unique(pid[tr]))))
            M[('S', f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(zS[tr], y[tr])
            M[('E', f, k)] = M[('S', f, k)] if f == CI else lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(zE[tr], y[tr])
    return M, ntr


def tta_run(M, arm):
    ZVA = ZV if arm == 'S' else ZVE
    P0 = d['oof'].to_numpy().copy() - 1e6
    for f in FAM:
        ix, pid, y, g, zS, zE, folds = FOLDS[f]
        z = zS if arm == 'S' else zE
        for k, (tr, te) in enumerate(folds):
            P0[QIX[ix[te]]] = M[(arm, f, k)].predict_proba(z[te])[:, 1] + 10
    TH, PD = np.zeros_like(P0), np.zeros_like(P0)
    for i, zv in enumerate(ZVA):
        p = P0.copy()
        for f in FAM:
            ix, pid, y, g, zS, zE, folds = FOLDS[f]
            for k, (tr, te) in enumerate(folds):
                p[QIX[ix[te]]] = M[(arm, f, k)].predict_proba(zv[f][te])[:, 1] + 10
        if i < NTHIN:
            TH += p
        else:
            PD += p
    return P0, (TH + PD + P0) / (len(ZVA) + 1)


sc = pl.read_parquet(CVS + 'experiments/codex_ideas_sweep_C_20260913/scores-greedy-step1-prefix_strength.parquet').sort(K)
assert sc.select(K).equals(d.select(K))
v124 = pl.read_parquet(E124 + 'audit_v1_scores.parquet')
assert v124.select(K).equals(d.select(K))
M1, N1 = fold_models(1.0, 0)
S1, V1 = tta_run(M1, 'S')
E1, EV1 = tta_run(M1, 'E')
CIR = QIX[FOLDS[CI][0]]
G = {'a_S_star_exact': bool(np.array_equal(S1, sc['S_plus_X'].to_numpy())), 'a_v1_exact': bool(np.array_equal(V1, v124['v1'].to_numpy())),
     'c_eq_parity_maxabs': PARITY, 'c_CI_identical': bool(np.array_equal(S1[CIR], E1[CIR]) and np.array_equal(V1[CIR], EV1[CIR])),
     'c_E_changes_scores': bool(not np.array_equal(S1, E1)), 'b_S_phi_replays': {}}
RES = {'registration_sha256': REG, 'gates': G, 'runs': []}
dump = lambda: json.dump(RES, open(OUT + 'results_step141.json', 'w', encoding='utf-8'), indent=1)
log('gates a c', G)
if not (G['a_S_star_exact'] and G['a_v1_exact'] and G['c_CI_identical'] and G['c_E_changes_scores']):
    dump(); sys.exit('ABORT: gate a/c')
fr0 = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', S1)))
assert fr0.select(K).equals(d.select(K))
t1 = align(fr0, pl.read_parquet(CVS + 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
dec = lambda p: decode(fr0.with_columns(pl.Series('score', p)), t1)
G['a_dec_v1_exact'] = bool(np.array_equal(dec(V1), v124['dec_v1'].to_numpy()))
if not G['a_dec_v1_exact']:
    dump(); sys.exit('ABORT: gate a (decoded v1)')
J133 = json.load(open(E133 + 'step133_results.json', encoding='utf-8'))
R133 = {(r_['phi'], r_['seed']): r_ for r_ in J133['runs']}
R133[(1.0, 0)] = J133['full']
APS = {}


def aps(P):
    out = {}
    for kind, fn in (('raw', lambda p: p), ('dec', dec)):
        out[kind] = {}
        for a, p in P.items():
            S_, M_ = summarize(key.with_columns(pl.Series('score', fn(p))), truths, fams)
            out[kind][a] = (float(S_['map5']), np.array([M_[q_] for q_ in QORD]))
    return out


LAB = [(1.0, 0)] + [(x, s) for x in PHIS for s in SEEDS]
for phi, seed in LAB:
    if phi == 1.0:
        ntr, P = N1, {'S': S1, 'v1': V1, 'E': E1, 'Ev1': EV1}
    else:
        M, ntr = fold_models(phi, seed)
        a0, a1 = tta_run(M, 'S'); b0, b1 = tta_run(M, 'E')
        P = {'S': a0, 'v1': a1, 'E': b0, 'Ev1': b1}
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


RI = {lab: i for i, lab in enumerate(LAB)}
NMEAN = np.array([r_['n_mean'] for r_ in RES['runs']])
NFAM = {f: np.array([r_['n_fam'][f] for r_ in RES['runs']]) for f in FAM}
V = {kind: {a: np.stack([APS[lab][kind][a] for lab in LAB]) for a in ('S', 'v1', 'E', 'Ev1')} for kind in ('raw', 'dec')}
np.savez(OUT + 'query_ap_step141.npz', qord=np.array(QORD), labels=np.array(LAB), **{kind + '_' + a: V[kind][a] for kind in V for a in V[kind]})
SER = {kind: {'Ev1-v1': V[kind]['Ev1'] - V[kind]['v1'], 'E-S': V[kind]['E'] - V[kind]['S'], 'v1-S': V[kind]['v1'] - V[kind]['S']} for kind in V}
TI = np.unique(np.array([ptab[q_] for q_ in QORD]), return_inverse=True)[1]
NT = int(TI.max()) + 1
fitj = json.load(open(CVS + 'outputs_stage2_r382_20260911/fit.json', encoding='utf-8'))['stage2']
NPROD = {f: int(fitj['r382/' + f]['training_pairs']) for f in FAM}
DF = {f: float(np.log2(NPROD[f] / RES['runs'][0]['n_fam'][f])) for f in FAM}
NQ = {f: int((famq == f).sum()) for f in FAM}
DBAR = float(sum(NQ[f] * DF[f] for f in FAM) / sum(NQ.values()))
assert abs(DBAR - 1.507) < 5e-4


def slope(Y, w, sel, nm=NMEAN):
    m = Y @ w / w.sum()
    ii = {x: [RI[(x, s)] for s in sel[x]] for x in FIT}
    xs = [np.log2(nm[0])] + [np.log2(nm[ii[x]].mean()) for x in FIT]
    ys = [m[0]] + [m[ii[x]].mean() for x in FIT]
    return float(np.polyfit(xs, ys, 1)[0]), float(m[0])


ALL5, ONE = {x: list(range(5)) for x in PHIS}, np.ones(len(QORD))
rng, rng2, rngT = np.random.default_rng(1330), np.random.default_rng(1331), np.random.default_rng(1410)
DRAWS = [(np.bincount(rng.integers(0, NT, NT), minlength=NT)[TI].astype(float), {x: rng.integers(0, 5, 5) for x in FIT}) for _ in range(2000)]
SDRAWS = [{x: rng2.integers(0, 5, 5) for x in FIT} for _ in range(2000)]
TW = np.stack([np.bincount(rngT.integers(0, NT, NT), minlength=NT)[TI].astype(float) for _ in range(2000)])
qq = lambda a_: [float(np.quantile(a_, .05)), float(np.quantile(a_, .95))]
CUR = {}
for kind in SER:
    CUR[kind] = {}
    for nm_, Y in SER[kind].items():
        s0, d1 = slope(Y, ONE, ALL5)
        bj = np.array([slope(Y, w, sel) for w, sel in DRAWS])
        bs = np.array([slope(Y, ONE, sel)[0] for sel in SDRAWS])
        pm = Y.mean(axis=1)
        e = {'phi1': {'mean': float(Y[0].mean()), 'q05_q95': qq(TW @ Y[0] / TW.sum(axis=1)), 'by_family': {f: float(Y[0][famq == f].mean()) for f in FAM},
                      'better_worse': [int((Y[0] > 1e-12).sum()), int((Y[0] < -1e-12).sum())]},
             'slope': s0, 'slope_q05_q95_joint': qq(bj[:, 0]), 'slope_q05_q95_seeds': qq(bs),
             'proj_production': d1 + s0 * DBAR, 'proj_q05_q95': qq(bj[:, 1] + bj[:, 0] * DBAR),
             'points': [{'phi': x, 'n_mean': float(NMEAN[[RI[(x, s)] for s in SEEDS]].mean()), 'mean': float(pm[[RI[(x, s)] for s in SEEDS]].mean()),
                         'sd': float(pm[[RI[(x, s)] for s in SEEDS]].std(ddof=1))} for x in PHIS], 'by_family': {}}
        for f in FAM:
            sf, f1 = slope(Y, (famq == f).astype(float), ALL5, NFAM[f])
            e['by_family'][f] = {'slope': sf, 'at_phi1': f1, 'proj_production': f1 + sf * DF[f]}
        CUR[kind][nm_] = e
    log('curve', kind, {nm_: [round(e_['phi1']['mean'], 4), [round(u, 4) for u in e_['phi1']['q05_q95']], round(e_['slope'], 4),
                              round(e_['proj_production'], 4), [round(u, 4) for u in e_['proj_q05_q95']]] for nm_, e_ in CUR[kind].items()})
rule = lambda e_: 'NO_GAIN' if (e_['phi1']['q05_q95'][1] < .003 or e_['proj_q05_q95'][1] < 0) else (
    'GAIN' if (e_['phi1']['q05_q95'][0] > 0 and e_['proj_production'] >= .003) else 'INCONCLUSIVE')
reading = rule(CUR['dec']['Ev1-v1'])
descriptive = {kind + '_' + nm_: rule(CUR[kind][nm_]) for kind in CUR for nm_ in ('Ev1-v1', 'E-S') if (kind, nm_) != ('dec', 'Ev1-v1')}
G['d_protected_unchanged'] = bool(ac.check_protected() == prot0)
RES.update(curve=CUR, reading=reading, descriptive_readings=descriptive, production_training_queries=NPROD, doublings_by_family=DF, dbar=DBAR,
           runtime_s=round(time.time() - T0))
dump()
e = CUR['dec']['Ev1-v1']
log('READING', reading, 'phi1', e['phi1'], 'slope', round(e['slope'], 4), e['slope_q05_q95_joint'], 'P', round(e['proj_production'], 4), e['proj_q05_q95'])
log('points', [(p_['phi'], round(p_['mean'], 4)) for p_ in e['points']], 'by_family', {f: {k_: round(v_, 4) for k_, v_ in e['by_family'][f].items()} for f in FAM})
log('descriptive', descriptive, 'gates', {k_: v_ for k_, v_ in G.items() if k_ != 'b_S_phi_replays'}, all(G['b_S_phi_replays'].values()))
