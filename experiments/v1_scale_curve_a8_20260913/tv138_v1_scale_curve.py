"""STEP138 (poker-1b, formerly poker-a8): v1 view-averaging gain on the STEP133 learning curve, projected to production scale.
Registration: PREDECLARE_v1_scale_curve.md. Loader, folds, views, pads and the v1 recipe copied read-only from replay_v1_step124.py;
curve and bootstraps copied from tv133_curve.py. Audit look only: no held read, nothing kept."""
REG_SHA = '08c4ac1c2771c66a208358e51dabc82749d46363a7dfa7017d41c4d9aafedff5'
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
OUT = CVS + 'experiments/v1_scale_curve_a8_20260913/'
E133 = CVS + 'experiments/learning_curve_a8_20260913/'
E124 = CVS + 'experiments/view_averaging_incumbent_C_20260913/'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
REG = sha(OUT + 'PREDECLARE_v1_scale_curve.md')
assert REG == REG_SHA, 'registration changed'
assert sha(E133 + 'step133_results.json') == '39ffd0415d151dcd2f3fd6febcf64a6eb117aa4043bcceebbc23ec00fb8fcba1'
assert sha(E124 + 'audit_v1_scores.parquet') == '1f157a9d15ccfb51030c46bf9969004fc24d0f27faef63e8ec480c2b3576525a'
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


def columns(f):
    return shared + (rcols if f != CI else []) + cs_cols + pf_cols


POOL = d.select(K + ['relevance', 'chrono_rank', 'oof', 'oof_rank'] + src).with_columns(pl.col('oof_rank').le(20).alias('top0'))
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
log('loaded', Q.height)


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


def fold_models(phi, seed):
    M, ntr = {}, {f: [] for f in FAM}
    for fi, f in enumerate(FAM):
        ix, pid, y, g, z, folds = FOLDS[f]
        for k, (tr, te) in enumerate(folds):
            if phi < 1:
                tabs = np.unique(g[tr])
                keep = np.random.default_rng([133, seed, fi, k]).choice(tabs, max(2, int(round(phi * len(tabs)))), replace=False)
                tr = tr[np.isin(g[tr], keep)]
            ntr[f].append(int(len(np.unique(pid[tr]))))
            M[(f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(z[tr], y[tr])
    return M, ntr


def tta_run(M):
    P0 = d['oof'].to_numpy().copy() - 1e6
    for f in FAM:
        ix, pid, y, g, z, folds = FOLDS[f]
        for k, (tr, te) in enumerate(folds):
            P0[QIX[ix[te]]] = M[(f, k)].predict_proba(z[te])[:, 1] + 10
    TH, PD = np.zeros_like(P0), np.zeros_like(P0)
    for i, zv in enumerate(ZV):
        p = P0.copy()
        for f in FAM:
            ix, pid, y, g, z, folds = FOLDS[f]
            for k, (tr, te) in enumerate(folds):
                p[QIX[ix[te]]] = M[(f, k)].predict_proba(zv[f][te])[:, 1] + 10
        if i < NTHIN:
            TH += p
        else:
            PD += p
    return P0, (TH + PD + P0) / (len(ZV) + 1)


sc = pl.read_parquet(CVS + 'experiments/codex_ideas_sweep_C_20260913/scores-greedy-step1-prefix_strength.parquet').sort(K)
assert sc.select(K).equals(d.select(K))
v124 = pl.read_parquet(E124 + 'audit_v1_scores.parquet')
assert v124.select(K).equals(d.select(K))
M1, N1 = fold_models(1.0, 0)
S1, V1 = tta_run(M1)
G = {'a_S_star_exact': bool(np.array_equal(S1, sc['S_plus_X'].to_numpy())), 'b_v1_exact': bool(np.array_equal(V1, v124['v1'].to_numpy())),
     'b_v1_maxabs': float(np.abs(V1 - v124['v1'].to_numpy()).max()), 'c_S_phi_replays': {}}
RES = {'registration_sha256': REG, 'gates': G, 'runs': []}
dump = lambda: json.dump(RES, open(OUT + 'results_v1_scale_curve.json', 'w', encoding='utf-8'), indent=1)
log('gates a b', G)
if not (G['a_S_star_exact'] and G['b_v1_exact']):
    dump(); sys.exit('ABORT: gate a/b (S* or v1 replay)')
fr0 = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', S1)))
assert fr0.select(K).equals(d.select(K))
t1 = align(fr0, pl.read_parquet(CVS + 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
dec = lambda p: decode(fr0.with_columns(pl.Series('score', p)), t1)
G['b_dec_v1_exact'] = bool(np.array_equal(dec(V1), v124['dec_v1'].to_numpy()))
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


for phi, seed in [(1.0, 0)] + [(x, s) for x in PHIS for s in SEEDS]:
    if phi == 1.0:
        ntr, P0, PV = N1, S1, V1
    else:
        M, ntr = fold_models(phi, seed)
        P0, PV = tta_run(M)
    m = aps({'S': P0, 'v1': PV})
    r_ = R133[(phi, seed)]
    n_mean = float(np.mean([v for f in FAM for v in ntr[f]]))
    ok = bool(abs(m['raw']['S'][0] - r_['raw']['S']['map5']) < 1e-12 and abs(m['dec']['S'][0] - r_['dec']['S']['map5']) < 1e-12 and abs(n_mean - r_['n_mean']) < 1e-9)
    G['c_S_phi_replays'][f'{phi}_{seed}'] = ok
    APS[(phi, seed)] = {kind: {a: m[kind][a][1] for a in m[kind]} for kind in m}
    RES['runs'].append({'phi': phi, 'seed': seed, 'n_mean': n_mean, 'n_fam': {f: float(np.mean(ntr[f])) for f in FAM},
                        'map5': {kind: {a: m[kind][a][0] for a in m[kind]} for kind in m}})
    dump(); log(phi, seed, ok, round(n_mean, 1), {kind: {a: round(m[kind][a][0], 4) for a in m[kind]} for kind in m})
    if not ok:
        sys.exit('ABORT: gate c (S_phi replay)')


LAB = [(1.0, 0)] + [(x, s) for x in PHIS for s in SEEDS]
RI = {lab: i for i, lab in enumerate(LAB)}
NMEAN = np.array([r_['n_mean'] for r_ in RES['runs']])
NFAM = {f: np.array([r_['n_fam'][f] for r_ in RES['runs']]) for f in FAM}
V = {kind: {a: np.stack([APS[lab][kind][a] for lab in LAB]) for a in ('S', 'v1')} for kind in ('raw', 'dec')}
np.savez(OUT + 'query_ap_v1_scale_curve.npz', qord=np.array(QORD), labels=np.array(LAB), **{kind + '_' + a: V[kind][a] for kind in V for a in V[kind]})
SER = {kind: {'S': V[kind]['S'], 'v1-S': V[kind]['v1'] - V[kind]['S']} for kind in V}
TI = np.unique(np.array([ptab[q_] for q_ in QORD]), return_inverse=True)[1]
NT = int(TI.max()) + 1
fitj = json.load(open(CVS + 'outputs_stage2_r382_20260911/fit.json', encoding='utf-8'))['stage2']
NPROD = {f: int(fitj['r382/' + f]['training_pairs']) for f in FAM}
DF = {f: float(np.log2(NPROD[f] / RES['runs'][0]['n_fam'][f])) for f in FAM}
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
for kind in SER:
    CUR[kind] = {}
    for nm_, Y in SER[kind].items():
        dl = nm_ != 'S'
        s0, d1 = slope(Y, ONE, ALL5)
        bj = np.array([slope(Y, w, sel) for w, sel in DRAWS])
        bs = np.array([slope(Y, ONE, sel)[0] for sel in SDRAWS])
        pm = Y.mean(axis=1)
        e = {'slope': s0, 'at_phi1': d1, 'slope_q05_q95_joint': qq(bj[:, 0]), 'slope_q05_q95_seeds': qq(bs),
             'proj_production': d1 * dl + s0 * DBAR, 'proj_q05_q95': qq(bj[:, 1] * dl + bj[:, 0] * DBAR),
             'points': [{'phi': x, 'n_mean': float(NMEAN[[RI[(x, s)] for s in SEEDS]].mean()), 'mean': float(pm[[RI[(x, s)] for s in SEEDS]].mean()),
                         'sd': float(pm[[RI[(x, s)] for s in SEEDS]].std(ddof=1))} for x in PHIS], 'by_family': {}}
        e['composite'] = .2 * e['proj_production']
        for f in FAM:
            sf, f1 = slope(Y, (famq == f).astype(float), ALL5, NFAM[f])
            e['by_family'][f] = {'slope': sf, 'at_phi1': f1, 'proj_production': f1 * dl + sf * DF[f]}
        CUR[kind][nm_] = e
    log('curve', kind, {nm_: [round(CUR[kind][nm_]['slope'], 4), round(CUR[kind][nm_]['proj_production'], 4), CUR[kind][nm_]['proj_q05_q95']] for nm_ in CUR[kind]})
e = CUR['dec']['v1-S']
reading = 'SURVIVES' if (e['proj_q05_q95'][0] > 0 and e['proj_production'] >= .005) else ('SHRINKS' if e['proj_q05_q95'][1] < .005 else 'INCONCLUSIVE')
G['d_protected_unchanged'] = bool(ac.check_protected() == prot0)
RES.update(curve=CUR, reading=reading, production_training_queries=NPROD, doublings_by_family=DF, dbar=DBAR, runtime_s=round(time.time() - T0))
dump()
log('READING', reading, {k_: e[k_] for k_ in ('slope', 'at_phi1', 'slope_q05_q95_joint', 'proj_production', 'proj_q05_q95')}, 'points', [(p_['phi'], round(p_['mean'], 4)) for p_ in e['points']])
log('by_family', e['by_family'], 'gates', {k_: v_ for k_, v_ in G.items() if k_ != 'c_S_phi_replays'}, all(G['c_S_phi_replays'].values()))
