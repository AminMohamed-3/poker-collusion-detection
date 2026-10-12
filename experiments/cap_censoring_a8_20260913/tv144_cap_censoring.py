"""STEP144 (poker-1b, formerly poker-a8): cap-aware censoring of ambiguous negatives in S* training, scored with the v1 views and the typed decoder.
Design from poker-11's unrun STEP139 draft (credited). Registration: PREDECLARE_step144_cap_censoring.md. Loader, folds, views and pads copied
read-only from tv138_v1_scale_curve.py. Audit look only: no held read, nothing kept."""
REG_SHA = '142bc00ea1c099f17ef6fe5f4d76d1351b18806830078c9b3db72514955d87dc'
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
from sklearn.metrics import log_loss
import audit_common as ac
from evidence_aug.frame import context
from evidence_aug.step104_C_typed_decoder_build import chronology, decode, align
from evidence_aug.step65_structural_evidence import summarize
CVS, CR = ROOT + 'Claude Version/', ROOT + 'Codex Resume/'
OUT = CVS + 'experiments/cap_censoring_a8_20260913/'
E124 = CVS + 'experiments/view_averaging_incumbent_C_20260913/'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
REG = sha(OUT + 'PREDECLARE_step144_cap_censoring.md')
assert REG == REG_SHA, 'registration changed'
assert sha(E124 + 'audit_v1_scores.parquet') == '1f157a9d15ccfb51030c46bf9969004fc24d0f27faef63e8ec480c2b3576525a'
K, CI, FAM = ['pair_id', 'hand_id'], 'coordinated_isolation', ac.FAMILIES
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


# masks: label structure of each query's own development list (training-only; applied to training rows only)
ev = pl.read_csv(ROOT + 'data/development_evidence.csv').select(K + ['evidence_rank'])
A_ = d.select(K + ['chrono_rank', 'relevance']).join(ev, on=K, how='left', validate='1:1')
qp = sorted(set(d['pair_id'].unique().to_list()))
assert ev.filter(pl.col('pair_id').is_in(qp)).height == int(A_['evidence_rank'].is_not_null().sum()) == int(d['relevance'].sum())
LS = A_.filter(pl.col('evidence_rank').is_not_null()).sort('pair_id', 'evidence_rank').group_by('pair_id', maintain_order=True).agg(
    pl.len().alias('L'), pl.col('chrono_rank').last().alias('t_cut'), (pl.col('chrono_rank').diff() < 0).sum().alias('nd'))
QM = Q.select(K + ['relevance', 'chrono_rank']).join(LS, on='pair_id', how='left', validate='m:1')
assert QM.select(K).equals(Q.select(K)) and QM['L'].null_count() == 0
AMB = ((QM['relevance'] == 0) & (QM['L'] == 5) & (QM['chrono_rank'] > QM['t_cut'])).to_numpy()
AMBD = AMB & (QM['nd'] >= 1).to_numpy()
famQ, pidQ, relQ, capQ = Q['behavior_family'].to_numpy(), Q['pair_id'].to_numpy(), Q['relevance'].to_numpy(), (QM['L'] == 5).to_numpy()
cnt = lambda m: {f: int(m[famQ == f].sum()) for f in FAM}
assert cnt(AMB) == {CI: 309, 'directed_transfer': 573, 'soft_play': 320} and cnt(AMBD) == {CI: 37, 'directed_transfer': 432, 'soft_play': 293}, (cnt(AMB), cnt(AMBD))
R0 = np.zeros(len(pidQ), bool)
for qi_, q_ in enumerate(qp):
    rows = np.flatnonzero(pidQ == q_)
    n_ = int(AMB[rows].sum())
    if capQ[rows[0]] and n_:
        R0[np.random.default_rng([1392, qi_]).choice(rows[relQ[rows] == 0], n_, replace=False)] = True
assert int(R0.sum()) == int(AMB.sum())
POOL = d.select(K + ['relevance', 'chrono_rank', 'oof', 'oof_rank'] + src).with_columns(pl.col('oof_rank').le(20).alias('top0'))
FOLDS = {}
for f in FAM:
    ix = np.flatnonzero((Q['behavior_family'] == f).to_numpy())
    q = Q[ix]; y = q['relevance'].to_numpy().astype(int); g = q['table_id'].to_numpy()
    FOLDS[f] = (ix, q['pair_id'].to_numpy(), y, g, q.select(columns(f)).to_numpy().astype(float), list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), y, g)))
truths = {k_[0]: set(g_['hand_id']) for k_, g_ in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
key = d.select('pair_id', 'hand_id', 'table_id')
QORD = sorted(summarize(key.with_columns(pl.Series('score', d['oof'].to_numpy())), truths, fams)[1])
assert QORD == qp and len(QORD) == 162
log('loaded', Q.height, 'AMB', cnt(AMB), 'AMBD', cnt(AMBD))


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
CFQ = Q['chrono_frac'].to_numpy().astype(float)


def matched(ix, tr, y, amb, rng):
    """Position-matched count control: non-AMB unlisted training rows matched on chrono_frac deciles of the fold's unlisted training rows."""
    U = tr[y[tr] == 0]
    Aq, N = U[amb[ix[U]]], U[~amb[ix[U]]]
    edges = np.quantile(CFQ[ix[U]], np.linspace(.1, .9, 9))
    dA, dN = np.searchsorted(edges, CFQ[ix[Aq]], side='right'), np.searchsorted(edges, CFQ[ix[N]], side='right')
    need_by, drop, carry = np.bincount(dA, minlength=10), [], 0
    for j in range(9, -1, -1):
        need, pool = int(need_by[j]) + carry, N[dN == j]
        take = min(need, len(pool))
        if take:
            drop.append(rng.choice(pool, take, replace=False))
        carry = need - take
    dr = np.concatenate(drop) if drop else np.array([], dtype=int)
    return dr, carry, (float(CFQ[ix[Aq]].mean()) if len(Aq) else None), (float(CFQ[ix[dr]].mean()) if len(dr) else None)


def fold_models(kind):
    M, info = {}, {}
    for fi, f in enumerate(FAM):
        ix, pid, y, g, z, folds = FOLDS[f]
        for k, (tr, te) in enumerate(folds):
            negs, w = int((y[tr] == 0).sum()), None
            keep = np.ones(len(tr), bool)
            if kind in ('K', 'KD', 'K0R'):
                m = {'K': AMB, 'KD': AMBD, 'K0R': R0}[kind][ix[tr]]
                assert not (m & (y[tr] == 1)).any()
                keep = ~m
                info[f'{f}|{k}'] = {'dropped': int(m.sum()), 'r': (negs - int(m.sum())) / negs}
            elif kind in ('K0P', 'KD0P'):
                dr, short, ma, md = matched(ix, tr, y, AMB if kind == 'K0P' else AMBD, np.random.default_rng([1393 if kind == 'K0P' else 1394, fi, k]))
                keep = ~np.isin(tr, dr)
                info[f'{f}|{k}'] = {'dropped': int(len(dr)), 'shortfall': int(short), 'amb_chrono_frac_mean': ma, 'dropped_chrono_frac_mean': md, 'r': (negs - len(dr)) / negs}
            elif kind == 'K50':
                w = np.where(AMB[ix[tr]], .5, 1.)
                info[f'{f}|{k}'] = {'weighted': int(AMB[ix[tr]].sum())}
            else:
                assert kind == 'S'
            M[(f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(z[tr][keep], y[tr][keep], sample_weight=w)
    return M, info


def tta_run(M, r=None):
    corr = (lambda p, f, k: p) if r is None else (lambda p, f, k: p * r[f'{f}|{k}'] / (p * r[f'{f}|{k}'] + 1 - p))
    P0 = d['oof'].to_numpy().copy() - 1e6
    for f in FAM:
        ix, pid, y, g, z, folds = FOLDS[f]
        for k, (tr, te) in enumerate(folds):
            P0[QIX[ix[te]]] = corr(M[(f, k)].predict_proba(z[te])[:, 1], f, k) + 10
    TH, PD = np.zeros_like(P0), np.zeros_like(P0)
    for i, zv in enumerate(ZV):
        p = P0.copy()
        for f in FAM:
            ix, pid, y, g, z, folds = FOLDS[f]
            for k, (tr, te) in enumerate(folds):
                p[QIX[ix[te]]] = corr(M[(f, k)].predict_proba(zv[f][te])[:, 1], f, k) + 10
        if i < NTHIN:
            TH += p
        else:
            PD += p
    return P0, (TH + PD + P0) / (len(ZV) + 1)


sc = pl.read_parquet(CVS + 'experiments/codex_ideas_sweep_C_20260913/scores-greedy-step1-prefix_strength.parquet').sort(K)
assert sc.select(K).equals(d.select(K))
v124 = pl.read_parquet(E124 + 'audit_v1_scores.parquet')
assert v124.select(K).equals(d.select(K))
MS, _ = fold_models('S')
S0, V1 = tta_run(MS)
G = {'a_S_star_exact': bool(np.array_equal(S0, sc['S_plus_X'].to_numpy())), 'b_v1_exact': bool(np.array_equal(V1, v124['v1'].to_numpy())),
     'a_mask_counts': {'AMB': cnt(AMB), 'AMBD': cnt(AMBD), 'K0R': cnt(R0)}}
fr0 = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', S0)))
assert fr0.select(K).equals(d.select(K))
t1 = align(fr0, pl.read_parquet(CVS + 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
dec = lambda p: decode(fr0.with_columns(pl.Series('score', p)), t1)
G['b_dec_v1_exact'] = bool(np.array_equal(dec(V1), v124['dec_v1'].to_numpy()))
RES = {'registration_sha256': REG, 'gates': G}
dump = lambda: json.dump(RES, open(OUT + 'results_step144.json', 'w', encoding='utf-8'), indent=1)
log('gates', G)
if not (G['a_S_star_exact'] and G['b_v1_exact'] and G['b_dec_v1_exact']):
    dump(); sys.exit('ABORT: replay gates')
SC, INFO = {'S_star': S0, 'v1': V1}, {}
for kind in ('K', 'KD', 'K0P', 'KD0P', 'K0R', 'K50'):
    M, INFO[kind] = fold_models(kind)
    SC[kind + '_noview'], SC[kind] = tta_run(M)
    if kind == 'K':
        SC['K_oc'] = tta_run(M, {fk: v['r'] for fk, v in INFO['K'].items()})[1]
    log('arm', kind)
DEC = {a: dec(p) for a, p in SC.items()}


def cmp(x, y, raw=False):
    src_ = SC if raw else DEC
    r_ = ac.evaluate(d, {'incumbent': src_[y], x: src_[x]})
    c_ = r_['comparisons'][x]
    return {'mean': float(c_['mean_delta']), 'ci95': [float(v) for v in c_['ci95']], 'by_family': c_['by_family'], 'halves': c_['halves'],
            'better_worse': [c_.get('pairs_better'), c_.get('pairs_worse')], 'map5_x': float(r_['sets'][x]['map5']), 'map5_y': float(r_['sets']['incumbent']['map5'])}


PAIRS = {'K-v1': ('K', 'v1'), 'K-K0P': ('K', 'K0P'), 'KD-v1': ('KD', 'v1'), 'KD-KD0P': ('KD', 'KD0P'), 'K0P-v1': ('K0P', 'v1'), 'KD0P-v1': ('KD0P', 'v1'),
         'K0R-v1': ('K0R', 'v1'), 'K50-v1': ('K50', 'v1'), 'K_oc-v1': ('K_oc', 'v1'), 'K_oc-K': ('K_oc', 'K'),
         'K_noview-S_star': ('K_noview', 'S_star'), 'KD_noview-S_star': ('KD_noview', 'S_star')}
C = {nm: cmp(*xy) for nm, xy in PAIRS.items()}
C['raw K-v1'], C['raw KD-v1'] = cmp('K', 'v1', True), cmp('KD', 'v1', True)
passesD = lambda c: c['mean'] > 0 and min(c['by_family'].values()) >= -.01 and min(c['halves'].values()) >= 0
K_kept = bool(passesD(C['K-v1']) and C['K-K0P']['mean'] > 0)
KD_kept = bool((not K_kept) and passesD(C['KD-v1']) and C['KD-KD0P']['mean'] > 0)
reading = 'K_KEPT' if K_kept else ('KD_KEPT' if KD_kept else 'NOT_KEPT')


def apq(p):
    S_, M_ = summarize(key.with_columns(pl.Series('score', p)), truths, fams)
    return np.array([M_[q_] for q_ in QORD])


APQ = {a: apq(DEC[a]) for a in DEC}
Lq, NDq = dict(LS.select('pair_id', 'L').iter_rows()), dict(LS.select('pair_id', 'nd').iter_rows())
grp = np.array(['uncapped' if Lq[q_] < 5 else ('capped_descent' if NDq[q_] >= 1 else 'capped_no_descent') for q_ in QORD])
GN = ('capped_descent', 'capped_no_descent', 'uncapped')
SPLIT = {'n': {gn: int((grp == gn).sum()) for gn in GN}}
for nm, (a, b) in (('K-v1', ('K', 'v1')), ('K-K0P', ('K', 'K0P')), ('KD-KD0P', ('KD', 'KD0P'))):
    SPLIT[nm] = {gn: float((APQ[a] - APQ[b])[grp == gn].mean()) for gn in GN}
yQ = d['relevance'].to_numpy()[QIX]
LL = {a: float(log_loss(yQ, np.clip(SC[a][QIX] - 10, 1e-6, 1 - 1e-6))) for a in SC}
G['c_protected_unchanged'] = bool(ac.check_protected() == prot0)
fr = pl.DataFrame({'pair_id': d['pair_id'], 'hand_id': d['hand_id'], **{a: SC[a] for a in SC}, 'dec_K': DEC['K'], 'dec_KD': DEC['KD']})
fr.write_parquet(OUT + 'audit_scores_step144.parquet')
np.savez(OUT + 'query_ap_step144.npz', qord=np.array(QORD), grp=grp, **{a: APQ[a] for a in APQ})
RES.update(reading=reading, K_kept=K_kept, KD_kept=KD_kept, comparisons=C, split_by_list_type=SPLIT, logloss_top20=LL, mask_info=INFO,
           map5_decoded={a: float(APQ[a].mean()) for a in APQ}, scores_sha256=sha(OUT + 'audit_scores_step144.parquet'), runtime_s=round(time.time() - T0))
dump()
log('READING', reading, {nm: (round(c['mean'], 4), [round(v, 4) for v in c['ci95']], {f[:4]: round(v, 4) for f, v in c['by_family'].items()},
                          {h: round(v, 4) for h, v in c['halves'].items()}) for nm, c in C.items()})
log('map5 decoded', {a: round(float(APQ[a].mean()), 4) for a in APQ}, 'split', SPLIT, 'logloss', {a: round(v, 4) for a, v in LL.items()}, 'gates', G)
