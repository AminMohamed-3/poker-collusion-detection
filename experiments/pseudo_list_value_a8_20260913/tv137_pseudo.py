"""STEP137 (poker-1b, formerly poker-a8): value of a pseudo-labelled evidence list, audit emulation.
Registration: PREDECLARE_step137_pseudo_list_value.md. Loader, folds and S* recipe copied read-only from tv133_curve.py
(itself from replay_v1_step124.py). Audit look only: no held pool, held truth or evaluation data; nothing kept."""
REG_SHA = '21c9a90f3d17d358d131a9f2954d6c1f69f17c26cb802d769dbe5907e76b2f07'
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
OUT = CVS + 'experiments/pseudo_list_value_a8_20260913/'
E133 = CVS + 'experiments/learning_curve_a8_20260913/'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
REG = sha(OUT + 'PREDECLARE_step137_pseudo_list_value.md')
assert REG == REG_SHA, 'registration changed'
assert sha(E133 + 'step133_results.json') == '39ffd0415d151dcd2f3fd6febcf64a6eb117aa4043bcceebbc23ec00fb8fcba1'
K, CI, FAM = ['pair_id', 'hand_id'], 'coordinated_isolation', ac.FAMILIES
PHIS, SEEDS, M_EVAL, N_REAL, SLOPE = (.30, .45, .60), range(5), 564, 372, .0340
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
FOLDS = {}
for f in FAM:
    ix = np.flatnonzero((Q['behavior_family'] == f).to_numpy())
    q = Q[ix]; y = q['relevance'].to_numpy().astype(int); g = q['table_id'].to_numpy()
    cols = shared + (rcols if f != CI else []) + cs_cols + pf_cols
    FOLDS[f] = (ix, q['pair_id'].to_numpy(), y, g, q.select(cols).to_numpy().astype(float), list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), y, g)))
truths = {k_[0]: set(g_['hand_id']) for k_, g_ in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
pools = sorted(set(ptab.values()))
key = d.select('pair_id', 'hand_id', 'table_id')
QORD = sorted(summarize(key.with_columns(pl.Series('score', d['oof'].to_numpy())), truths, fams)[1])
assert len(QORD) == 162
famq = np.array([fams[q_] for q_ in QORD])
qt = np.array([pools.index(ptab[q_]) for q_ in QORD])
NT = len(pools)
log('loaded', Q.height, NT)
ARMS = ('S', 'PLd', 'PLr', 'PLs', 'PLx')
fit = lambda X, y_, w_=None: lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(X, y_, sample_weight=w_)


def top5(score, pid_):
    lab = np.zeros(len(score), dtype=int)
    for p_ in np.unique(pid_):
        m_ = np.flatnonzero(pid_ == p_)
        lab[m_[np.argsort(-score[m_], kind='stable')[:5]]] = 1
    return lab


def run137(phi, seed):
    P = {a: d['oof'].to_numpy().copy() - 1e6 for a in ARMS}
    info = {f: {'kept_q': [], 'hidden_q': [], 'pl_precision_d': [], 'pl_precision_r': []} for f in FAM}
    for fi, f in enumerate(FAM):
        ix, pid, y, g, z, folds = FOLDS[f]
        for k, (tr, te) in enumerate(folds):
            tabs = np.unique(g[tr])
            keep = np.random.default_rng([133, seed, fi, k]).choice(tabs, max(2, int(round(phi * len(tabs)))), replace=False)
            kt, ht = tr[np.isin(g[tr], keep)], tr[~np.isin(g[tr], keep)]
            assert len(np.intersect1d(kt, ht)) == 0 and len(kt) + len(ht) == len(tr) and not np.isin(ht, te).any() and len(ht) > 0
            w = QIX[ix[te]]
            mS = fit(z[kt], y[kt])
            P['S'][w] = mS.predict_proba(z[te])[:, 1] + 10
            ph = mS.predict_proba(z[ht])[:, 1]
            v = np.full(d.height, -1e6)
            v[QIX[ix[ht]]] = ph + 10
            dh = dec(v)[QIX[ix[ht]]]
            lab = {'PLd': top5(dh, pid[ht]), 'PLr': top5(ph, pid[ht]),
                   'PLx': top5(np.random.default_rng([1371, seed, fi, k]).random(len(ht)), pid[ht])}
            Xs = np.vstack([z[kt], z[ht]])
            for a in ('PLd', 'PLr', 'PLx'):
                P[a][w] = fit(Xs, np.r_[y[kt], lab[a]]).predict_proba(z[te])[:, 1] + 10
            P['PLs'][w] = fit(np.vstack([z[kt], z[ht], z[ht]]), np.r_[y[kt], np.ones(len(ht), dtype=int), np.zeros(len(ht), dtype=int)],
                              np.r_[np.ones(len(kt)), ph, 1 - ph]).predict_proba(z[te])[:, 1] + 10
            info[f]['kept_q'].append(int(len(np.unique(pid[kt]))))
            info[f]['hidden_q'].append(int(len(np.unique(pid[ht]))))
            for a, kk in (('PLd', 'pl_precision_d'), ('PLr', 'pl_precision_r')):
                info[f][kk].append(float(y[ht][lab[a] == 1].mean()))
    return P, info


def thin(seed, which):
    P = d['oof'].to_numpy().copy() - 1e6
    for fi, f in enumerate(FAM):
        ix, pid, y, g, z, folds = FOLDS[f]
        for k, (tr, te) in enumerate(folds):
            r = np.random.default_rng([1372, seed, fi, k]).random(len(tr))
            t2 = tr[~((r < .5) & (y[tr] == (0 if which == 'NEG' else 1)))]
            P[QIX[ix[te]]] = fit(z[t2], y[t2]).predict_proba(z[te])[:, 1] + 10
    return P


sc = pl.read_parquet(CVS + 'experiments/codex_ideas_sweep_C_20260913/scores-greedy-step1-prefix_strength.parquet').sort(K)
assert sc.select(K).equals(d.select(K))
SS = d['oof'].to_numpy().copy() - 1e6
for fi, f in enumerate(FAM):
    ix, pid, y, g, z, folds = FOLDS[f]
    for k, (tr, te) in enumerate(folds):
        SS[QIX[ix[te]]] = fit(z[tr], y[tr]).predict_proba(z[te])[:, 1] + 10
G = {'a_S_star_exact': bool(np.array_equal(SS, sc['S_plus_X'].to_numpy())), 'b_S_phi_replays': {}}
RES = {'registration_sha256': REG, 'gates': G, 'runs': {}}
dump = lambda: json.dump(RES, open(OUT + 'step137_results.json', 'w', encoding='utf-8'), indent=1)
log('gate a', G)
if not G['a_S_star_exact']:
    dump(); sys.exit('ABORT: gate a (S* replay)')
fr0 = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', SS)))
assert fr0.select(K).equals(d.select(K))
t1 = align(fr0, pl.read_parquet(CVS + 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
dec = lambda p: decode(fr0.with_columns(pl.Series('score', p)), t1)
R133 = {(r_['phi'], r_['seed']): r_ for r_ in json.load(open(E133 + 'step133_results.json', encoding='utf-8'))['runs']}
APS = {}


def mapq(P):
    out = {}
    for kind, fn in (('raw', lambda p: p), ('dec', dec)):
        for a, p in P.items():
            S_, M_ = summarize(key.with_columns(pl.Series('score', fn(p))), truths, fams)
            out[(kind, a)] = (float(S_['map5']), np.array([M_[q_] for q_ in QORD]))
    return out


S1 = mapq({'S1': SS})
RES['S1'] = {kk[0]: v[0] for kk, v in S1.items()}
for phi in PHIS:
    for seed in SEEDS:
        P, info = run137(phi, seed)
        m = mapq(P)
        r_ = R133[(phi, seed)]
        ok = bool(abs(m[('raw', 'S')][0] - r_['raw']['S']['map5']) < 1e-12 and abs(m[('dec', 'S')][0] - r_['dec']['S']['map5']) < 1e-12)
        G['b_S_phi_replays'][f'{phi}_{seed}'] = ok
        APS[(phi, seed)] = {kk: v[1] for kk, v in m.items()}
        RES['runs'][f'{phi}_{seed}'] = {'map5': {f'{kk[0]}_{kk[1]}': v[0] for kk, v in m.items()}, 'info': info}
        dump(); log(phi, seed, ok, {kk[1]: round(v[0], 4) for kk, v in m.items() if kk[0] == 'dec'})
        if not ok:
            sys.exit('ABORT: gate b (S_phi replay)')
for seed in SEEDS:
    m = mapq({'NEG': thin(seed, 'NEG'), 'POS': thin(seed, 'POS')})
    APS[('thin', seed)] = {kk: v[1] for kk, v in m.items()}
    RES['runs'][f'thin_{seed}'] = {f'{kk[0]}_{kk[1]}': v[0] for kk, v in m.items()}
    dump(); log('thin', seed, {f'{kk[0]}_{kk[1]}': round(v[0], 4) for kk, v in m.items()})


def readout(kind, arm, ctrl='PLx', nb=2000):
    s1 = S1[(kind, 'S1')][1]
    A = {phi: {a: np.stack([APS[(phi, s)][(kind, a)] for s in SEEDS]) for a in (arm, ctrl, 'S')} for phi in PHIS}

    def stat(cnt, sidx, fmask=None):
        wq = cnt[qt].astype(float) * (1.0 if fmask is None else fmask)
        tot = wq.sum()
        dpl, gg, dx = [], [], []
        for j, phi in enumerate(PHIS):
            mS, mA, mX = ((A[phi][a][sidx[j]] @ wq).mean() / tot for a in ('S', arm, ctrl))
            dpl.append(mA - mS); gg.append(s1 @ wq / tot - mS); dx.append(mA - mX)
        return float(np.mean(dpl)), float(np.sum(dpl) / np.sum(gg)), float(np.mean(dx)), dpl, gg
    one, alls = np.ones(NT), [np.arange(5)] * len(PHIS)
    pt = stat(one, alls)
    rng = np.random.default_rng(1370)
    B = np.array([stat(np.bincount(rng.integers(0, NT, NT), minlength=NT), [rng.integers(0, 5, 5) for _ in PHIS])[:3] for _ in range(nb)])
    q = lambda c, x: float(np.quantile(B[:, c], x))
    out = {'DPL': pt[0], 'ETA': pt[1], 'DX': pt[2], 'dPL_by_phi': [float(x) for x in pt[3]], 'G_by_phi': [float(x) for x in pt[4]],
           'DPL_q05_q95': [q(0, .05), q(0, .95)], 'ETA_q05_q95': [q(1, .05), q(1, .95)], 'DX_q05_q95': [q(2, .05), q(2, .95)]}
    out['by_family_descriptive'] = {f: dict(zip(('DPL', 'ETA', 'DX'), stat(one, alls, (famq == f).astype(float))[:3])) for f in FAM}
    return out


RD = {kind: {arm: readout(kind, arm) for arm in ('PLd', 'PLr', 'PLs', 'PLx')} for kind in ('dec', 'raw')}
PR = RD['dec']['PLd']
reading = 'PROMISING' if (PR['DPL_q05_q95'][0] > 0 and PR['ETA'] >= .25 and PR['DX_q05_q95'][0] > 0) else ('NULL' if PR['ETA_q05_q95'][1] < .10 else 'INCONCLUSIVE')
Dpl = lambda e: float(np.log2(1 + max(e, 0.) * M_EVAL / N_REAL))
proj = {'D_pl': Dpl(PR['ETA']), 'decoded_gain': SLOPE * Dpl(PR['ETA']), 'composite': .2 * SLOPE * Dpl(PR['ETA']),
        'decoded_gain_at_eta_q05_q95': [SLOPE * Dpl(PR['ETA_q05_q95'][0]), SLOPE * Dpl(PR['ETA_q05_q95'][1])]}
TH = {}
for kind in ('dec', 'raw'):
    s1 = S1[(kind, 'S1')][1]
    for a in ('NEG', 'POS'):
        M_ = np.stack([APS[('thin', s)][(kind, a)] for s in SEEDS])
        rng, bb = np.random.default_rng(1373), []
        for _ in range(2000):
            wq = np.bincount(rng.integers(0, NT, NT), minlength=NT)[qt].astype(float)
            si = rng.integers(0, 5, 5)
            bb.append((M_[si] @ wq).mean() / wq.sum() - s1 @ wq / wq.sum())
        TH[f'{kind}_{a}'] = {'minus_S1': float(M_.mean() - s1.mean()), 'q05_q95': [float(np.quantile(bb, .05)), float(np.quantile(bb, .95))]}
RES['pl_precision'] = {str(phi): {kk: float(np.mean([v for s in SEEDS for f in FAM for v in RES['runs'][f'{phi}_{s}']['info'][f][kk]]))
                                  for kk in ('pl_precision_d', 'pl_precision_r')} for phi in PHIS}
G['c_code_assertions'] = True
G['d_protected_unchanged'] = bool(ac.check_protected() == prot0)
RES.update(readouts=RD, reading=reading, projection=proj, thin_descriptive=TH, runtime_s=round(time.time() - T0))
dump()
np.savez_compressed(OUT + 'step137_query_ap.npz', qord=np.array(QORD), **{f'{kk[0]}_{kk[1]}_{k1[0]}_{k1[1]}': v for k1, dd in APS.items() for kk, v in dd.items()})
log('READING', reading, 'ETA', round(PR['ETA'], 3), PR['ETA_q05_q95'], 'DPL', round(PR['DPL'], 4), PR['DPL_q05_q95'], 'DX', round(PR['DX'], 4), PR['DX_q05_q95'])
log('projection', proj, 'protected', G['d_protected_unchanged'], 'precision', RES['pl_precision'])
for kind in RD:
    for arm, r in RD[kind].items():
        log(kind, arm, {k_: v for k_, v in r.items() if k_ != 'by_family_descriptive'})
        log('   fam', r['by_family_descriptive'])
log('thin', TH)
