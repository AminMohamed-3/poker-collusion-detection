"""Step 109 audit look [C] (scratch; training-only 162-query audit; no held read): label-exact pool-thinning augmentation of stage 2.

Premise (step 105, descriptive): an evidence list is the top five events by (type, time), so deleting an UNLISTED hand from a pair's
pool leaves the list unchanged. Augmented copies of each training query delete unlisted hands at random, recompute every
pool-dependent input exactly (chrono_rank, chrono_frac, n_hands, oof_rank, the 136 within-pair context_ fields; full-pool replay
max diff 0.0) and keep the labels. Models train on original + augmented rows of their training folds only.
Arms: tail (unlisted non-top-20 hands: query rows fixed, only pool features move), all (unlisted hands anywhere); dose u .3/.6;
2 copies. Controls: dup (2 unchanged copies), corrupt (delete hands regardless of label, u .3). Common random numbers (seed 109).
Readouts: stage-2 MAP@5 vs S*, decode vs decode(S*), top-20 log-loss. Every arm is an audit look and is recorded.
"""
import sys; sys.dont_write_bytecode = True
import os
for k in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '2'
CVS = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
CR = 'C:/Data/Projects/Money Making/work/poker/Codex Resume/'
SP = 'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/2271ec9c-faf5-4b66-b00c-0a0961e27c17/scratchpad/s109/'
sys.path.insert(0, CVS + 'src'); sys.path.insert(0, CR + 'src')
import json, time
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.model_selection import GroupKFold
import audit_common as ac
from evidence_aug.frame import context
from evidence_aug.step104_C_typed_decoder_build import chronology, decode, align
K = ['pair_id', 'hand_id']; CI = 'coordinated_isolation'; FAM = ac.FAMILIES
POOLF = ['chrono_rank', 'chrono_frac', 'n_hands', 'oof_rank']
T0 = time.time()
d, shared, rcols, saved = ac.load_current()
top = (d['oof_rank'] <= 20).to_numpy()
ctxc = [c for c in rcols if c.startswith('context_')]; src = [c[8:] for c in ctxc]
assert all(s in d.columns for s in src) and len(ctxc) == 136 and set(POOLF) <= set(shared)
cs = pl.read_parquet(CR + 'research/contestable_state_20260912/features.parquet').sort(K)
cs_cols = [c for c in cs.columns if c not in K]; assert len(cs_cols) == 29
pf = pl.read_parquet(CR + 'research/retrospective_strength_20260912/prefix.parquet').sort(K)
pf_cols = json.load(open(CR + 'research/retrospective_strength_20260912/freeze.json', encoding='utf-8'))['features']
assert pf.columns == K + pf_cols
expected = d.filter(pl.col('oof_rank') <= 20).select(K)
assert cs.select(K).equals(expected) and pf.select(K).equals(expected)
Q = d.filter(pl.col('oof_rank') <= 20).join(cs, on=K, validate='1:1').join(pf, on=K, validate='1:1').sort(K)
QIX = np.flatnonzero(top); assert d[QIX].select(K).equals(Q.select(K))


def columns(f, prefix=True):
    return shared + (rcols if f != CI else []) + cs_cols + (pf_cols if prefix else [])


POOL = d.select(K + ['relevance', 'chrono_rank', 'oof', 'oof_rank'] + src).with_columns(pl.col('oof_rank').le(20).alias('top0'))


def augment(mode, u, m, seed):
    """Copies of every audit query; rows = surviving original top-20 rows with pool features recomputed on the thinned pool."""
    rel, top0, parts = POOL['relevance'].to_numpy(), POOL['top0'].to_numpy(), []
    for c in range(m):
        r = np.random.default_rng([seed, c]).random(POOL.height)
        drop = {'dup': np.zeros(POOL.height, bool), 'tail': (r < u) & (rel == 0) & ~top0, 'all': (r < u) & (rel == 0), 'corrupt': r < u}[mode]
        parts.append(POOL.filter(pl.Series(~drop)).with_columns(pl.lit(c).alias('copy')))
    a = pl.concat(parts).with_columns(pl.concat_str([pl.col('pair_id'), pl.col('copy').cast(pl.String)], separator='#').alias('pc'))
    a = a.with_columns((pl.col('chrono_rank').rank('ordinal').over('pc') - 1).cast(pl.Float64).alias('chrono_rank'),
                       pl.len().over('pc').cast(pl.Float64).alias('n_hands'),
                       pl.col('oof').rank('ordinal', descending=True).over('pc').cast(pl.Float64).alias('oof_rank'))
    a = a.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac'))
    a = context(a.rename({'pair_id': 'true_pair'}).rename({'pc': 'pair_id'}), src).drop('pair_id').rename({'true_pair': 'pair_id'})
    a = a.filter(pl.col('top0')).select(K + ['copy'] + POOLF + ctxc)
    keep = [c for c in Q.columns if c not in POOLF + ctxc]
    return a.join(Q.select(keep), on=K, how='left', validate='m:1')


# ---- step 110 audit looks [C]: shared harness (part A is the step-109 loader, unchanged) ----
OUT = SP.replace('s109/', 's110/')
FOLDS = {}
for f in FAM:
    ix = np.flatnonzero((Q['behavior_family'] == f).to_numpy())
    q = Q[ix]; y = q['relevance'].to_numpy().astype(int); g = q['table_id'].to_numpy()
    FOLDS[f] = (ix, q, y, list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), y, g)))


def run(name, prefix=True, aug=None, mcs=1, extra=None):
    """aug: extra labelled rows keyed by pair (only training-fold pairs are used); mcs: min_child_samples multiplier;
    extra(f, k, q, z, y, tr): synthetic rows built from the training rows of that fold only."""
    pred = d['oof'].to_numpy().copy() - 1e6
    params = dict(ac.CLF); params['min_child_samples'] = int(round(params['min_child_samples'] * mcs))
    for f in FAM:
        ix, q, y, folds = FOLDS[f]
        cols = columns(f, prefix); z = q.select(cols).to_numpy().astype(float)
        if aug is not None:
            af = aug.filter(pl.col('behavior_family') == f)
            za, ya, ap = af.select(cols).to_numpy().astype(float), af['relevance'].to_numpy().astype(int), af['pair_id'].to_numpy()
            assert np.isfinite(za).all()
        for k, (tr, te) in enumerate(folds):
            X, Y = z[tr], y[tr]
            if aug is not None:
                sel = np.isin(ap, np.unique(q['pair_id'].to_numpy()[tr])); X, Y = np.vstack([X, za[sel]]), np.r_[Y, ya[sel]]
            if extra is not None:
                Xs, Ys = extra(f, k, q, z, y, tr); X, Y = np.vstack([X, Xs]), np.r_[Y, Ys]
            m = lgb.LGBMClassifier(**params, random_state=11).fit(X, Y)
            pred[QIX[ix[te]]] = m.predict_proba(z[te])[:, 1] + 10
    print(f'  fit {name} at {round(time.time() - T0)}s', flush=True)
    return pred


sc = pl.read_parquet(CVS + 'experiments/codex_ideas_sweep_C_20260913/scores-greedy-step1-prefix_strength.parquet').sort(K)
assert sc.select(K).equals(d.select(K))
SS = run('S_star'); assert np.array_equal(SS, sc['S_plus_X'].to_numpy())
fr0 = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', SS)))
assert fr0.select(K).equals(d.select(K))
t1 = align(fr0, pl.read_parquet(CVS + 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
dec = lambda p: decode(fr0.with_columns(pl.Series('score', p)), t1)
yv = d['relevance'].to_numpy()[top]
R = lambda x: round(float(x), 6)


def logloss(p):
    v = np.clip(p[top] - 10, 1e-6, 1 - 1e-6)
    return R(-np.mean(yv * np.log(v) + (1 - yv) * np.log(1 - v)))


def cmp(inc, name, arm):
    c = ac.evaluate(d, {'incumbent': inc, name: arm})['comparisons'][name]
    return {'mean': R(c['mean_delta']), 'ci95': [R(v) for v in c['ci95']], 'fam': {k[:2]: R(v) for k, v in c['by_family'].items()},
            'halves': {k: R(v) for k, v in c['halves'].items()}, 'gate': bool(c['gate_pass'])}


DS = dec(SS)
ER = pl.read_csv('C:/Data/Projects/Money Making/work/poker/data/development_evidence.csv', columns=['pair_id', 'hand_id', 'evidence_rank'])
QE = Q.join(ER, on=K, how='left', validate='1:1')
assert (QE['evidence_rank'].is_not_null().cast(pl.Int64) == QE['relevance'].cast(pl.Int64)).all()
RES = {'S_star_exact_vs_saved': True, 'S_star_logloss': logloss(SS)}


def record(name, p):
    RES[name] = {'logloss': logloss(p), 'vs_S_star': cmp(SS, name, p), 'decode_vs_decode_S_star': cmp(DS, name, dec(p))}
    print(name, json.dumps(RES[name]), flush=True)
    json.dump(RES, open(OUT + 'trials110_results.json', 'w', encoding='utf-8'), indent=1)


# ---- step 116 [C] audit looks: TEST-TIME augmentation of the fixed S* models over label-free thinned pool views ----
# View: drop each hand OUTSIDE stage-1 top-20 with prob u (no labels used), recompute chrono_rank/chrono_frac/n_hands/oof_rank and
# the 136 context_ fields on the thinned pool, keep every top-20 row; cs_/prefix unchanged. Models are the S* fold models (no refit).
# Registered before results: an arm qualifies only if it passes ac gate vs S* AND decode(arm) passes ac gate vs decode(S*).
OUT = SP.replace('s109/', 's116/')


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



# ---- STEP112 [C] typed membership experts, table-bagged, stacked on v1 (registration112.json) ----
import hashlib
from scipy.stats import rankdata
from evidence_aug.paths import check_protected
check_protected()


def record(*a, **k):
    raise RuntimeError('record() writes to the peer scratchpad; unused in STEP112')


MY = 'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/2902d2e3-d89a-4488-96d3-ba0785bfb759/scratchpad/s113/'
OUT = MY
EXP = CVS + 'experiments/membership_experts_C_20260913/'
REGB = open(EXP + 'registration112.json', 'rb').read(); REG = json.loads(REGB); REGSHA = hashlib.sha256(REGB).hexdigest()
print('registration112', REGSHA, flush=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
V1P = CVS + REG['incumbent']['file']; assert sha(V1P) == REG['incumbent']['sha256']
xv = d.select(K).join(pl.read_parquet(V1P), on=K, how='left', validate='1:1', maintain_order='left'); assert xv.select(K).equals(d.select(K))
V1, DV1 = xv['v1'].to_numpy(), xv['dec_v1'].to_numpy()
assert np.array_equal(xv['S_star'].to_numpy(), SS) and np.array_equal(dec(V1), DV1)
at = REG['attributes']; AP = CVS + at['source']; assert sha(AP) == at['source_sha256'] and sha(CVS + at['t1_file']) == at['t1_sha256']
A = Q.select(K).join(pl.read_parquet(AP).select(K + at['raw16']), on=K, how='left', validate='1:1', maintain_order='left')
assert A.select(K).equals(Q.select(K))
RAW = A.select([pl.col(c).cast(pl.Float64) for c in at['raw16']]).to_numpy()
T1Q = t1[QIX].astype(float); qid = Q['pair_id'].to_numpy()
assert RAW.shape == (Q.height, 16) and int(T1Q.sum()) == 818
print('T1 rows', int(T1Q.sum()), 'NaN counts T1/nonT1', dict(zip(at['raw16'], zip(np.isnan(RAW[T1Q == 1]).sum(0).tolist(), np.isnan(RAW[T1Q == 0]).sum(0).tolist()))), flush=True)


def permute_within(M, seed):
    P = M.copy()
    for i, pid in enumerate(np.unique(qid)):
        r = np.flatnonzero(qid == pid)
        P[r] = M[r[np.random.default_rng([seed, i]).permutation(len(r))]]
    return P


def gated(M):
    i1 = [at['raw16'].index(c) for c in at['T1_block']]; i0 = [at['raw16'].index(c) for c in at['nonT1_block']]
    return np.hstack([T1Q[:, None], np.where(T1Q[:, None] == 1, M[:, i1], np.nan), np.where(T1Q[:, None] == 0, M[:, i0], np.nan)])


EXTRA = {'C0': None, 'C1': np.hstack([T1Q[:, None], RAW]), 'C2': gated(RAW), 'C3': gated(permute_within(RAW, 1123))}
ARMS = ('C0', 'C1', 'C2', 'C3'); BAGS, RATE = REG['bagging']['bags'], REG['bagging']['table_rate']


def zmat(frame, f, ix, arm):
    z = frame[ix].select(columns(f)).to_numpy().astype(float)
    return z if EXTRA[arm] is None else np.hstack([z, EXTRA[arm][ix]])


MB, ORIG, MD0 = {}, {}, {}
for arm in ARMS:
    pred = d['oof'].to_numpy().copy() - 1e6
    for fi, f in enumerate(FAM):
        ix, q, y, folds = FOLDS[f]; z = zmat(Q, f, ix, arm); g = q['table_id'].to_numpy()
        for k, (tr, te) in enumerate(folds):
            tabs = np.unique(g[tr]); s = np.zeros(len(te))
            for b in range(BAGS):
                keep = np.random.default_rng([112, fi, k, b]).choice(tabs, size=int(round(RATE * len(tabs))), replace=False)
                trb = tr[np.isin(g[tr], keep)]
                MB[(arm, f, k, b)] = m = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(z[trb], y[trb])
                s += m.predict_proba(z[te])[:, 1]
            pred[QIX[ix[te]]] = s / BAGS + 10
    ORIG[arm] = pred; print('fit', arm, round(time.time() - T0), 's', flush=True)
for f in FAM:
    ix, q, y, folds = FOLDS[f]; z = zmat(Q, f, ix, 'C0')
    for k, (tr, te) in enumerate(folds):
        MD0[(f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(z[tr], y[tr])
P0 = SS.copy()
for f in FAM:
    ix, q, y, folds = FOLDS[f]; z = zmat(Q, f, ix, 'C0')
    for k, (tr, te) in enumerate(folds):
        P0[QIX[ix[te]]] = MD0[(f, k)].predict_proba(z[te])[:, 1] + 10
assert np.array_equal(P0, SS); del P0
NAMES = ('S0',) + ARMS
BASE = {a: (SS if a == 'S0' else ORIG[a]) for a in NAMES}
TH = {a: np.zeros_like(SS) for a in NAMES}; PD = {a: np.zeros_like(SS) for a in NAMES}; n = 0


def fill(v, a):
    p = BASE[a].copy()
    for f in FAM:
        ix, q, y, folds = FOLDS[f]; zv = zmat(v, f, ix, 'C0' if a == 'S0' else a)
        for k, (tr, te) in enumerate(folds):
            if a == 'S0':
                p[QIX[ix[te]]] = MD0[(f, k)].predict_proba(zv[te])[:, 1] + 10
            else:
                p[QIX[ix[te]]] = np.mean([MB[(a, f, k, b)].predict_proba(zv[te])[:, 1] for b in range(BAGS)], axis=0) + 10
    return p


for tgt, kind, seed in ((TH, 'thin', 1241), (TH, 'thin', 2241), (PD, 'pad', 1242), (PD, 'pad', 2242)):
    VV = views(.5, 10, seed) if kind == 'thin' else pads(.5, 10, seed, False)
    for v in VV:
        for a in NAMES:
            tgt[a] += fill(v, a)
        n += 1
    del VV; print('views', n, round(time.time() - T0), 's', flush=True)
assert n == 40
VA = {a: (TH[a] + PD[a] + BASE[a]) / (n + 1) for a in NAMES}
np.savez(MY + 'va112.npz', **VA, **{'orig_' + a: ORIG[a] for a in ARMS})
REP = {'S0_maxabs_vs_v1_all_rows': float(np.abs(VA['S0'] - V1).max()), 'S0_bit_identical': bool(np.array_equal(VA['S0'], V1))}
REP['S0_decoded_delta_vs_dec_v1'] = cmp(DV1, 'a', dec(VA['S0']))['mean']
print('replay', REP, flush=True)
assert REP['S0_maxabs_vs_v1_all_rows'] < 1e-9 and REP['S0_decoded_delta_vs_dec_v1'] == 0


def wq_auc(p):
    fr = pl.DataFrame({'pair_id': d['pair_id'].to_numpy()[QIX], 's': p[QIX], 'y': d['relevance'].to_numpy()[QIX]})
    out = []
    for g_ in fr.partition_by('pair_id'):
        yy, ss = g_['y'].to_numpy(), g_['s'].to_numpy(); n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 and n0:
            rr = rankdata(ss); out.append((rr[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return R(np.mean(out))


DEC = {a: dec(VA[a]) for a in ARMS}
RES = {'registration_sha256': REGSHA, 'views': n, 'replays': REP, 'S_star_exact_vs_saved': True, 'MD0_replays_S_star': True,
       'v1_ref': {'logloss': logloss(V1), 'auc_raw': wq_auc(V1), 'auc_dec': wq_auc(DV1)}}
for a in ARMS:
    RES[a] = {'logloss': logloss(VA[a]), 'auc_raw': wq_auc(VA[a]), 'auc_dec': wq_auc(DEC[a]), 'vs_v1': cmp(V1, 'a', VA[a]),
              'dec_vs_dec_v1': cmp(DV1, 'a', DEC[a]), 'descriptive_orig_logloss': logloss(ORIG[a]),
              'descriptive_orig_dec_vs_decode_S_star': cmp(DS, 'a', dec(ORIG[a]))}
    print(a, json.dumps(RES[a]), flush=True)
RES['pairs'] = {'C2_vs_C0_dec': cmp(DEC['C0'], 'a', DEC['C2']), 'C2_vs_C0_raw': cmp(VA['C0'], 'a', VA['C2']),
                'C2_vs_C3_dec': cmp(DEC['C3'], 'a', DEC['C2']), 'C2_vs_C1_dec': cmp(DEC['C1'], 'a', DEC['C2']),
                'C3_vs_C0_dec': cmp(DEC['C0'], 'a', DEC['C3']), 'C1_vs_C0_dec': cmp(DEC['C0'], 'a', DEC['C1'])}
c0, c2 = RES['C0']['dec_vs_dec_v1'], RES['C2']['dec_vs_dec_v1']
K1 = bool(RES['C0']['vs_v1']['mean'] > 0 and c0['mean'] > 0 and min(c0['fam'].values()) >= -.01)
K2 = bool(RES['C2']['vs_v1']['mean'] > 0 and c2['mean'] > 0 and min(c2['fam'].values()) >= -.01
          and RES['pairs']['C2_vs_C0_dec']['mean'] > 0 and RES['pairs']['C2_vs_C3_dec']['mean'] > 0)
RES['keep'] = {'K1_C0': K1, 'K2_C2': K2, 'v2': 'C2' if K2 else ('C0' if K1 else None)}
cols_out = [pl.Series(a, VA[a]) for a in ARMS] + [pl.Series('dec_' + a, DEC[a]) for a in ARMS]
d.select(K).with_columns(cols_out).write_parquet(EXP + 'audit_step112_scores.parquet')
RES['artifact_sha256'] = sha(EXP + 'audit_step112_scores.parquet')
json.dump(RES, open(EXP + 'results112.json', 'w', encoding='utf-8'), indent=1)
print('pairs', json.dumps(RES['pairs']), flush=True)
print('keep', json.dumps(RES['keep']), 'artifact', RES['artifact_sha256'], flush=True)
check_protected()
print('done', round(time.time() - T0), 's', flush=True)
