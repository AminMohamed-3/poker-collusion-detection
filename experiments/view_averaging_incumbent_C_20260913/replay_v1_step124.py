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




# ---- step 124 [C] audit incumbent v1 under Amin's stacking directive: decode(S* averaged over thinned + padded pool views) ----
OUT = SP.replace('s109/', 's124/')
import hashlib
from evidence_aug.paths import check_protected
check_protected()
EXP = CVS + 'experiments/view_averaging_incumbent_C_20260913/'
os.makedirs(EXP, exist_ok=True)
REG = {'study': 'STEP124 audit incumbent v1: view-averaged S* (stacking directive)', 'held_read_count_mine': 9,
       'directive': 'Amin 2026-09-13: stack improvements; keep small progress and build on it; agents build on each other',
       'evidence_so_far': 'fixed-model view averaging positive on every read: steps 116/117/119/121/122 MAP +.004..+.009, decoded +.004..+.007 (all below the .016 detection limit)',
       'primary_v1': 'S* fold models; mean over original + 20 thinned views (drop non-top-20 hands w.p. .5; seeds 1241, 2241) + 20 padded views (duplicate non-top-20 hands after all hands w.p. .5; seeds 1242, 2242); label-free; no refit',
       'descriptive': 'thin20 only, pad20 only (same views)',
       'keep_rule': 'KEEP v1 as the audit incumbent iff MAP mean vs S* > 0 AND decode(v1) mean vs decode(S*) > 0 AND every family decoded delta >= -.01 (a user-directed small-gain keep rule, not a confirmation gate; ac gate reported)',
       'artifact': 'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet (pair_id, hand_id, S_star, v1, dec_S_star, dec_v1) for the peer sessions to stack on',
       'prediction': {'map': .006, 'decoded': .005, 'P_keep': .7}}
rb = (json.dumps(REG, indent=1) + '\n').encode('utf-8'); open(OUT + 'registration124.json', 'wb').write(rb); open(EXP + 'registration124.json', 'wb').write(rb)
print('registration124 sha256', hashlib.sha256(rb).hexdigest(), flush=True)
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


def fitm(aug=None, mcs=1):
    params = dict(ac.CLF); params['min_child_samples'] = int(round(params['min_child_samples'] * mcs)); MD, pred = {}, d['oof'].to_numpy().copy() - 1e6
    for f in FAM:
        ix, q, y, folds = FOLDS[f]; cols = columns(f); z = q.select(cols).to_numpy().astype(float)
        if aug is not None:
            za = np.vstack([v[ix].select(cols).to_numpy().astype(float) for v in aug])
            ya, ap = np.tile(y, len(aug)), np.tile(q['pair_id'].to_numpy(), len(aug))
        for k, (tr, te) in enumerate(folds):
            X, Y = z[tr], y[tr]
            if aug is not None:
                sel = np.isin(ap, np.unique(q['pair_id'].to_numpy()[tr])); X, Y = np.vstack([X, za[sel]]), np.r_[Y, ya[sel]]
            MD[(f, k)] = lgb.LGBMClassifier(**params, random_state=11).fit(X, Y)
            pred[QIX[ix[te]]] = MD[(f, k)].predict_proba(z[te])[:, 1] + 10
    return MD, pred


def tta(MD, orig, VV):
    PV = []
    for v in VV:
        p = orig.copy()
        for f in FAM:
            ix, q, y, folds = FOLDS[f]; zv = v[ix].select(columns(f)).to_numpy().astype(float)
            for k, (tr, te) in enumerate(folds):
                p[QIX[ix[te]]] = MD[(f, k)].predict_proba(zv[te])[:, 1] + 10
        PV.append(p)
    return np.mean(PV + [orig], axis=0)


def rd(p, ref=None):
    r = {'logloss': logloss(p), 'vs_S_star': cmp(SS, 'a', p), 'decode_vs_decode_S_star': cmp(DS, 'a', dec(p))}
    if ref is not None:
        r['vs_control'] = cmp(ref, 'a', p); r['decode_vs_control'] = cmp(dec(ref), 'a', dec(p))
    r['qualifies'] = bool(r['vs_S_star']['gate'] and r['decode_vs_decode_S_star']['gate'] and (ref is None or r['vs_control']['mean'] > 0))
    return r



MD0, P0 = fitm(); assert np.array_equal(P0, SS)


def acc(build):
    tot, n = np.zeros_like(SS), 0
    for VV in build:
        for v in VV:
            p = SS.copy()
            for f in FAM:
                ix, q, y, folds = FOLDS[f]; zv = v[ix].select(columns(f)).to_numpy().astype(float)
                for k, (tr, te) in enumerate(folds):
                    p[QIX[ix[te]]] = MD0[(f, k)].predict_proba(zv[te])[:, 1] + 10
            tot += p; n += 1
        del VV
        print('  views', n, round(time.time() - T0), 's', flush=True)
    return tot, n


TH, nt = acc(views(.5, 10, s) for s in (1241, 2241))
PD, npd = acc(pads(.5, 10, s, False) for s in (1242, 2242))
V1 = (TH + PD + SS) / (nt + npd + 1)
RES = {'registration_sha256': hashlib.sha256(rb).hexdigest(), 'views': [nt, npd]}
for name, p in (('v1_thin20_pad20', V1), ('thin20', (TH + SS) / (nt + 1)), ('pad20', (PD + SS) / (npd + 1))):
    RES[name] = rd(p); RES[name].pop('qualifies')
r = RES['v1_thin20_pad20']
r['keep'] = bool(r['vs_S_star']['mean'] > 0 and r['decode_vs_decode_S_star']['mean'] > 0 and min(r['decode_vs_decode_S_star']['fam'].values()) >= -.01)
r['ac_gate_both'] = bool(r['vs_S_star']['gate'] and r['decode_vs_decode_S_star']['gate'])
out = d.select(K).with_columns(pl.Series('S_star', SS), pl.Series('v1', V1), pl.Series('dec_S_star', DS), pl.Series('dec_v1', dec(V1)))
out.write_parquet(EXP + 'audit_v1_scores.parquet')
RES['artifact_sha256'] = hashlib.sha256(open(EXP + 'audit_v1_scores.parquet', 'rb').read()).hexdigest()
json.dump(RES, open(OUT + 'results124.json', 'w', encoding='utf-8'), indent=1); json.dump(RES, open(EXP + 'results124.json', 'w', encoding='utf-8'), indent=1)
for k, v in RES.items():
    if isinstance(v, dict):
        print(k, 'vsS', v['vs_S_star']['mean'], v['vs_S_star']['ci95'], v['vs_S_star']['fam'], 'dec', v['decode_vs_decode_S_star']['mean'],
              v['decode_vs_decode_S_star']['ci95'], v['decode_vs_decode_S_star']['fam'], 'll', v['logloss'], 'keep', v.get('keep'), 'gate', v.get('ac_gate_both'))
print('artifact sha', RES['artifact_sha256'])
check_protected()
print('done', round(time.time() - T0), 's')
