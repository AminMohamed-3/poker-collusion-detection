"""STEP261 -- Amin's pair-side questions of 2026-09-18, measured in the STEP247 / STEP254-D2 pair-proxy harness.

Q2 "did we try increasing the feature count?"  -> arms AGG / AGG6 / ALL: per-pair aggregates (mean, std ddof=1, max, q95
   nearest = the STEP247 convention) of ALL 662 STEP254 mechanical per-hand columns (+2,648 pair columns), with a
   same-size Gaussian NOISE control for dilution.
Q3 "did we try a 30-40 model ensemble with different hyper-parameters?" -> arms ENS30 (30 LightGBM configurations drawn
   from a registered distribution, mean of logits) on the OLD and on the AGG6 feature sets, and MLP5 (five scikit-learn
   MLPs on standardised inputs = the "MLP" part of Q2) alone and added to ENS30.
Same population (1,860 labelled development pairs), same views (full + 70 % + 40 % training views, T1 = full test view,
T2 = 65 % thinned test view = evaluation-like), same folds, same view RNG channels as step254_D2_pair.py.
Single-model arms: the harness LightGBM, 3 view seeds.  Ensemble arms: view seed 236 only (cost).
Registration: experiments/step261_pair_feature_expansion_20260918/registration261.json (written first).
Development phase only.  Builds no submission.  Stage: python step261_... build | arms | ens
"""
from __future__ import annotations
import os
os.environ.setdefault('POLARS_MAX_THREADS', '3')
import glob, json, sys, time
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

ROOT = r'C:\Data\Projects\Money Making\work\poker'
CV = ROOT + r'\Claude Version'
sys.path.insert(0, CV + r'\src')
OUT = CV + r'\experiments\step261_pair_feature_expansion_20260918'
S254 = CV + r'\experiments\step254_mechanical_tensor_20260917'
S247 = CV + r'\experiments\step247_pair_new_block_20260917\cache'
EVENT = CV + r'\experiments\step232_event_detector_20260917\hand_oof.parquet'
SEEDS = [236, 237, 238]
CH = {'v70': (10, .70), 'v40': (11, .40), 'test65': (90, .65)}
NJ = int(os.environ.get('STEP261_NJOBS', '3'))
LGB = dict(n_estimators=226, num_leaves=15, learning_rate=0.03, reg_lambda=5, min_child_samples=20, n_jobs=NJ,
           random_state=236, verbose=-1)


def agg_block(X):
    n = X.shape[0]
    std = X.std(axis=0, ddof=1) if n > 1 else np.zeros(X.shape[1], X.dtype)
    return np.concatenate([X.mean(axis=0), std, X.max(axis=0), np.quantile(X, .95, axis=0, method='nearest')]).astype(np.float32)


def pair_aggs6(p, q99, q999):
    c = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(c / (1 - c)); s = np.sort(lg)[::-1]
    return np.array([float((p > q999).sum()), float((p > q99).sum()), float(s[:1].mean()), float(s[:3].mean()), float(s[:5].mean()),
                     float(-np.log1p(-np.clip(p, 0, 1 - 1e-9)).sum())], dtype=np.float32)


def frame():
    meta = json.load(open(S247 + r'\store_meta.json', encoding='utf-8'))
    feat = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(ROOT + r'\outputs_v2\hands_development\batch_*.parquet'))])
    hand_oof = pl.read_parquet(EVENT)
    df = feat.select('pair_id', 'hand_id', 'table_id').join(hand_oof.select('pair_id', 'hand_id', 'table_id', 'fold'),
                                                            on=['pair_id', 'hand_id', 'table_id'], how='inner')
    assert df.height == feat.height
    df = df.sort('pair_id')
    pid = df['pair_id'].to_numpy()
    ids, start = np.unique(pid, return_index=True)
    o = np.argsort(start); ids, start = ids[o], start[o]
    ends = np.append(start[1:], len(pid))
    assert (ids == np.array(meta['pair_ids'])).all()
    ph = pl.read_parquet(ROOT + r'\data\hands.parquet', columns=['hand_id', 'phase']).join(df.select('hand_id').unique(), on='hand_id')
    assert ph['phase'].unique().to_list() == ['development']
    return meta, feat, df, start, ends


def stage_build():
    t0 = time.time()
    meta, feat, df, start, ends = frame()
    n_pairs = meta['n_pairs']
    fx = pl.read_parquet(S254 + r'\step254_features.parquet')
    MECH = [c for c in fx.columns if c not in ('pair_id', 'hand_id')]
    j = df.select('pair_id', 'hand_id').join(fx, on=['pair_id', 'hand_id'], how='left', maintain_order='left')
    assert j[MECH[0]].null_count() == 0 and (j['pair_id'] == df['pair_id']).all() and (j['hand_id'] == df['hand_id']).all()
    M = j.select(MECH).to_numpy().astype(np.float32)
    # parity anchor: the cached OLD block of one thinned view must be reproduced from the raw 163 columns with these subsets
    blocks = np.load(S247 + r'\blocks.npz')
    raw_cols = [c for c in feat.columns if c not in ('pair_id', 'hand_id', 'table_id')]
    R = df.select('pair_id', 'hand_id').join(feat.select(['pair_id', 'hand_id'] + raw_cols), on=['pair_id', 'hand_id'], how='left',
                                             maintain_order='left').select(raw_cols).to_numpy().astype(np.float32)
    oof = pl.read_parquet(S254 + r'\hand_oof_step254.parquet').select('pair_id', 'hand_id', 'mechplus_p_fmax')
    P = df.select('pair_id', 'hand_id').join(oof, on=['pair_id', 'hand_id'], how='left', maintain_order='left')['mechplus_p_fmax'].to_numpy()
    th = json.load(open(S254 + r'\results_mechplus.json', encoding='utf-8'))['mechplus']['_thresholds']
    q99, q999 = th['honest_q99'], th['honest_q999']
    views = ['full'] + [f'{nm}_{s}' for nm in CH for s in SEEDS]
    A = {v: np.zeros((n_pairs, 4 * len(MECH)), np.float32) for v in views}
    A6 = {v: np.zeros((n_pairs, 6), np.float32) for v in views}
    par = 0.0
    for i in range(n_pairs):
        a, b = start[i], ends[i]; n_i = b - a
        A['full'][i] = agg_block(M[a:b]); A6['full'][i] = pair_aggs6(P[a:b], q99, q999)
        for s in SEEDS:
            for nm, (ch, frac) in CH.items():
                k = max(1, int(round(frac * n_i)))
                idx = np.random.default_rng([s, i, ch]).choice(n_i, size=k, replace=False)
                A[f'{nm}_{s}'][i] = agg_block(M[a:b][idx]); A6[f'{nm}_{s}'][i] = pair_aggs6(P[a:b][idx], q99, q999)
                if nm == 'test65' and s == 236 and i % 50 == 0:
                    chk = agg_block(R[a:b][idx]); ref = blocks['old_test65_236'][i]
                    par = max(par, float(np.nanmax(np.abs(chk - ref) / (1 + np.abs(ref)))))
    print('parity of re-derived OLD test65_236 block (max rel diff on sampled pairs):', par, flush=True)
    assert par < 1e-3, 'view subsets do not reproduce the cached OLD block: row order or RNG channel differs'
    np.savez(OUT + r'\mech_aggs.npz', **{'agg_' + v: A[v] for v in views}, **{'six_' + v: A6[v] for v in views})
    json.dump({'mech_cols': MECH, 'parity_max_rel_diff': par, 'seconds': round(time.time() - t0, 1)},
              open(OUT + r'\build261.json', 'w', encoding='utf-8'), indent=1)
    print('built', round(time.time() - t0, 1), 's')


def load_all():
    meta = json.load(open(S247 + r'\store_meta.json', encoding='utf-8'))
    blocks = np.load(S247 + r'\blocks.npz'); Z = np.load(OUT + r'\mech_aggs.npz')
    from evidence_aug.step5_evaluate import half
    hv = np.array([half(t) for t in meta['table']])
    lab = np.array(meta['label'], dtype=np.int64); fold = np.array(meta['fold'], dtype=np.int64)
    low = np.zeros(meta['n_pairs'], bool); low[np.array(meta['lowdose_idx'], dtype=np.int64)] = True
    rng = np.random.default_rng(261)
    noise = {}

    def X_of(arm, view, seed=None):
        key = 'full' if view == 'full' else f'{view}_{seed}'
        old = blocks['old_full'] if view == 'full' else blocks[f'old_{view}_{seed}']
        n = (blocks['n_full'] if view == 'full' else blocks[f'n_{view}_{seed}'])[:, None]
        parts = [old]
        if arm in ('M6', 'AGG6', 'ALL'):
            parts.append(Z['six_' + key])
        if arm in ('AGG', 'AGG6', 'ALL'):
            parts.append(Z['agg_' + key])
        if arm == 'ALL':
            parts.append(blocks['newreal_full'] if view == 'full' else blocks[f'newreal_{view}_{seed}'])
        if arm == 'NOISE':
            if key not in noise:
                noise[key] = rng.standard_normal((old.shape[0], Z['agg_full'].shape[1])).astype(np.float32)
            parts.append(noise[key])
        return np.concatenate(parts + [n], axis=1)

    def score(ts):
        r = {'AP': round(float(np.mean([average_precision_score(lab, o) for o in ts])), 5),
             'AP_seeds': [round(float(average_precision_score(lab, o)), 5) for o in ts]}
        keep = low | (lab == 0)
        r['lowdose_AP'] = round(float(np.mean([average_precision_score(lab[keep], o[keep]) for o in ts])), 5)
        for h in (0, 1):
            m = hv == h
            r[f'AP_half{h}'] = round(float(np.mean([average_precision_score(lab[m], o[m]) for o in ts])), 5)
            r[f'lowdose_AP_half{h}'] = round(float(np.mean([average_precision_score(lab[keep & m], o[keep & m]) for o in ts])), 5)
        return r
    return meta, lab, fold, X_of, score


def fit_predict(make, arm, s, lab, fold, X_of, n_pairs, transform=None):
    Xpool = np.concatenate([X_of(arm, 'full'), X_of(arm, 'v70', s), X_of(arm, 'v40', s)], axis=0)
    ypool = np.tile(lab, 3); fpool = np.tile(fold, 3); w = np.full(len(ypool), 1 / 3)
    Xf, Xt = X_of(arm, 'full'), X_of(arm, 'test65', s)
    o1, o2 = np.zeros(n_pairs), np.zeros(n_pairs)
    for f in range(5):
        tr = fpool != f; te = np.where(fold == f)[0]
        if transform is not None:
            tf = transform(Xpool[tr]); m = make().fit(tf(Xpool[tr]), ypool[tr]); g = lambda A: m.predict_proba(tf(A))[:, 1]
        else:
            m = make().fit(Xpool[tr], ypool[tr], sample_weight=w[tr]); g = lambda A: m.predict_proba(A)[:, 1]
        o1[te] = g(Xf[te]); o2[te] = g(Xt[te])
    return o1, o2


def stage_arms():
    meta, lab, fold, X_of, score = load_all(); n = meta['n_pairs']; t0 = time.time(); out = {}
    for arm in ('OLD', 'M6', 'AGG', 'AGG6', 'ALL', 'NOISE'):
        t1s, t2s = [], []
        for s in SEEDS:
            o1, o2 = fit_predict(lambda: LGBMClassifier(**LGB), arm, s, lab, fold, X_of, n); t1s.append(o1); t2s.append(o2)
        out[arm] = {'dim': int(X_of(arm, 'full').shape[1]), 'T1': score(t1s), 'T2': score(t2s)}
        np.savez(OUT + f'\\oof_{arm}.npz', T1=np.stack(t1s), T2=np.stack(t2s))
        print(arm, json.dumps(out[arm]), round(time.time() - t0), 's', flush=True)
        json.dump(out, open(OUT + r'\results261_arms.json', 'w', encoding='utf-8'), indent=1)


def stage_ens():
    meta, lab, fold, X_of, score = load_all(); n = meta['n_pairs']; t0 = time.time(); out = {}
    rng = np.random.default_rng(2610)
    cfgs = [dict(n_estimators=int(rng.integers(150, 801)), num_leaves=int(rng.choice([7, 15, 31, 63])), learning_rate=float(10 ** rng.uniform(-2, -1)),
                 min_child_samples=int(rng.integers(5, 51)), colsample_bytree=float(rng.uniform(.2, .9)), subsample=float(rng.uniform(.6, 1.0)), subsample_freq=1,
                 reg_lambda=float(rng.uniform(0, 20)), extra_trees=bool(rng.random() < .3), n_jobs=NJ, verbose=-1, random_state=int(rng.integers(1, 10 ** 6))) for _ in range(30)]
    json.dump(cfgs, open(OUT + r'\ens30_configs.json', 'w', encoding='utf-8'), indent=1)
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import QuantileTransformer

    def qt(Xtr):
        q = QuantileTransformer(n_quantiles=200, output_distribution='normal', subsample=10 ** 6, random_state=0).fit(np.nan_to_num(Xtr))
        return lambda A: q.transform(np.nan_to_num(A))
    for arm in ('OLD', 'AGG6'):
        L1, L2 = [], []
        for ci, c in enumerate(cfgs):
            o1, o2 = fit_predict(lambda c=c: LGBMClassifier(**c), arm, 236, lab, fold, X_of, n); L1.append(lg(o1)); L2.append(lg(o2))
            if ci in (0, 4, 9, 19, 29):
                out[f'{arm}_ENS{ci + 1}'] = {'T1': score([np.mean(L1, axis=0)]), 'T2': score([np.mean(L2, axis=0)])}
                print(arm, 'ENS', ci + 1, json.dumps(out[f'{arm}_ENS{ci + 1}']['T2']), round(time.time() - t0), 's', flush=True)
        out[f'{arm}_ENS30_members_T2_AP'] = [round(float(average_precision_score(lab, o)), 5) for o in L2]
        M1, M2 = [], []
        for ms in range(5):
            o1, o2 = fit_predict(lambda ms=ms: MLPClassifier(hidden_layer_sizes=(128, 32), alpha=.03, max_iter=300, early_stopping=True, random_state=ms),
                                 arm, 236, lab, fold, X_of, n, transform=qt); M1.append(lg(o1)); M2.append(lg(o2))
        out[f'{arm}_MLP5'] = {'T1': score([np.mean(M1, axis=0)]), 'T2': score([np.mean(M2, axis=0)])}
        z = lambda a: (a - a.mean()) / a.std()
        out[f'{arm}_ENS30+MLP5'] = {'T1': score([z(np.mean(L1, axis=0)) + z(np.mean(M1, axis=0))]), 'T2': score([z(np.mean(L2, axis=0)) + z(np.mean(M2, axis=0))])}
        print(arm, 'MLP5', json.dumps(out[f'{arm}_MLP5']['T2']), 'ENS30+MLP5', json.dumps(out[f'{arm}_ENS30+MLP5']['T2']), flush=True)
        np.savez(OUT + f'\\oof_ens_{arm}.npz', L1=np.stack(L1), L2=np.stack(L2), M1=np.stack(M1), M2=np.stack(M2))
        json.dump(out, open(OUT + r'\results261_ens.json', 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    assert os.path.exists(OUT + r'\registration261.json'), 'register first'
    {'build': stage_build, 'arms': stage_arms, 'ens': stage_ens}[sys.argv[1]]()
