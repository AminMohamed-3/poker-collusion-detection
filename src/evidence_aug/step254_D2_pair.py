"""STEP254 D2: do pooled per-pair aggregates of the MECHPLUS event probability help the PAIR model?

The pair-proxy harness of src/pu_pairs/step247_pair_aggregates_new_block.py, reproduced here without
touching that script: the same pair ordering, the same per-pair view RNG channels (seeds 236/237/238,
CH_V70 = 10, CH_V40 = 11, CH_TEST65 = 90), the same pooled training set (full + 70% + 40% views,
weight 1/3 each), the same LightGBM configuration, 5 folds by table.

  OLD   = step247's OLD arm  (4 x 163 aggregates of the production per-hand features + shared_hands)
  MECHP = OLD + six pooled aggregates of the MECHPLUS per-hand p_fmax over the SAME view:
          count above the honest q99.9, count above the honest q99, top-1 / top-3 / top-5 mean logit,
          noisy-or.  (The honest quantiles are the ones in results_mechplus.json.)

Readouts: overall AP on T1 (full history) and T2 (65% thinning), the low-visibility group (positives
with 3 or 4 listed hands, step247's low-dose group) as an AP against all negatives, each split by
table half. Registered pass rule: low-visibility AP >= +.02 on BOTH halves with no overall loss.
Development phase only. Writes results254_D2.json.
"""
from __future__ import annotations

import glob
import json
import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import time

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
OUT = CV + r'\experiments\step254_mechanical_tensor_20260917'
S247 = CV + r'\experiments\step247_pair_new_block_20260917\cache'
EVENT = CV + r'\experiments\step232_event_detector_20260917\hand_oof.parquet'
SEEDS = [236, 237, 238]
CH_V70, CH_V40, CH_TEST65 = 10, 11, 90
LGB = dict(n_estimators=226, num_leaves=15, learning_rate=0.03, reg_lambda=5,
           min_child_samples=20, n_jobs=5, random_state=236, verbose=-1, importance_type='gain')
AGG_NAMES = ['cnt_q999', 'cnt_q99', 'top1_logit', 'top3_logit', 'top5_logit', 'noisy_or']


def half(table_id: str) -> int:
    """the audit's table half: parity of the table id hash, as src/evidence_aug/step5_evaluate.half"""
    from evidence_aug.step5_evaluate import half as H
    return H(table_id)


def pair_aggs(p: np.ndarray, q99: float, q999: float) -> np.ndarray:
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    s = np.sort(lg)[::-1]
    return np.array([float((p > q999).sum()), float((p > q99).sum()),
                     float(s[:1].mean()), float(s[:3].mean()), float(s[:5].mean()),
                     float(-np.log1p(-np.clip(p, 0, 1 - 1e-9)).sum())], dtype=np.float32)


def main() -> None:
    import sys
    sys.path.insert(0, CV + r'\src')
    t0 = time.time()
    meta = json.load(open(S247 + r'\store_meta.json', encoding='utf-8'))
    blocks = np.load(S247 + r'\blocks.npz')
    pair_ids = np.array(meta['pair_ids'])
    label_arr = np.array(meta['label'], dtype=np.int64)
    fold_arr = np.array(meta['fold'], dtype=np.int64)
    table_arr = np.array(meta['table'])
    n_listed = np.array(meta['n_listed'], dtype=np.int64)
    lowdose = np.array(meta['lowdose_idx'], dtype=np.int64)
    n_pairs = meta['n_pairs']

    # --- reproduce step247's hand row order exactly
    feat = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(ROOT + r'\outputs_v2\hands_development\batch_*.parquet'))])
    hand_oof = pl.read_parquet(EVENT)
    df = feat.select('pair_id', 'hand_id', 'table_id').join(
        hand_oof.select('pair_id', 'hand_id', 'table_id', 'fold'),
        on=['pair_id', 'hand_id', 'table_id'], how='inner')
    assert df.height == feat.height
    df = df.sort('pair_id')
    pid_col = df['pair_id'].to_numpy()
    ids, start = np.unique(pid_col, return_index=True)
    order = np.argsort(start)
    ids, start = ids[order], start[order]
    ends = np.append(start[1:], len(pid_col))
    assert (ids == pair_ids).all(), 'pair order differs from step247'

    res_json = json.load(open(OUT + r'\results_mechplus.json', encoding='utf-8'))['mechplus']
    q99, q999 = res_json['_thresholds']['honest_q99'], res_json['_thresholds']['honest_q999']
    oof254 = pl.read_parquet(OUT + r'\hand_oof_step254.parquet')
    key = dict(zip((oof254['pair_id'] + '|' + oof254['hand_id']).to_list(), range(oof254.height)))
    pv = oof254['mechplus_p_fmax'].to_numpy()
    gidx = np.fromiter((key[k] for k in (df['pair_id'] + '|' + df['hand_id']).to_list()),
                       dtype=np.int64, count=df.height)
    P = pv[gidx]
    print('attached MECHPLUS p_fmax', P.shape, 'q99', q99, 'q999', q999, flush=True)

    # --- per-pair aggregate blocks, one per view
    views = {'full': None}
    A = {'full': np.zeros((n_pairs, 6), np.float32)}
    for nm in ('v70', 'v40', 'test65'):
        for s in SEEDS:
            A[f'{nm}_{s}'] = np.zeros((n_pairs, 6), np.float32)
    for i in range(n_pairs):
        pi = P[start[i]:ends[i]]
        n_i = len(pi)
        A['full'][i] = pair_aggs(pi, q99, q999)
        for s in SEEDS:
            for nm, ch, frac in (('v70', CH_V70, .70), ('v40', CH_V40, .40), ('test65', CH_TEST65, .65)):
                k = max(1, int(round(frac * n_i)))
                idx = np.random.default_rng([s, i, ch]).choice(n_i, size=k, replace=False)
                A[f'{nm}_{s}'][i] = pair_aggs(pi[idx], q99, q999)
    print('aggregates built', round(time.time() - t0, 1), 's', flush=True)

    def old_block(view, seed=None):
        if view == 'full':
            return np.concatenate([blocks['old_full'], blocks['n_full'][:, None]], axis=1)
        return np.concatenate([blocks[f'old_{view}_{seed}'], blocks[f'n_{view}_{seed}'][:, None]], axis=1)

    def X_of(arm, view, seed=None):
        o = old_block(view, seed)
        if arm == 'OLD':
            return o
        a = A['full'] if view == 'full' else A[f'{view}_{seed}']
        return np.concatenate([o[:, :-1], a, o[:, -1:]], axis=1)

    hv = np.array([half(t) for t in table_arr])
    out = {'arms': {}, 'n_pairs': int(n_pairs), 'n_lowdose': int(len(lowdose)),
           'half_counts': [int((hv == 0).sum()), int((hv == 1).sum())],
           'thresholds': {'honest_q99': q99, 'honest_q999': q999}}
    oof = {}
    for arm in ('OLD', 'MECHP'):
        t1s, t2s = [], []
        for s in SEEDS:
            Xpool = np.concatenate([X_of(arm, 'full'), X_of(arm, 'v70', s), X_of(arm, 'v40', s)], axis=0)
            ypool = np.tile(label_arr, 3)
            wpool = np.full(Xpool.shape[0], 1.0 / 3)
            foldpool = np.tile(fold_arr, 3)
            Xf, Xt = X_of(arm, 'full'), X_of(arm, 'test65', s)
            o1, o2 = np.zeros(n_pairs), np.zeros(n_pairs)
            for f in range(5):
                m = LGBMClassifier(**LGB).fit(Xpool[foldpool != f], ypool[foldpool != f],
                                              sample_weight=wpool[foldpool != f])
                te = np.where(fold_arr == f)[0]
                o1[te] = m.predict_proba(Xf[te])[:, 1]
                o2[te] = m.predict_proba(Xt[te])[:, 1]
            t1s.append(o1)
            t2s.append(o2)
            print(arm, 'seed', s, 'dim', Xpool.shape[1], round(time.time() - t0, 1), 's', flush=True)
        oof[arm] = (t1s, t2s)

        def block(ts):
            r = {}
            aps = [average_precision_score(label_arr, o) for o in ts]
            r['AP'] = round(float(np.mean(aps)), 5)
            r['AP_seeds'] = [round(float(a), 5) for a in aps]
            sel = np.zeros(n_pairs, bool)
            sel[lowdose] = True
            keep = sel | (label_arr == 0)
            r['lowdose_AP'] = round(float(np.mean([average_precision_score(label_arr[keep], o[keep])
                                                   for o in ts])), 5)
            for h in (0, 1):
                mh = hv == h
                r[f'AP_half{h}'] = round(float(np.mean([average_precision_score(label_arr[mh], o[mh])
                                                        for o in ts])), 5)
                kh = keep & mh
                r[f'lowdose_AP_half{h}'] = round(float(np.mean(
                    [average_precision_score(label_arr[kh], o[kh]) for o in ts])), 5)
            return r

        out['arms'][arm] = {'T1': block(t1s), 'T2': block(t2s)}
        print(arm, json.dumps(out['arms'][arm]), flush=True)
    d = {}
    for v in ('T1', 'T2'):
        d[v] = {k: round(out['arms']['MECHP'][v][k] - out['arms']['OLD'][v][k], 5)
                for k in out['arms']['OLD'][v] if k != 'AP_seeds'}
    out['delta_MECHP_minus_OLD'] = d
    ok = all(d[v]['lowdose_AP_half0'] >= .02 and d[v]['lowdose_AP_half1'] >= .02 and d[v]['AP'] >= 0
             for v in ('T2',))
    out['D2'] = 'PASS' if ok else 'FAIL'
    out['pass_rule'] = ('low-visibility (3-4 listed) AP >= +.02 on both table halves with no overall '
                        'AP loss, on the 65% thinned test view T2')
    out['seconds'] = round(time.time() - t0, 1)
    json.dump(out, open(OUT + r'\results254_D2.json', 'w', encoding='utf-8'), indent=1)
    print(json.dumps(d, indent=1))
    print('D2', out['D2'])


if __name__ == '__main__':
    main()
