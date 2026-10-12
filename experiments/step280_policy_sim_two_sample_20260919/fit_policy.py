"""STEP280 stage 3: fit the honest conditional action policy.

Two heads, each a 5-model bag over POLICY_TRAIN table folds (registration P3):
  action head  - 5-class {fold, check, call, raise, all_in}
  size head    - 8 pot-fraction buckets, aggressive (bet/raise) rows only
Held-out log-loss is reported on POLICY_HELD against two frequency baselines, both
renormalised over the same legal mask as the model.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_lib as S
from prep_data import SCRATCH

HERE = Path(__file__).resolve().parent
NBAG = 5
PARAMS_A = dict(objective='multiclass', num_class=5, learning_rate=.05, num_leaves=63,
                min_child_samples=50, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
                num_threads=4, verbosity=-1)
PARAMS_S = dict(objective='multiclass', num_class=S.N_SIZE, learning_rate=.05, num_leaves=63,
                min_child_samples=50, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
                num_threads=4, verbosity=-1)
N_TREES_A, N_TREES_S = 600, 400


def masked_logloss(prob, y, mask, nclass):
    bits = np.stack([(mask >> k) & 1 for k in range(nclass)], axis=1).astype(np.float64)
    p = prob * bits
    p = p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, None)).mean())


def plain_logloss(prob, y):
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-15, None)).mean())


def main():
    t0 = time.time()
    df = pl.read_parquet(SCRATCH / 'dec_POLICY_TRAIN_POLICY_HELD.parquet')
    df = df.with_columns(
        pl.col('table_id').map_elements(S.policy_bag, return_dtype=pl.Int64).alias('bag'))
    tr = df.filter(pl.col('grp') == 'POLICY_TRAIN')
    he = df.filter(pl.col('grp') == 'POLICY_HELD')
    print(f'train {tr.height} held {he.height} ({time.time() - t0:.0f}s)', flush=True)

    Xtr = tr.select(S.FEAT_NAMES).to_numpy().astype(np.float32)
    Xhe = he.select(S.FEAT_NAMES).to_numpy().astype(np.float32)
    ya_tr = tr['y_action'].to_numpy().astype(np.int64)
    ya_he = he['y_action'].to_numpy().astype(np.int64)
    mk_tr = tr['legal_mask'].to_numpy().astype(np.int64)
    mk_he = he['legal_mask'].to_numpy().astype(np.int64)
    bag_tr = tr['bag'].to_numpy()

    # ---- action head
    models_a, pa = [], np.zeros((len(ya_he), 5))
    for b in range(NBAG):
        m = lgb.train({**PARAMS_A, 'seed': 7329 + b, 'bagging_seed': 7329 + b,
                       'feature_fraction_seed': 7329 + b},
                      lgb.Dataset(Xtr[bag_tr != b], label=ya_tr[bag_tr != b],
                                  feature_name=S.FEAT_NAMES),
                      num_boost_round=N_TREES_A)
        models_a.append(m)
        pa += m.predict(Xhe) / NBAG
        print(f'  action bag {b} done ({time.time() - t0:.0f}s)', flush=True)

    # ---- size head (aggressive rows only)
    agg_tr = tr.filter(pl.col('y_size') >= 0)
    agg_he = he.filter(pl.col('y_size') >= 0)
    Xs_tr = agg_tr.select(S.FEAT_NAMES).to_numpy().astype(np.float32)
    Xs_he = agg_he.select(S.FEAT_NAMES).to_numpy().astype(np.float32)
    ys_tr = agg_tr['y_size'].to_numpy().astype(np.int64)
    ys_he = agg_he['y_size'].to_numpy().astype(np.int64)
    bs = agg_tr['bag'].to_numpy()
    models_s, ps = [], np.zeros((len(ys_he), S.N_SIZE))
    for b in range(NBAG):
        m = lgb.train({**PARAMS_S, 'seed': 7329 + b, 'bagging_seed': 7329 + b,
                       'feature_fraction_seed': 7329 + b},
                      lgb.Dataset(Xs_tr[bs != b], label=ys_tr[bs != b],
                                  feature_name=S.FEAT_NAMES),
                      num_boost_round=N_TREES_S)
        models_s.append(m)
        ps += m.predict(Xs_he) / NBAG
        print(f'  size bag {b} done ({time.time() - t0:.0f}s)', flush=True)

    # ---- baselines
    glob = np.bincount(ya_tr, minlength=5) / len(ya_tr)
    base_g = np.tile(glob, (len(ya_he), 1))
    key_tr = tr['ctx_street'].to_numpy() * 2 + tr['ctx_facing'].to_numpy()
    key_he = he['ctx_street'].to_numpy() * 2 + he['ctx_facing'].to_numpy()
    cond = np.zeros((8, 5))
    for k in range(8):
        sel = key_tr == k
        cond[k] = np.bincount(ya_tr[sel], minlength=5) / max(sel.sum(), 1) if sel.sum() else glob
    cond = np.clip(cond, 1e-6, None)
    cond /= cond.sum(1, keepdims=True)
    base_c = cond[key_he]

    # within-bucket pot-fraction pools for size realisation
    pools = {}
    for row in agg_tr.select('ctx_street', 'ctx_facing', 'y_size', 'y_frac').iter_rows():
        pools.setdefault((row[0], row[1], row[2]), []).append(row[3])
    pools = {f'{k[0]}_{k[1]}_{k[2]}': np.asarray(v, dtype=np.float32) for k, v in pools.items()}

    res = {
        'decisions_train': int(tr.height), 'decisions_held': int(he.height),
        'aggressive_train': int(agg_tr.height), 'aggressive_held': int(agg_he.height),
        'action_class_share_train': {c: float((ya_tr == i).mean())
                                     for i, c in enumerate(S.ACTION_CLASSES)},
        'action_head': {
            'held_logloss_masked': masked_logloss(pa, ya_he, mk_he, 5),
            'held_logloss_plain': plain_logloss(pa, ya_he),
            'baseline_global_masked': masked_logloss(base_g, ya_he, mk_he, 5),
            'baseline_street_facing_masked': masked_logloss(base_c, ya_he, mk_he, 5),
            'baseline_global_plain': plain_logloss(base_g, ya_he),
            'baseline_street_facing_plain': plain_logloss(base_c, ya_he),
            'held_accuracy': float((pa.argmax(1) == ya_he).mean()),
        },
        'size_head': {
            'held_logloss': plain_logloss(ps, ys_he),
            'baseline_marginal': plain_logloss(
                np.tile(np.bincount(ys_tr, minlength=S.N_SIZE) / len(ys_tr), (len(ys_he), 1)),
                ys_he),
            'held_accuracy': float((ps.argmax(1) == ys_he).mean()),
        },
        'seconds': round(time.time() - t0, 1),
    }
    ah = res['action_head']
    ah['improvement_vs_street_facing_masked'] = round(
        1 - ah['held_logloss_masked'] / ah['baseline_street_facing_masked'], 4)
    ah['improvement_vs_global_masked'] = round(
        1 - ah['held_logloss_masked'] / ah['baseline_global_masked'], 4)
    (HERE / 'policy_metrics.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    with open(SCRATCH / 'policy.pkl', 'wb') as fh:
        pickle.dump({'action': models_a, 'size': models_s, 'pools': pools}, fh)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
