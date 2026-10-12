"""STEP257 part 1, stage D: pair aggregates, the schema / drift readouts and the receipt.

  1. eval_pair_mech_aggregates.parquet - one row per evaluation pair, n_hands plus the six STEP254-D2
     aggregates of p_fmax over the pair's FULL evaluation history, computed with
     src/evidence_aug/step254_D2_pair.py:pair_aggs verbatim and the registered DEVELOPMENT constants
     q99 = .05541, q99.9 = .44939 (results_mechplus.json).
  2. G_schema  - column order / dtype check, nulls, and the columns that are constant on evaluation
     but not on development (reported).
  3. G_drift   - per column AUC separating a development sample from a 300k-row evaluation sample;
     the 30 most drifting columns are listed.  Nothing is dropped or transformed.
     Plus the share of evaluation pair-hands above the development honest q99 / q99.9 of p_fmax,
     restricted to pairs whose STEP251 production risk score is below .01.
  4. receipt257.json

Usage: python step257_aggregate.py
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import glob
import hashlib
import json
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True

import step254_D2_pair as D2                 # noqa: E402  (pair_aggs, verbatim)
import step257_prep_eval as PREP             # noqa: E402

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
OUT = CV + r'\experiments\step257_eval_mech_scores_20260918'
MODELS = OUT + r'\models'
S254 = CV + r'\experiments\step254_mechanical_tensor_20260917'
S237 = CV + r'\experiments\step237_invisible_events_20260917'
SCRATCH = PREP.SCRATCH
KEYS = ['pair_id', 'hand_id']
Q99, Q999 = .05541, .44939
AGG_NAMES = list(D2.AGG_NAMES)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def auc_from_ranks(x_dev: np.ndarray, x_eval: np.ndarray) -> float:
    """Mann-Whitney AUC with mid-ranks for ties; 1 = the column separates evaluation above."""
    n1, n2 = len(x_dev), len(x_eval)
    a = np.concatenate([x_dev, x_eval])
    n = len(a)
    o = np.argsort(a, kind='stable')
    s = a[o]
    gid = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1
    cnt = np.bincount(gid)
    csum = np.cumsum(cnt)
    avg = (csum - cnt + 1 + csum) / 2.0          # mean of ranks inside each tie group
    r = np.empty(n, np.float64)
    r[o] = avg[gid]
    return float((r[n1:].sum() - n2 * (n2 + 1) / 2.0) / (n1 * n2))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    assert mode in ('pairaggs', 'full')
    t0 = time.time()
    order = json.load(open(MODELS + r'\feature_order.json', encoding='utf-8'))
    cols = order['all']
    files = sorted(glob.glob(OUT + r'\eval_hand_scores_batch_*.parquet'))
    assert len(files) == 20, len(files)
    sc = pl.concat([pl.read_parquet(f) for f in files])
    assert sc.height == 9651820, sc.height
    assert sc.select(KEYS).n_unique() == sc.height, 'duplicate pair-hand'
    ev = pl.read_csv(DATA + r'\evaluation_pairs.csv').select('pair_id', 'shared_hands')
    print('scores', sc.shape, round(time.time() - t0, 1), 's', flush=True)

    # ---------------- 1. pair aggregates (step254_D2_pair.pair_aggs, verbatim)
    sc = sc.sort('pair_id')
    pid = sc['pair_id'].to_numpy()
    P = sc['p_fmax'].to_numpy().astype(np.float64)
    ids, start = np.unique(pid, return_index=True)
    o = np.argsort(start)
    ids, start = ids[o], start[o]
    ends = np.append(start[1:], len(pid))
    A = np.zeros((len(ids), 6), np.float32)
    nh = np.zeros(len(ids), np.int64)
    for i in range(len(ids)):
        pi = P[start[i]:ends[i]]
        nh[i] = len(pi)
        A[i] = D2.pair_aggs(pi, Q99, Q999)
    agg = pl.DataFrame({'pair_id': ids, 'n_hands': nh,
                        **{c: A[:, i] for i, c in enumerate(AGG_NAMES)}})
    agg = agg.join(ev, on='pair_id', how='left')
    assert agg.height == 112540 and (agg['n_hands'] == agg['shared_hands']).all(), \
        'pair hand counts differ from evaluation_pairs.csv'
    agg = agg.drop('shared_hands')
    if not os.path.exists(OUT + r'\eval_pair_mech_aggregates.parquet'):
        agg.write_parquet(OUT + r'\eval_pair_mech_aggregates.parquet')
    print('aggregates', agg.shape, round(time.time() - t0, 1), 's', flush=True)
    if mode == 'pairaggs':
        print('pairaggs only: done', round(time.time() - t0, 1), 's')
        return

    # ---------------- 3b. exceedance over the development honest thresholds, low-risk pairs
    sub = pl.read_csv(CV + r'\outputs_step251_20260917\submission_step251_ens4_dtfp3_on_step222.csv',
                      columns=['pair_id', 'risk_score'])
    low = sub.filter(pl.col('risk_score') < .01).select('pair_id')
    s_low = sc.join(low, on='pair_id', how='semi')
    exceed = {
        'n_low_risk_pairs': low.height,
        'n_low_risk_pair_hands': s_low.height,
        'share_above_q99': round(float((s_low['p_fmax'] > Q99).mean()), 6),
        'share_above_q999': round(float((s_low['p_fmax'] > Q999).mean()), 6),
        'share_above_q99_all_eval': round(float((sc['p_fmax'] > Q99).mean()), 6),
        'share_above_q999_all_eval': round(float((sc['p_fmax'] > Q999).mean()), 6),
        'thresholds': {'q99': Q99, 'q999': Q999,
                       'source': 'results_mechplus.json development out-of-fold honest hands'},
    }
    # the same shares on the development honest hands, as the yardstick
    oof = pl.read_parquet(S254 + r'\hand_oof_step254.parquet',
                          columns=['label', 'mechplus_p_fmax'])
    hon = oof.filter(pl.col('label') == 0)['mechplus_p_fmax'].to_numpy()
    exceed['development_honest_share_above_q99'] = round(float((hon > Q99).mean()), 6)
    exceed['development_honest_share_above_q999'] = round(float((hon > Q999).mean()), 6)
    print('exceedance', json.dumps(exceed), flush=True)
    del s_low, sc
    import gc
    gc.collect()

    # ---------------- 2/3. schema and drift
    cstats = [np.load(SCRATCH + f'\\colstats_b{c:03d}.npz') for c in range(20)]
    cmin = np.min(np.stack([z['cmin'] for z in cstats]), axis=0)
    cmax = np.max(np.stack([z['cmax'] for z in cstats]), axis=0)
    Xe = np.concatenate([np.load(SCRATCH + f'\\driftsample_b{c:03d}.npy') for c in range(20)], axis=0)
    print('eval drift sample', Xe.shape, round(time.time() - t0, 1), 's', flush=True)

    import step254_eval as S254E
    h, feats = S254E.load()
    fx = pl.read_parquet(S254 + r'\step254_features.parquet')
    f2 = pl.read_parquet(S237 + r'\step237_features.parquet')
    h = h.join(f2, on=KEYS, how='left').join(fx, on=KEYS, how='left')
    del fx, f2
    gc.collect()
    Xd = np.empty((h.height, len(cols)), np.float32)
    for i, c in enumerate(cols):
        Xd[:, i] = h[c].to_numpy()
    del h
    gc.collect()
    nd = min(300000, Xd.shape[0])
    rng = np.random.default_rng(257)
    idx = np.sort(rng.choice(Xd.shape[0], nd, replace=False)) if Xd.shape[0] > nd \
        else np.arange(Xd.shape[0])
    Xd = Xd[idx]
    ne = min(300000, Xe.shape[0])
    if Xe.shape[0] > ne:
        Xe = Xe[np.sort(rng.choice(Xe.shape[0], ne, replace=False))]
    print('dev sample', Xd.shape, 'eval sample', Xe.shape, round(time.time() - t0, 1), 's', flush=True)

    drift = {}
    const_dev, const_eval = [], []
    for i, c in enumerate(cols):
        dv = Xd[:, i].astype(np.float64)
        evv = Xe[:, i].astype(np.float64)
        drift[c] = round(auc_from_ranks(dv, evv), 5)
        if dv.min() == dv.max():
            const_dev.append(c)
        if cmin[i] == cmax[i]:
            const_eval.append(c)
        if (i + 1) % 100 == 0:
            print('  drift', i + 1, '/', len(cols), round(time.time() - t0, 1), 's', flush=True)
    top30 = sorted(drift.items(), key=lambda kv: -abs(kv[1] - .5))[:30]
    schema = {'n_columns': len(cols), 'dtype': 'float32',
              'constant_on_evaluation': const_eval,
              'constant_on_development': const_dev,
              'constant_on_evaluation_but_not_on_development':
                  [c for c in const_eval if c not in set(const_dev)],
              'evaluation_min_max_sample': {c: [float(cmin[i]), float(cmax[i])]
                                            for i, c in enumerate(cols)}}

    colagg = sorted(glob.glob(OUT + r'\eval_pair_mech_colaggs_batch_*.parquet'))
    outs = ['eval_pair_mech_aggregates.parquet'] + [os.path.basename(f) for f in files] \
        + [os.path.basename(f) for f in colagg]
    rec = {'step': 'STEP257', 'date': '2026-09-18',
           'rows_scored': 9651820, 'pairs': 112540, 'batches': len(files),
           'models': {os.path.basename(p): sha(p)
                      for p in sorted(glob.glob(MODELS + r'\*.txt')) + [MODELS + r'\feature_order.json',
                                                                       MODELS + r'\fold_tables.json']},
           'fit_gate': json.load(open(MODELS + r'\fit_gate257.json', encoding='utf-8')),
           'parity_gate': json.load(open(OUT + r'\gate_parity257.json', encoding='utf-8')),
           'colagg_parity_gate': (json.load(open(OUT + r'\gate_colagg_parity257.json',
                                                 encoding='utf-8'))
                                  if os.path.exists(OUT + r'\gate_colagg_parity257.json') else None),
           'pair_column_aggregates': {
               'files': len(colagg), 'convention': 'step247 agg_block (mean/std ddof=1/max/q95 '
               'nearest), block order mean|std|max|q95 over the 662 mechanical + 25 STEP237 '
               'columns = 2748 columns, plus pair_id and n_hands, float32',
               'pairs': int(sum(pl.read_parquet(f, columns=['pair_id']).height for f in colagg))},
           'batch_logs': [json.load(open(OUT + f'\\log_batch_{c:03d}.json', encoding='utf-8'))
                          for c in range(20)],
           'exceedance_over_development_thresholds': exceed,
           'schema_gate': {k: v for k, v in schema.items() if k != 'evaluation_min_max_sample'},
           'drift_top30_auc_dev_vs_eval': top30,
           'sha256': {o: sha(OUT + '\\' + o) for o in outs},
           'seconds': round(time.time() - t0, 1)}
    json.dump(rec, open(OUT + r'\receipt257.json', 'w', encoding='utf-8'), indent=1)
    json.dump({'drift_auc_all_columns': drift, 'schema': schema},
              open(OUT + r'\drift257.json', 'w', encoding='utf-8'), indent=1)
    print('TOP DRIFT', json.dumps(top30, indent=1))
    print('done', rec['seconds'], 's')


if __name__ == '__main__':
    main()
