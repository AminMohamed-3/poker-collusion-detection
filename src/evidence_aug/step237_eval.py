"""STEP237 step 4: does the new block dent the invisible share?

Exactly the step232 recipe and the step232 table folds: listed hands of positive pairs (1) against
every hand of the confirmed non-target pairs (0), 5-fold cross-fit by table, LightGBM
(400 / 15 / .03 / 30 / .8 / 1 / .5 / 5, seed 232), one model per family plus 'any'.
Configs: raw = the 163 production features; plus = raw + the 25 step237 features;
perm = raw + the step237 features permuted across the hands of the same pair (control).
Per family: OOF AUC / AP / TPR at honest false-alarm rates 1e-3 and 1e-2, and the share of listed
hands below the 99% quantile of that model's own honest p_fmax (the 'invisible' share).
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '1')

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
OUT = ROOT + r'\Claude Version\experiments\step237_invisible_events_20260917'
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
P = dict(n_estimators=400, num_leaves=15, learning_rate=.03, min_child_samples=30, subsample=.8,
         subsample_freq=1, colsample_bytree=.5, reg_lambda=5.0, n_jobs=1, verbosity=-1,
         random_state=232)

from step237_features import FEATURES, OUTSIDER_CARD_FEATURES


def load() -> tuple[pl.DataFrame, list[str]]:
    h = pl.read_parquet(ROOT + r'\outputs_v2\hands_development\batch_*.parquet')
    lab = pl.read_csv(DATA + r'\development_labels.csv').select('pair_id', 'label', 'behavior_family')
    ev = (pl.read_csv(DATA + r'\development_evidence.csv').select('pair_id', 'hand_id')
          .with_columns(pl.lit(1).alias('listed')))
    phase = pl.read_parquet(DATA + r'\hands.parquet', columns=['hand_id', 'phase'])
    h = h.join(phase, on='hand_id', how='left')
    assert (h['phase'] == 'development').all(), 'non-development hand in the feature cache'
    h = h.drop('phase')
    h = (h.join(lab, on='pair_id').join(ev, on=['pair_id', 'hand_id'], how='left')
         .with_columns(pl.col('listed').fill_null(0)))
    feats = [c for c in h.columns
             if c not in ('pair_id', 'hand_id', 'table_id', 'label', 'behavior_family', 'listed')]
    tables = h['table_id'].unique().sort().to_list()
    rng = np.random.default_rng(232)
    perm = rng.permutation(len(tables))
    fold_of = {t: int(i % 5) for t, i in zip(tables, np.argsort(perm))}
    h = h.with_columns(pl.col('table_id').replace_strict(fold_of, return_dtype=pl.Int64).alias('fold'))
    ref = pl.read_parquet(ROOT + r'\Claude Version\experiments\step232_event_detector_20260917'
                          r'\hand_oof.parquet', columns=['pair_id', 'hand_id', 'fold'])
    chk = h.select('pair_id', 'hand_id', 'fold').join(ref, on=['pair_id', 'hand_id'], suffix='_ref')
    assert chk.height == h.height and (chk['fold'] == chk['fold_ref']).all(), 'fold mismatch vs step232'
    return h, feats


def run(tag: str, h: pl.DataFrame, cols: list[str]) -> dict:
    X = h.select(cols).to_numpy().astype(np.float32)
    y_listed = h['listed'].to_numpy()
    lbl = h['label'].to_numpy()
    fold = h['fold'].to_numpy()
    fam = h['behavior_family'].to_numpy()
    res, oof, gains = {}, {}, {}
    for name, posmask in (('any', y_listed == 1), *[(f, (y_listed == 1) & (fam == f)) for f in FAMS]):
        pr = np.full(h.height, np.nan)
        gain = np.zeros(len(cols))
        t0 = time.time()
        for k in range(5):
            tr = (fold != k) & (posmask | (lbl == 0))
            te = fold == k
            m = lgb.LGBMClassifier(**P).fit(X[tr], posmask[tr].astype(int))
            pr[te] = m.predict_proba(X[te])[:, 1]
            gain += m.booster_.feature_importance('gain')
        oof[name] = pr
        gains[name] = gain
        ev_mask = posmask | (lbl == 0)
        yy = posmask[ev_mask].astype(int)
        ss = pr[ev_mask]
        neg = np.sort(ss[yy == 0])[::-1]
        out = {'auc': round(float(roc_auc_score(yy, ss)), 5),
               'ap': round(float(average_precision_score(yy, ss)), 4),
               'n_pos': int(yy.sum()), 'seconds': round(time.time() - t0, 1)}
        for fpr in (1e-4, 1e-3, 1e-2):
            thr = neg[int(fpr * len(neg))]
            out[f'tpr@fpr{fpr}'] = round(float((ss[yy == 1] > thr).mean()), 4)
        res[name] = out
        print(tag, name, out, flush=True)
    pmax = np.max(np.stack([oof[f] for f in FAMS]), axis=0)
    honest = pmax[lbl == 0]
    q99, q999 = float(np.quantile(honest, .99)), float(np.quantile(honest, .999))
    res['_thresholds'] = {'honest_q99': round(q99, 5), 'honest_q999': round(q999, 5)}
    for f in FAMS:
        sel = (y_listed == 1) & (fam == f)
        res[f]['invisible_share'] = round(float((pmax[sel] < q99).mean()), 4)
        res[f]['n_listed'] = int(sel.sum())
    res['any']['invisible_share'] = round(float((pmax[y_listed == 1] < q99).mean()), 4)
    top = {}
    for name in gains:
        order = np.argsort(gains[name])[::-1][:12]
        tot = gains[name].sum()
        top[name] = [(cols[i], round(float(gains[name][i] / tot), 4)) for i in order]
        newset = set(FEATURES) | {'perm_' + c for c in FEATURES}
        new_share = sum(gains[name][i] for i, c in enumerate(cols) if c in newset) / tot
        res[name]['new_block_gain_share'] = round(float(new_share), 4)
    res['_top_gain'] = top
    np.save(OUT + f'\\oof_{tag}.npy', np.stack([oof['any'], *[oof[f] for f in FAMS]]))
    return res


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    h, feats = load()
    fx = pl.read_parquet(OUT + r'\step237_features.parquet')
    h = h.join(fx, on=['pair_id', 'hand_id'], how='left')
    assert h[FEATURES[0]].null_count() == 0, 'feature join left nulls'
    results = {}
    path = OUT + f'\\results_{which}.json'
    if which in ('all', 'raw'):
        results['raw'] = run('raw', h, feats)
    if which in ('all', 'plus'):
        results['plus'] = run('plus', h, feats + FEATURES)
    if which == 'noout':  # can production do without the outsiders' hole cards?
        keep = [c for c in FEATURES if c not in OUTSIDER_CARD_FEATURES]
        results['noout'] = run('noout', h, feats + keep)
    if which == 'iso1':  # how much of the gain is the single top feature?
        results['iso1'] = run('iso1', h, feats + ['iso_weak_agg_behind'])
    if which in ('all', 'perm'):
        rng = np.random.default_rng(1237)
        idx = np.arange(h.height)
        pid = h['pair_id'].to_numpy()
        order = np.argsort(pid, kind='stable')
        starts = np.flatnonzero(np.r_[True, pid[order][1:] != pid[order][:-1]])
        for a, b in zip(starts, np.r_[starts[1:], len(order)]):
            blk = order[a:b].copy()
            rng.shuffle(blk)
            idx[order[a:b]] = blk
        shuffled = h.select(FEATURES).to_numpy()[idx]
        hp = h.with_columns([pl.Series('perm_' + c, shuffled[:, i])
                             for i, c in enumerate(FEATURES)])
        results['perm'] = run('perm', hp, feats + ['perm_' + c for c in FEATURES])
    json.dump(results, open(path, 'w', encoding='utf-8'), indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
