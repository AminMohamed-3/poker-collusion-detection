"""STEP254: arm runner for the mechanical pair-relative block.

Exactly the step232 recipe and the step232 table folds: listed hands of positive pairs (1) against
every hand of the confirmed non-target pairs (0), unlisted hands of positive pairs excluded from
training, 5-fold cross-fit by table, LightGBM (400 / 15 / .03 / 30 / .8 / 1 / .5 / 5, seed 232),
one model per family plus 'any'.

Arms
  raw       the 163 production per-hand features                       (reproduction)
  plus      raw + the 25 STEP237 features                              (reproduction)
  mech      raw + the STEP254 mechanical block
  mechplus  raw + 25 + the mechanical block
  perm      raw + 25 + the mechanical block permuted across the hands
            of the SAME pair (the STEP237 control)

Development phase only (asserted). Writes results_<arm>.json and oof_<arm>.npy.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
OUT = CV + r'\experiments\step254_mechanical_tensor_20260917'
S237 = CV + r'\experiments\step237_invisible_events_20260917'
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
NJOBS = int(os.environ.get('STEP254_NJOBS', '4'))
P = dict(n_estimators=400, num_leaves=15, learning_rate=.03, min_child_samples=30, subsample=.8,
         subsample_freq=1, colsample_bytree=.5, reg_lambda=5.0, n_jobs=NJOBS, verbosity=-1,
         random_state=232)

F237 = [
    'fp_best_live', 'fp_gap', 'fp_premium_pf', 'fp_price_min', 'fp_wouldwin', 'fp_best_pot',
    'fp_partner_wins', 'fp_betfold', 'fp_limpfold', 'fp_count',
    'iso_out_fold_share', 'iso_clean', 'iso_weak_agg_behind', 'iso_agg_partner_folds',
    'oof_clearway',
    'tr_internal_bb', 'tr_loser_share', 'hu_pair_streets', 'pay_partner_bb', 'passive_winner',
    'checkdown_best', 'coldcall_partner_bb', 'call_partner_dominated_bb', 'both_agg_live',
    'pair_pot_share',
]


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
    ref = pl.read_parquet(CV + r'\experiments\step232_event_detector_20260917'
                          r'\hand_oof.parquet', columns=['pair_id', 'hand_id', 'fold'])
    chk = h.select('pair_id', 'hand_id', 'fold').join(ref, on=['pair_id', 'hand_id'], suffix='_ref')
    assert chk.height == h.height and (chk['fold'] == chk['fold_ref']).all(), 'fold mismatch vs step232'
    return h, feats


def run(tag: str, h: pl.DataFrame, cols: list[str], newset: set[str]) -> dict:
    # column-by-column fill: never materialises a float64 copy of the whole matrix
    X = np.empty((h.height, len(cols)), np.float32)
    for i, c in enumerate(cols):
        X[:, i] = h[c].to_numpy()
    y_listed = h['listed'].to_numpy()
    lbl = h['label'].to_numpy()
    fold = h['fold'].to_numpy()
    fam = h['behavior_family'].to_numpy()
    h = h.select('pair_id')          # release the feature frame
    gc.collect()
    res, oof, gains = {}, {}, {}
    n = X.shape[0]
    for name, posmask in (('any', y_listed == 1), *[(f, (y_listed == 1) & (fam == f)) for f in FAMS]):
        pr = np.full(n, np.nan)
        gain = np.zeros(len(cols))
        t0 = time.time()
        for k in range(5):
            tr = (fold != k) & (posmask | (lbl == 0))
            te = fold == k
            Xtr = X[tr]
            m = lgb.LGBMClassifier(**P).fit(Xtr, posmask[tr].astype(int))
            del Xtr
            gc.collect()
            pr[te] = m.predict_proba(X[te])[:, 1]
            gain += m.booster_.feature_importance('gain')
            del m
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
        order = np.argsort(gains[name])[::-1][:30]
        tot = gains[name].sum()
        top[name] = [(cols[i], round(float(gains[name][i] / tot), 5)) for i in order]
        new_share = sum(gains[name][i] for i, c in enumerate(cols) if c in newset) / tot
        res[name]['new_block_gain_share'] = round(float(new_share), 4)
        # per-family (1)-(5) share of the mechanical block
        byfam = {}
        for i, c in enumerate(cols):
            if c.startswith('t') and len(c) > 2 and c[1].isdigit() and c[2] == '_':
                byfam[c[:2]] = byfam.get(c[:2], 0.) + float(gains[name][i])
        res[name]['mech_family_gain_share'] = {k: round(v / tot, 4) for k, v in sorted(byfam.items())}
    res['_top_gain'] = top
    np.save(OUT + f'\\oof_{tag}.npy', np.stack([oof['any'], *[oof[f] for f in FAMS]]))
    return res


def main() -> None:
    which = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    h, feats = load()
    need_new = which in ('mech', 'mechplus', 'perm', 'mechplus_not4', 'mechplus_t4only')
    MECH: list[str] = []
    if need_new:
        fx = pl.read_parquet(OUT + r'\step254_features.parquet')
        MECH = [c for c in fx.columns if c not in ('pair_id', 'hand_id')]
        h = h.join(fx, on=['pair_id', 'hand_id'], how='left')
        assert h[MECH[0]].null_count() == 0, 'mechanical join left nulls'
    if which in ('plus', 'mechplus', 'perm', 'mechplus_not4', 'mechplus_t4only'):
        f2 = pl.read_parquet(S237 + r'\step237_features.parquet')
        h = h.join(f2, on=['pair_id', 'hand_id'], how='left')
        assert h[F237[0]].null_count() == 0, 'step237 join left nulls'
    t0 = time.time()
    if which == 'raw':
        res = run('raw', h, feats, set())
    elif which == 'plus':
        res = run('plus', h, feats + F237, set(F237))
    elif which == 'mech':
        res = run('mech', h, feats + MECH, set(MECH))
    elif which == 'mechplus':
        res = run('mechplus', h, feats + F237 + MECH, set(F237) | set(MECH))
    elif which == 'mechplus_not4':     # POST HOC: the mechanical block without the chronology family
        keep = [c for c in MECH if not c.startswith('t4_')]
        res = run('mechplus_not4', h, feats + F237 + keep, set(F237) | set(keep))
    elif which == 'mechplus_t4only':   # POST HOC: only the chronology / chip-flow family
        keep = [c for c in MECH if c.startswith('t4_')]
        res = run('mechplus_t4only', h, feats + F237 + keep, set(F237) | set(keep))
    elif which == 'perm':
        rng = np.random.default_rng(1254)
        idx = np.arange(h.height)
        pid = h['pair_id'].to_numpy()
        order = np.argsort(pid, kind='stable')
        starts = np.flatnonzero(np.r_[True, pid[order][1:] != pid[order][:-1]])
        for a, b in zip(starts, np.r_[starts[1:], len(order)]):
            blk = order[a:b].copy()
            rng.shuffle(blk)
            idx[order[a:b]] = blk
        shuffled = h.select(MECH).to_numpy()[idx]
        hp = h.with_columns([pl.Series('perm_' + c, shuffled[:, i]) for i, c in enumerate(MECH)])
        res = run('perm', hp, feats + F237 + ['perm_' + c for c in MECH],
                  set(F237) | {'perm_' + c for c in MECH})
    else:
        raise SystemExit('unknown arm ' + which)
    res['_runtime_s'] = round(time.time() - t0, 1)
    res['_n_cols'] = len(feats) + (len(F237) if which != 'raw' else 0) + len(MECH)
    res['_njobs'] = NJOBS
    json.dump({which: res}, open(OUT + f'\\results_{which}.json', 'w', encoding='utf-8'), indent=1)
    print('wrote', OUT + f'\\results_{which}.json', 'in', res['_runtime_s'], 's')


if __name__ == '__main__':
    main()
