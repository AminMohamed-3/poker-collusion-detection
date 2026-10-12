"""STEP266 stage 1: the reproduction gate and the NESTED 3-fold MECHPLUS fits for the DT and SP heads.

Why nested: an evaluation-phase hand is out-of-sample for all FIVE production models, so production can
average them (step257_eval_build.py: `acc / 5.0`).  A development fold-k hand is out-of-sample only for
the fold-k model, so the registered OOF cannot be averaged at all.  To put an averaging ensemble on the
162-query audit, every averaged model must be out-of-fold for the rows it scores.  Hence, for each
scoring fold k and each held-out fold j != k, a model fitted on the rows with fold not in {k, j} and
applied to ALL rows of fold k: four out-of-fold probabilities per row from four different 3-fold models.

Recipe: STEP254 MECHPLUS unchanged (850 columns in step257 models/feature_order.json order, step232
table folds, E.P with random_state 232).  Training rows for head f and excluded folds E:
(fold not in E) & ((listed == 1 & family == f) | label == 0).

Usage:  python step266_nested_folds.py gate   DT|SP
        python step266_nested_folds.py nested DT|SP

Writes (new files only, into experiments/step266_fold_average_calibration_20260918):
  gate_<HEAD>.json          refit-vs-registered max abs diff, Spearman, AP of both
  nested_preds_<HEAD>.npy   4 x 223,749 float64, axis 0 = the four held-out folds j in ascending
                            order with j == k skipped, i.e. for a row of fold k the entry [i] is the
                            model that excluded folds {k, sorted(j != k)[i]}
  nested_log_<HEAD>.json    per-fit row counts, APs and timings
Development phase only (asserted inside step254_eval.load()).  Never uploads.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '2')

import gc
import json
import sys
import time

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True

from evidence_aug import step254_eval as E                       # noqa: E402

OUT = E.CV + r'\experiments\step266_fold_average_calibration_20260918'
S257M = E.CV + r'\experiments\step257_eval_mech_scores_20260918\models'
NJ = int(os.environ.get('STEP266_NJOBS', '7'))
P = dict(E.P, n_jobs=NJ)
SHORT = {'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}
LONG = {v: k for k, v in SHORT.items()}
HEADROW = {'any': 0, 'DT': 1, 'SP': 2, 'CI': 3}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def load_matrix():
    """the 850-column MECHPLUS matrix in the STEP254 oof row order, plus the row metadata."""
    h, feats = E.load()
    assert h.height == 223749, h.height
    fx = pl.read_parquet(E.OUT + r'\step254_features.parquet')
    MECH = [c for c in fx.columns if c not in ('pair_id', 'hand_id')]
    h = h.join(fx, on=['pair_id', 'hand_id'], how='left')
    del fx
    f2 = pl.read_parquet(E.S237 + r'\step237_features.parquet')
    h = h.join(f2, on=['pair_id', 'hand_id'], how='left')
    del f2
    cols = feats + E.F237 + MECH
    assert len(cols) == len(set(cols)) == 850, len(cols)
    order = json.load(open(S257M + r'\feature_order.json', encoding='utf-8'))
    assert order['all'] == cols, 'column order differs from the production feature_order.json'
    assert h.select(cols).null_count().to_numpy().sum() == 0, 'join left nulls'
    # the registered npy row order == hand_oof_step254.parquet row order == this row order
    ref_pq = pl.read_parquet(E.OUT + r'\hand_oof_step254.parquet',
                             columns=['pair_id', 'hand_id', 'fold', 'listed', 'label', 'family'])
    assert ref_pq.select('pair_id', 'hand_id').equals(h.select('pair_id', 'hand_id')), \
        'row order differs from hand_oof_step254.parquet'
    assert np.array_equal(ref_pq['fold'].to_numpy(), h['fold'].to_numpy())
    assert np.array_equal(ref_pq['listed'].to_numpy(), h['listed'].to_numpy())
    X = np.empty((h.height, len(cols)), np.float32)
    for i, c in enumerate(cols):
        X[:, i] = h[c].to_numpy()
    M = {'listed': h['listed'].to_numpy(), 'label': h['label'].to_numpy(),
         'fold': h['fold'].to_numpy(), 'fam': h['behavior_family'].to_numpy()}
    del h
    gc.collect()
    return X, M, cols


def ap_auc(pr, posmask, clean):
    ev = posmask | clean
    yy = posmask[ev].astype(int)
    ss = pr[ev]
    return {'ap': round(float(average_precision_score(yy, ss)), 4),
            'auc': round(float(roc_auc_score(yy, ss)), 5), 'n_pos': int(yy.sum()),
            'n_eval': int(ev.sum())}


def gate(head: str) -> None:
    X, M, _ = load_matrix()
    fam_long = LONG[head]
    posmask = (M['listed'] == 1) & (M['fam'] == fam_long)
    clean = M['label'] == 0
    fold = M['fold']
    ref = np.load(E.OUT + r'\oof_mechplus.npy')[HEADROW[head]]
    got = np.full(X.shape[0], np.nan)
    t0 = time.time()
    rows = []
    for k in range(5):
        tr = (fold != k) & (posmask | clean)
        m = lgb.LGBMClassifier(**P).fit(X[tr], posmask[tr].astype(int))
        got[fold == k] = m.predict_proba(X[fold == k])[:, 1]
        rows.append({'fold': k, 'n_train': int(tr.sum()), 'n_pos_train': int((tr & posmask).sum()),
                     's': round(time.time() - t0, 1)})
        print(head, 'gate fold', k, json.dumps(rows[-1]), flush=True)
        del m
        gc.collect()
    assert np.isfinite(got).all()
    d = float(np.abs(got - ref).max())
    rec = {'head': head, 'n_jobs': NJ, 'max_abs_diff_vs_registered': d,
           'exact_1e6': bool(d < 1e-6), 'pass_1e3': bool(d < 1e-3),
           'spearman': round(spearman(got, ref), 8),
           'refit': ap_auc(got, posmask, clean), 'registered': ap_auc(ref, posmask, clean),
           'ap_identical_4dp': None, 'folds': rows, 'seconds': round(time.time() - t0, 1)}
    rec['ap_identical_4dp'] = rec['refit']['ap'] == rec['registered']['ap']
    rec['gate'] = 'PASS_EXACT' if rec['exact_1e6'] else (
        'PASS' if (rec['pass_1e3'] and rec['ap_identical_4dp']) else 'FAIL')
    np.save(OUT + f'\\gate_refit_{head}.npy', got)
    json.dump(rec, open(OUT + f'\\gate_{head}.json', 'w', encoding='utf-8'), indent=1)
    print('GATE', head, rec['gate'], json.dumps({k: rec[k] for k in
          ('max_abs_diff_vs_registered', 'spearman', 'refit', 'registered')}), flush=True)


def nested(head: str) -> None:
    X, M, _ = load_matrix()
    fam_long = LONG[head]
    posmask = (M['listed'] == 1) & (M['fam'] == fam_long)
    clean = M['label'] == 0
    fold = M['fold']
    n = X.shape[0]
    preds = np.full((4, n), np.nan)
    t0 = time.time()
    log = []
    for k in range(5):
        te = fold == k
        for i, j in enumerate([x for x in range(5) if x != k]):     # j ascending, k skipped
            tr = (fold != k) & (fold != j) & (posmask | clean)
            m = lgb.LGBMClassifier(**P).fit(X[tr], posmask[tr].astype(int))
            preds[i, te] = m.predict_proba(X[te])[:, 1]
            log.append({'k': k, 'j': j, 'slot': i, 'n_train': int(tr.sum()),
                        'n_pos_train': int((tr & posmask).sum()), 'n_test': int(te.sum()),
                        's': round(time.time() - t0, 1)})
            print(head, 'nested', json.dumps(log[-1]), flush=True)
            del m
            gc.collect()
    assert np.isfinite(preds).all()
    np.save(OUT + f'\\nested_preds_{head}.npy', preds)
    rec = {'head': head, 'n_jobs': NJ, 'shape': list(preds.shape),
           'slot_order': 'axis 0 index i = the model that also excluded fold sorted([j != k])[i]',
           'per_slot_ap': [ap_auc(preds[i], posmask, clean) for i in range(4)],
           'mean_ap': ap_auc(preds.mean(0), posmask, clean),
           'fits': log, 'seconds': round(time.time() - t0, 1)}
    json.dump(rec, open(OUT + f'\\nested_log_{head}.json', 'w', encoding='utf-8'), indent=1)
    print('NESTED', head, 'done', rec['seconds'], 's', flush=True)


if __name__ == '__main__':
    assert os.path.exists(OUT + r'\registration266.json'), 'register first'
    mode, hd = sys.argv[1], sys.argv[2]
    assert hd in ('DT', 'SP')
    if mode == 'gate':
        gate(hd)
    elif mode == 'nested':
        nested(hd)
    else:
        raise SystemExit('usage: gate|nested DT|SP')
