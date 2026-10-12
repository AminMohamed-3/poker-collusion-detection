"""STEP255 step 3 -- base arms: plus (163 raw + 25 step237), N_lr (tf-idf n-grams + SGD logistic),
N_gbm (LightGBM on the top 3,000 n-grams by training-fold document frequency).

For every (arm, family) the script produces
  * loo[N]      the ordinary out-of-fold prediction (model trained on the 4 folds != k predicts k)
  * lto[5, N]   leave-TWO-out predictions: lto[k, i] for a row i in fold j != k comes from the model
                trained on the 3 folds that are neither j nor k, so it never saw fold k. These are
                the stacker's training features for outer fold k (registration255.json).
Development phase only; folds are the step232 table folds, reused verbatim.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '5')

import itertools
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
OUT = ROOT + r'\Claude Version\experiments\step255_sequence_view_20260917'
S232 = ROOT + r'\Claude Version\experiments\step232_event_detector_20260917\hand_oof.parquet'
S237 = ROOT + r'\Claude Version\experiments\step237_invisible_events_20260917\step237_features.parquet'
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
MODELS = ('any',) + FAMS
LGB_P = dict(n_estimators=400, num_leaves=15, learning_rate=.03, min_child_samples=30, subsample=.8,
             subsample_freq=1, colsample_bytree=.5, reg_lambda=5.0, n_jobs=5, verbosity=-1,
             random_state=232)
ALPHAS = (1e-6, 1e-5, 1e-4)
TOPK_GBM = 3000


def load():
    order = pl.read_parquet(OUT + r'\row_order255.parquet').with_row_index('rid')
    oof = pl.read_parquet(S232)
    j = order.join(oof, on=['pair_id', 'hand_id'], how='left').sort('rid')
    assert j.height == order.height and j['fold'].null_count() == 0
    # production 163 + step237 25, in the same row order
    h = pl.read_parquet(ROOT + r'\outputs_v2\hands_development\batch_*.parquet')
    phase = pl.read_parquet(DATA + r'\hands.parquet', columns=['hand_id', 'phase'])
    h = h.join(phase, on='hand_id', how='left')
    assert (h['phase'] == 'development').all(), 'non-development hand in the feature cache'
    h = h.drop('phase')
    feats = [c for c in h.columns if c not in ('pair_id', 'hand_id', 'table_id')]
    fx = pl.read_parquet(S237)
    f25 = [c for c in fx.columns if c not in ('pair_id', 'hand_id')]
    d = (order.join(h, on=['pair_id', 'hand_id'], how='left')
         .join(fx, on=['pair_id', 'hand_id'], how='left').sort('rid'))
    assert d.height == order.height, 'plus feature join changed the row count'
    assert d[feats[0]].null_count() == 0 and d[f25[0]].null_count() == 0, 'plus join left nulls'
    Xp = d.select(feats + f25).to_numpy().astype(np.float32)
    X = sparse.load_npz(OUT + r'\ngrams255.npz').tocsr()
    assert X.shape[0] == order.height
    meta = dict(fold=j['fold'].to_numpy(), lbl=j['label'].to_numpy(),
                listed=j['listed'].to_numpy(), fam=j['behavior_family'].to_numpy(),
                pair=j['pair_id'].to_numpy(), hand=j['hand_id'].to_numpy(),
                table=j['table_id'].to_numpy())
    return meta, Xp, X, feats + f25


def fit_predict(arm, Xp, Xcsc, X, tr_idx, y_tr, te_idx, fold_tr):
    if arm == 'plus':
        m = lgb.LGBMClassifier(**LGB_P).fit(Xp[tr_idx], y_tr)
        return m.predict_proba(Xp[te_idx])[:, 1], None
    if arm == 'N_gbm':
        Xtr_all = X[tr_idx]
        dfc = Xtr_all.getnnz(axis=0)
        cols = np.sort(np.argsort(dfc)[::-1][:TOPK_GBM])
        sub = Xcsc[:, cols].tocsr()
        m = lgb.LGBMClassifier(**LGB_P).fit(sub[tr_idx], y_tr)
        return m.predict_proba(sub[te_idx])[:, 1], cols
    if arm == 'N_lr':
        tf = TfidfTransformer(sublinear_tf=True, norm='l2').fit(X[tr_idx])
        A = tf.transform(X[tr_idx])
        B = tf.transform(X[te_idx])
        inner = fold_tr.max()
        iv = fold_tr == inner
        best, best_ap = ALPHAS[0], -1.
        if iv.sum() > 0 and y_tr[~iv].sum() > 0 and y_tr[iv].sum() > 0:
            for al in ALPHAS:
                s = SGDClassifier(loss='log_loss', penalty='l2', alpha=al, class_weight='balanced',
                                  max_iter=25, tol=1e-4, random_state=255).fit(A[~iv], y_tr[~iv])
                ap = average_precision_score(y_tr[iv], s.predict_proba(A[iv])[:, 1])
                if ap > best_ap:
                    best, best_ap = al, ap
        s = SGDClassifier(loss='log_loss', penalty='l2', alpha=best, class_weight='balanced',
                          max_iter=25, tol=1e-4, random_state=255).fit(A, y_tr)
        return s.predict_proba(B)[:, 1], best
    raise ValueError(arm)


def main() -> None:
    t0 = time.time()
    arms = sys.argv[1:] or ['plus', 'N_lr', 'N_gbm']
    meta, Xp, X, cols = load()
    N = len(meta['fold'])
    print('rows', N, 'plus features', Xp.shape[1], 'ngram features', X.shape[1], flush=True)
    Xcsc = X.tocsc()
    fold, lbl, listed, fam = meta['fold'], meta['lbl'], meta['listed'], meta['fam']
    store = {}
    notes = []
    for arm in arms:
        for mdl in MODELS:
            pos = (listed == 1) if mdl == 'any' else ((listed == 1) & (fam == mdl))
            elig = pos | (lbl == 0)
            loo = np.full(N, np.nan, dtype=np.float64)
            lto = np.full((5, N), np.nan, dtype=np.float64)
            for k in range(5):
                tr = np.flatnonzero(elig & (fold != k))
                te = np.flatnonzero(fold == k)
                p, note = fit_predict(arm, Xp, Xcsc, X, tr, pos[tr].astype(int), te, fold[tr])
                loo[te] = p
                if arm == 'N_lr':
                    notes.append((arm, mdl, 'loo', k, note))
            for j, k in itertools.combinations(range(5), 2):
                tr = np.flatnonzero(elig & (fold != j) & (fold != k))
                te = np.flatnonzero((fold == j) | (fold == k))
                p, _ = fit_predict(arm, Xp, Xcsc, X, tr, pos[tr].astype(int), te, fold[tr])
                pj = te[fold[te] == j]
                pk = te[fold[te] == k]
                lto[k, pj] = p[fold[te] == j]   # used when the outer test fold is k
                lto[j, pk] = p[fold[te] == k]
            store[f'loo_{arm}_{mdl}'] = loo.astype(np.float32)
            store[f'lto_{arm}_{mdl}'] = lto.astype(np.float32)
            m = elig
            from sklearn.metrics import roc_auc_score
            print(f'{arm} {mdl} oof AUC {roc_auc_score(pos[m], loo[m]):.5f} '
                  f'AP {average_precision_score(pos[m], loo[m]):.4f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(OUT + r'\preds255_' + '_'.join(arms) + '.npz', **store)
    if notes:
        print('N_lr alphas', notes[:20], flush=True)
    print('DONE', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
