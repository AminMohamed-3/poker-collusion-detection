"""STEP265 (b): per-family stacks of the MECHPLUS event heads (STEP254) with the raw action-sequence
n-gram heads of STEP255 (N_gbm, N_lr), cross-fitted on the existing `fold` column.

Arms (registration265.json, written before any fit):
  A1  LRSTACK    logistic on standardised logits of [mechplus_p_h, p_N_gbm_h, p_N_lr_h]
  A2  LGBSTACK   LightGBM on the 12 logits (three models x four heads)
  C1  PERMSTACK  control: logistic on [mechplus_p_h, p_PERM_h]          (PERM = STEP255 within-pair shuffle)
  C0  MECHONLY   sanity:  logistic on [mechplus_p_h] alone              (monotone recalibration)

Labels and AP are STEP254's exactly (src/evidence_aug/step254_eval.py):
  head 'any'      positives = listed == 1
  head f          positives = (listed == 1) & (family == f)
  negatives       every hand of the confirmed non-target pairs (label == 0)
  training rows   (fold != k) & (posmask | (label == 0))        -- unlisted hands of positive pairs excluded
  evaluation mask posmask | (label == 0)

Development phase only; no evaluation row is read. Writes oof_{A1,A2,C1,C0}.npy (4 x 223,749, rows
any/DT/SP/CI, in the STEP254 parquet's row order) and results265.json.
"""
from __future__ import annotations

import os

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
os.environ.setdefault('POLARS_MAX_THREADS', '4')

import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

CV = r'C:\Data\Projects\Money Making\work\poker\Claude Version'
S254 = CV + r'\experiments\step254_mechanical_tensor_20260917'
S255 = CV + r'\experiments\step255_sequence_view_20260917'
OUT = CV + r'\experiments\step265_seqstack_fusion_20260918'
K = ['pair_id', 'hand_id']
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
HEADS = (('any', None, 'any'), ('DT', 'directed_transfer', 'directed_transfer'),
         ('SP', 'soft_play', 'soft_play'), ('CI', 'coordinated_isolation', 'coordinated_isolation'))
LGB_P = dict(n_estimators=200, num_leaves=15, learning_rate=.05, min_child_samples=100,
             colsample_bytree=1.0, n_jobs=4, verbosity=-1, random_state=0)
LR_P = dict(C=1.0, max_iter=2000, solver='lbfgs')
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(q / (1 - q))


def load() -> tuple[pl.DataFrame, dict]:
    A = pl.read_parquet(S254 + r'\hand_oof_step254.parquet')
    B = pl.read_parquet(S255 + r'\hand_oof_step255.parquet')
    ref = np.load(S254 + r'\oof_mechplus.npy')
    for i, nm in enumerate(('any', 'DT', 'SP', 'CI')):
        assert np.allclose(A[f'mechplus_p_{nm}'].to_numpy(), ref[i], atol=1e-7), nm
    bcols = [f'p_{m}_{f}' for m in ('N_gbm', 'N_lr', 'PERM') for f in ('any', *FAMS)]
    j = (A.select(K + ['fold', 'listed', 'label', 'family'] + [f'mechplus_p_{n}' for n in ('any', 'DT', 'SP', 'CI')])
         .join(B.select(K + ['fold', 'listed', 'label', 'behavior_family'] + bcols),
               on=K, how='left', suffix='_b', validate='1:1', maintain_order='left'))
    assert j.height == A.height and j.select(K).equals(A.select(K)), 'row order / join'
    assert sum(j[c].null_count() for c in bcols) == 0, 'join left nulls'
    assert (j['fold'] == j['fold_b']).all() and (j['listed'] == j['listed_b']).all()
    assert (j['label'] == j['label_b']).all() and (j['family'] == j['behavior_family']).all()
    M = {'fold': j['fold'].to_numpy(), 'listed': j['listed'].to_numpy(),
         'label': j['label'].to_numpy(), 'fam': j['family'].to_numpy()}
    return j, M


def features(j: pl.DataFrame) -> dict:
    """Per head: the three single-model logits; plus the full 12-logit block for A2."""
    L = {}
    for sh, _f, s255 in HEADS:
        L[sh] = {'M': logit(j[f'mechplus_p_{sh}'].to_numpy()),
                 'G': logit(j[f'p_N_gbm_{s255}'].to_numpy()),
                 'R': logit(j[f'p_N_lr_{s255}'].to_numpy()),
                 'P': logit(j[f'p_PERM_{s255}'].to_numpy())}
    X12 = np.column_stack([L[sh][m] for m in ('M', 'G', 'R') for sh, _f, _s in HEADS])
    return L, X12


def fit_arm(arm: str, L: dict, X12: np.ndarray, M: dict) -> tuple[np.ndarray, dict]:
    listed, lbl, fold, fam = M['listed'], M['label'], M['fold'], M['fam']
    n = len(listed)
    clean = lbl == 0
    oof, res = [], {}
    for sh, f, _s in HEADS:
        posmask = (listed == 1) if f is None else ((listed == 1) & (fam == f))
        if arm == 'A2':
            X = X12
        elif arm == 'A1':
            X = np.column_stack([L[sh]['M'], L[sh]['G'], L[sh]['R']])
        elif arm == 'C1':
            X = np.column_stack([L[sh]['M'], L[sh]['P']])
        elif arm == 'C0':
            X = L[sh]['M'][:, None]
        else:
            raise SystemExit('unknown arm ' + arm)
        pr = np.full(n, np.nan)
        t0 = time.time()
        for k in range(5):
            tr = (fold != k) & (posmask | clean)
            te = fold == k
            Xtr, ytr = X[tr], posmask[tr].astype(int)
            if arm == 'A2':
                m = lgb.LGBMClassifier(**LGB_P).fit(Xtr, ytr)
                pr[te] = m.predict_proba(X[te])[:, 1]
            else:
                mu, sd = Xtr.mean(0), Xtr.std(0)
                sd[sd < 1e-12] = 1.0
                m = LogisticRegression(**LR_P).fit((Xtr - mu) / sd, ytr)
                pr[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
            del m
        assert np.isfinite(pr).all()
        ev = posmask | clean
        yy, ss = posmask[ev].astype(int), pr[ev]
        mp = L[sh]['M']
        li_all, li_fam = listed == 1, posmask
        out = {'auc': round(float(roc_auc_score(yy, ss)), 5),
               'ap': round(float(average_precision_score(yy, ss)), 4),
               'ap_mechplus_ref': round(float(average_precision_score(yy, mp[ev])), 4),
               'n_pos': int(yy.sum()), 'n_eval': int(ev.sum()),
               'spearman_vs_mechplus_listed_all': round(float(spearmanr(pr[li_all], mp[li_all]).statistic), 4),
               'spearman_vs_mechplus_listed_head': round(float(spearmanr(pr[li_fam], mp[li_fam]).statistic), 4),
               'spearman_vs_mechplus_clean': round(float(spearmanr(pr[clean], mp[clean]).statistic), 4),
               'seconds': round(time.time() - t0, 1)}
        out['ap_delta_vs_mechplus'] = round(out['ap'] - out['ap_mechplus_ref'], 4)
        res[sh] = out
        oof.append(pr)
        print(arm, sh, json.dumps(out), flush=True)
    return np.stack(oof), res


def main() -> None:
    assert os.path.exists(OUT + r'\registration265.json'), 'register first'
    t0 = time.time()
    j, M = load()
    L, X12 = features(j)
    allres = {'step': 'STEP265',
              'label_definition': "STEP254's: head 'any' positives = listed==1; head f positives = listed==1 & family==f; "
                                  'negatives = every hand of label==0 pairs; training rows (fold!=k)&(posmask|label==0); '
                                  'AP = average_precision_score on the mask posmask|label==0',
              'lr_params': {**LR_P, 'standardised': True, 'clip': EPS},
              'lgb_params': {k: v for k, v in LGB_P.items() if k != 'n_jobs'},
              'n_rows': int(len(M['fold'])), 'arms': {}}
    for arm in ('A1', 'A2', 'C1', 'C0'):
        arr, res = fit_arm(arm, L, X12, M)
        np.save(OUT + f'\\oof_{arm}.npy', arr)
        allres['arms'][arm] = res
    # the reference row, recomputed here with the same code path
    listed, lbl, fam = M['listed'], M['label'], M['fam']
    mref = {}
    for sh, f, _s in HEADS:
        posmask = (listed == 1) if f is None else ((listed == 1) & (fam == f))
        ev = posmask | (lbl == 0)
        yy, ss = posmask[ev].astype(int), L[sh]['M'][ev]
        mref[sh] = {'auc': round(float(roc_auc_score(yy, ss)), 5),
                    'ap': round(float(average_precision_score(yy, ss)), 4), 'n_pos': int(yy.sum())}
    allres['arms']['mechplus'] = mref
    print('mechplus', json.dumps(mref), flush=True)
    allres['_runtime_s'] = round(time.time() - t0, 1)
    json.dump(allres, open(OUT + r'\results265.json', 'w', encoding='utf-8'), indent=1)
    print('wrote results265.json in', allres['_runtime_s'], 's')


if __name__ == '__main__':
    main()
