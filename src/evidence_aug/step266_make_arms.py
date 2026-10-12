"""STEP266 stage 2: turn the nested 3-fold predictions into the arm files, and score their head AP.

Arms (each a 4 x 223,749 float64 array in the STEP254 oof row order, rows any / DT / SP / CI, with the
'any' and 'CI' rows copied unchanged from the registered oof_mechplus.npy so that only the two fused
heads differ between arms):

  REF        the registered oof_mechplus.npy itself (one 4-fold model per scoring fold)
  S3         the single nested model with j = (k + 1) mod 5          (capacity control)
  A4         arithmetic mean of the four nested probabilities        (the production form)
  G4         geometric mean = sigmoid(mean of logits), p clipped to [1e-7, 1 - 1e-7]
  M4         median of the four nested probabilities
  SINGLE_j0..SINGLE_j3   each nested slot on its own; their four fusion deltas are averaged in the
             report as the single-model reference level (A4_S3AVG in the brief)

Also writes arms266.json: head AP / AUC per arm on the STEP254 evaluation mask
(listed-of-family-f | label == 0), plus the pairwise mean absolute probability difference between
REF, S3, A4 and G4 and the low-tail behaviour that motivates the study.

Usage: python step266_make_arms.py       Development only.  Never uploads.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = r'C:\Data\Projects\Money Making\work\poker'
CV = ROOT + r'\Claude Version'
S254 = CV + r'\experiments\step254_mechanical_tensor_20260917'
OUT = CV + r'\experiments\step266_fold_average_calibration_20260918'
HEADS = ('DT', 'SP')
ROW = {'any': 0, 'DT': 1, 'SP': 2, 'CI': 3}
LONG = {'DT': 'directed_transfer', 'SP': 'soft_play'}
CLIP = 1e-7


def logit(p):
    q = np.clip(p, CLIP, 1 - CLIP)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    assert os.path.exists(OUT + r'\registration266.json'), 'register first'
    ref = np.load(S254 + r'\oof_mechplus.npy')
    assert ref.shape == (4, 223749), ref.shape
    meta = pl.read_parquet(S254 + r'\hand_oof_step254.parquet',
                           columns=['fold', 'listed', 'label', 'family'])
    for i, nm in enumerate(('any', 'DT', 'SP', 'CI')):
        col = pl.read_parquet(S254 + r'\hand_oof_step254.parquet',
                              columns=[f'mechplus_p_{nm}'])[f'mechplus_p_{nm}'].to_numpy()
        assert np.allclose(col, ref[i], atol=1e-7), nm
    fold = meta['fold'].to_numpy()
    listed = meta['listed'].to_numpy()
    lbl = meta['label'].to_numpy()
    fam = meta['family'].to_numpy()
    n = len(fold)

    nested = {h: np.load(OUT + f'\\nested_preds_{h}.npy') for h in HEADS}
    for h in HEADS:
        assert nested[h].shape == (4, n) and np.isfinite(nested[h]).all()

    # slot of the j = (k + 1) mod 5 model, per scoring fold
    slot_of_k = {}
    for k in range(5):
        js = [x for x in range(5) if x != k]
        slot_of_k[k] = js.index((k + 1) % 5)

    arms: dict[str, dict[str, np.ndarray]] = {'REF': {h: ref[ROW[h]].copy() for h in HEADS}}
    for h in HEADS:
        p4 = nested[h]
        s3 = np.empty(n)
        for k in range(5):
            m = fold == k
            s3[m] = p4[slot_of_k[k], m]
        arms.setdefault('S3', {})[h] = s3
        arms.setdefault('A4', {})[h] = p4.mean(axis=0)
        arms.setdefault('G4', {})[h] = sigmoid(logit(p4).mean(axis=0))
        arms.setdefault('M4', {})[h] = np.median(p4, axis=0)
        for i in range(4):
            arms.setdefault(f'SINGLE_j{i}', {})[h] = p4[i].copy()

    rec = {'clip': CLIP, 'slot_of_k_for_S3': slot_of_k, 'arms': {}}
    for a, d in arms.items():
        arr = ref.copy()
        for h in HEADS:
            arr[ROW[h]] = d[h]
        np.save(OUT + f'\\oof_{a}.npy', arr)
        rec['arms'][a] = {}
        for h in HEADS:
            pos = (listed == 1) & (fam == LONG[h])
            ev = pos | (lbl == 0)
            yy = pos[ev].astype(int)
            ss = d[h][ev]
            neg = np.sort(ss[yy == 0])[::-1]
            rec['arms'][a][h] = {
                'ap': round(float(average_precision_score(yy, ss)), 4),
                'auc': round(float(roc_auc_score(yy, ss)), 5),
                **{f'tpr@fpr{q}': round(float((ss[yy == 1] > neg[int(q * len(neg))]).mean()), 4)
                   for q in (1e-4, 1e-3, 1e-2)},
                'p_mean': round(float(d[h].mean()), 6),
                'p_q01': round(float(np.quantile(d[h], .01)), 8),
                'p_q50': round(float(np.quantile(d[h], .50)), 8),
                'p_q99': round(float(np.quantile(d[h], .99)), 6),
                'clean_p_mean': round(float(d[h][lbl == 0].mean()), 6),
                'clean_p_q50': round(float(np.quantile(d[h][lbl == 0], .50)), 8)}

    # how far apart the arms are, and what the averaging does to the low tail
    cmp = {}
    for h in HEADS:
        base = arms['REF'][h]
        for a in ('S3', 'A4', 'G4', 'M4'):
            v = arms[a][h]
            cmp[f'{h}:{a}_vs_REF'] = {
                'mean_abs_diff': round(float(np.abs(v - base).mean()), 6),
                'spearman_like_corr': round(float(np.corrcoef(np.argsort(np.argsort(v)),
                                                              np.argsort(np.argsort(base)))[0, 1]), 6),
                'median_ratio_low_decile': round(float(
                    np.median(v[base <= np.quantile(base, .10)]
                              / np.maximum(base[base <= np.quantile(base, .10)], 1e-12))), 4)}
        cmp[f'{h}:A4_vs_G4'] = {
            'mean_abs_diff': round(float(np.abs(arms['A4'][h] - arms['G4'][h]).mean()), 6),
            'A4_over_G4_median_low_decile': round(float(np.median(
                arms['A4'][h][arms['A4'][h] <= np.quantile(arms['A4'][h], .10)]
                / np.maximum(arms['G4'][h][arms['A4'][h] <= np.quantile(arms['A4'][h], .10)],
                             1e-12))), 4)}
    rec['arm_distance'] = cmp
    json.dump(rec, open(OUT + r'\arms266.json', 'w', encoding='utf-8'), indent=1)
    for a in rec['arms']:
        print(a, json.dumps({h: {k: rec['arms'][a][h][k] for k in ('ap', 'auc', 'clean_p_q50')}
                             for h in HEADS}), flush=True)
    print('wrote', OUT + r'\arms266.json')


if __name__ == '__main__':
    sys.exit(main())
