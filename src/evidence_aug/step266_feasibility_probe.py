"""STEP266 feasibility probe (measurement only; nothing is built and no evaluation-phase data is read).

Two measurements for the STEP259b feasibility note:

  1. PREDICT COST.  Time the ten saved production boosters
     (experiments/step257_eval_mech_scores_20260918/models/mechplus_{DT,SP}_fold{0..4}.txt) on a
     50,000-row slice of the DEVELOPMENT 850-column matrix, so that the per-row cost of re-scoring the
     fusion scope with five models per head can be quoted from a measurement rather than a guess.
     (Development rows only - the probe keeps the STEP266 registration's 'no evaluation-phase data'
     promise; the matrix layout is identical in both phases, which is what the timing depends on.)

  2. ARITHMETIC vs GEOMETRIC on the SAME five production models.  Those five boosters are applied to
     development rows they partly trained on, so their probabilities are NOT a valid accuracy
     measurement - but the DISPERSION between the arithmetic and the geometric mean of five models,
     and how it behaves in the low tail after prior_shift(., F=10) ** .25, is a property of the
     averaging and is worth recording alongside the audited arms.  Labelled in-sample; never used as
     evidence for or against the adoption rule.

Usage: python step266_feasibility_probe.py      Writes feasibility_probe266.json.  Never uploads.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True

import lightgbm as lgb                                            # noqa: E402
from evidence_aug import step254_eval as E                        # noqa: E402
from evidence_aug.step266_nested_folds import load_matrix         # noqa: E402

OUT = E.CV + r'\experiments\step266_fold_average_calibration_20260918'
MODELS = E.CV + r'\experiments\step257_eval_mech_scores_20260918\models'
NROW = 50000
NJ = int(os.environ.get('STEP266_NJOBS', '6'))


def shift(p, F=10.0):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    o = q / (1 - q) * F
    return o / (1 + o)


def main() -> None:
    X, M, _ = load_matrix()
    rng = np.random.default_rng(266)
    idx = np.sort(rng.choice(X.shape[0], NROW, replace=False))
    Xs = np.ascontiguousarray(X[idx])
    del X
    rec = {'n_rows_timed': NROW, 'num_threads': NJ, 'heads': {}}
    P = {}
    for h in ('DT', 'SP'):
        bs = [lgb.Booster(model_file=MODELS + f'\\mechplus_{h}_fold{k}.txt') for k in range(5)]
        assert all(b.num_feature() == 850 for b in bs)
        t0 = time.time()
        pp = np.stack([b.predict(Xs, num_threads=NJ) for b in bs])
        dt = time.time() - t0
        P[h] = pp
        rec['heads'][h] = {'seconds_5_models': round(dt, 2),
                           'seconds_per_1k_rows_5_models': round(dt / NROW * 1000, 4),
                           'n_trees': int(bs[0].num_trees())}
        print(h, json.dumps(rec['heads'][h]), flush=True)
    rec['seconds_per_1k_rows_both_heads_5_models'] = round(
        sum(rec['heads'][h]['seconds_5_models'] for h in ('DT', 'SP')) / NROW * 1000, 4)
    # arithmetic vs geometric on the same five production models (IN-SAMPLE; dispersion only)
    ins = {}
    for h in ('DT', 'SP'):
        pp = P[h]
        a = pp.mean(0)
        q = np.clip(pp, 1e-7, 1 - 1e-7)
        g = 1.0 / (1.0 + np.exp(-np.log(q / (1 - q)).mean(0)))
        lo = a <= np.quantile(a, .10)
        ins[h] = {'mean_abs_diff_A5_G5': round(float(np.abs(a - g).mean()), 6),
                  'A5_over_G5_median_low_decile': round(float(np.median(a[lo] / np.maximum(g[lo], 1e-12))), 4),
                  'shift_pow_A5_median_low_decile': round(float(np.median(shift(a[lo]) ** .25)), 5),
                  'shift_pow_G5_median_low_decile': round(float(np.median(shift(g[lo]) ** .25)), 5),
                  'shift_pow_single_fold0_median_low_decile': round(float(np.median(shift(pp[0][lo]) ** .25)), 5),
                  'between_model_sd_median': round(float(np.median(pp.std(0))), 8),
                  'note': 'IN-SAMPLE (these five models trained on 4/5 of these rows each) - dispersion only'}
        print('insample', h, json.dumps(ins[h]), flush=True)
    rec['in_sample_five_model_dispersion'] = ins
    json.dump(rec, open(OUT + r'\feasibility_probe266.json', 'w', encoding='utf-8'), indent=1)
    print('wrote', OUT + r'\feasibility_probe266.json')


if __name__ == '__main__':
    main()
