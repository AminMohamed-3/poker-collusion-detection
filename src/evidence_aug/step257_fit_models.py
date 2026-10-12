"""STEP257 part 1: fit and SAVE the five step232-fold MECHPLUS event models per head.

Development phase only (asserted, through step254_eval.load()).  The recipe, the row set, the fold
assignment and the column order all come from STEP254 unchanged - this file only re-runs the
MECHPLUS arm's fits and writes the boosters to disk so that any later step (this one, or the pair
agent's) can score the evaluation phase without refitting.

Reproduction gate: fold-k model applied to the fold-k development rows must reproduce the
mechplus_p_<head> column of hand_oof_step254.parquet (target max abs < 1e-6; if threading makes the
refit non-bit-exact the difference is reported and Spearman > .9999 is required).

Writes
  models/mechplus_{any,DT,SP,CI}_fold{0..4}.txt
  models/feature_order.json
  models/fold_tables.json
  models/fit_gate257.json
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import json
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True

import lightgbm as lgb                                   # noqa: E402
import step254_eval as S254                              # noqa: E402

ROOT = r'C:\Data\Projects\Money Making\work\poker'
CV = ROOT + r'\Claude Version'
S254DIR = CV + r'\experiments\step254_mechanical_tensor_20260917'
OUT = CV + r'\experiments\step257_eval_mech_scores_20260918'
MODELS = OUT + r'\models'
FAMS = S254.FAMS
SHORT = {'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main() -> None:
    assert os.path.exists(OUT + r'\registration257.json'), 'register first'
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.time()
    h, feats = S254.load()                      # asserts development phase on every hand
    assert h.height == 223749, h.height
    fx = pl.read_parquet(S254DIR + r'\step254_features.parquet')
    MECH = [c for c in fx.columns if c not in ('pair_id', 'hand_id')]
    assert len(MECH) == 662, len(MECH)
    h = h.join(fx, on=['pair_id', 'hand_id'], how='left')
    del fx
    f2 = pl.read_parquet(S254DIR.replace('step254_mechanical_tensor_20260917',
                                         'step237_invisible_events_20260917')
                         + r'\step237_features.parquet')
    h = h.join(f2, on=['pair_id', 'hand_id'], how='left')
    del f2
    cols = feats + S254.F237 + MECH
    assert len(cols) == 850, len(cols)
    assert len(set(cols)) == 850, 'duplicate column name'
    assert h.select(cols).null_count().to_numpy().sum() == 0, 'join left nulls'

    json.dump({'n': len(cols), 'all': cols,
               'block_163_production': feats, 'block_25_step237': list(S254.F237),
               'block_662_mechanical': MECH,
               'source': {'163': 'outputs_v2/hands_* batch column order',
                          '25': 'evidence_aug.step237_features.FEATURES order (step254_eval.F237)',
                          '662': 'step254_features.parquet column order (degeneracy prune frozen '
                                 'from the development build)'}},
              open(MODELS + r'\feature_order.json', 'w', encoding='utf-8'), indent=1)
    ft = dict(h.select('table_id', 'fold').unique().sort('table_id').iter_rows())
    json.dump({'table_to_fold': ft,
               'fold_sizes': {str(k): int((h['fold'] == k).sum()) for k in range(5)},
               'n_tables': len(ft),
               'source': 'experiments/step232_event_detector_20260917/hand_oof.parquet (asserted '
                         'row by row inside step254_eval.load)'},
              open(MODELS + r'\fold_tables.json', 'w', encoding='utf-8'), indent=1)
    print('wrote feature_order.json and fold_tables.json', round(time.time() - t0, 1), 's', flush=True)

    X = np.empty((h.height, len(cols)), np.float32)
    for i, c in enumerate(cols):
        X[:, i] = h[c].to_numpy()
    y_listed = h['listed'].to_numpy()
    lbl = h['label'].to_numpy()
    fold = h['fold'].to_numpy()
    fam = h['behavior_family'].to_numpy()
    keys = h.select('pair_id', 'hand_id')
    del h
    import gc
    gc.collect()
    print('matrix', X.shape, round(time.time() - t0, 1), 's', flush=True)

    ref = pl.read_parquet(S254DIR + r'\hand_oof_step254.parquet')
    ref = keys.join(ref, on=['pair_id', 'hand_id'], how='left', validate='1:1',
                    maintain_order='left')
    assert ref.select('pair_id', 'hand_id').equals(keys), 'oof reference lost the row order'

    gate = {'n_jobs': S254.NJOBS, 'params': {k: v for k, v in S254.P.items()}, 'heads': {}}
    heads = [('any', y_listed == 1)] + [(SHORT[f], (y_listed == 1) & (fam == f)) for f in FAMS]
    for name, posmask in heads:
        got = np.full(len(lbl), np.nan)
        for k in range(5):
            tr = (fold != k) & (posmask | (lbl == 0))
            te = fold == k
            m = lgb.LGBMClassifier(**S254.P).fit(X[tr], posmask[tr].astype(int))
            m.booster_.save_model(MODELS + f'\\mechplus_{name}_fold{k}.txt')
            got[te] = m.predict_proba(X[te])[:, 1]
            del m
            gc.collect()
            print(f'  {name} fold {k} saved  n_train {int(tr.sum())}  '
                  f'{time.time() - t0:.0f}s', flush=True)
        want = ref[f'mechplus_p_{name}'].to_numpy().astype(np.float64)
        d = float(np.abs(got - want).max())
        gate['heads'][name] = {'max_abs_diff_vs_step254_oof': d,
                               'spearman': round(spearman(got, want), 8),
                               'exact': bool(d < 1e-6), 'n_pos': int(posmask.sum())}
        print(name, gate['heads'][name], flush=True)
    gate['G_repro'] = 'PASS_EXACT' if all(v['exact'] for v in gate['heads'].values()) else (
        'PASS_SPEARMAN' if all(v['spearman'] > .9999 for v in gate['heads'].values()) else 'FAIL')
    gate['seconds'] = round(time.time() - t0, 1)
    json.dump(gate, open(MODELS + r'\fit_gate257.json', 'w', encoding='utf-8'), indent=1)
    print('G_repro', gate['G_repro'], gate['seconds'], 's')


if __name__ == '__main__':
    main()
