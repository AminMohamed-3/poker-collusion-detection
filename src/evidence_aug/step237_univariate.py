"""STEP237 diagnostic: univariate AUC of each new feature, invisible-listed vs honest hands.

Also reports visible-listed vs honest for contrast and the per-family event rates.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '1')

import json

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = r'C:\Data\Projects\Money Making\work\poker'
OOF = ROOT + r'\Claude Version\experiments\step232_event_detector_20260917\hand_oof.parquet'
OUT = ROOT + r'\Claude Version\experiments\step237_invisible_events_20260917'
INVIS, VIS = .0852, .4662

from step237_features import FEATURES


def main() -> None:
    oof = pl.read_parquet(OOF)
    fx = pl.read_parquet(OUT + r'\step237_features.parquet')
    d = oof.join(fx, on=['pair_id', 'hand_id'], how='left')
    assert d.height == oof.height and d[FEATURES[0]].null_count() == 0
    hon = d.filter(pl.col('label') == 0)
    rows = {}
    for fam in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
        inv = d.filter((pl.col('listed') == 1) & (pl.col('behavior_family') == fam)
                       & (pl.col('p_fmax') < INVIS))
        vis = d.filter((pl.col('listed') == 1) & (pl.col('behavior_family') == fam)
                       & (pl.col('p_fmax') > VIS))
        for name in FEATURES:
            hv = hon[name].to_numpy().astype(np.float64)
            for grp, sub in (('inv', inv), ('vis', vis)):
                sv = sub[name].to_numpy().astype(np.float64)
                y = np.r_[np.ones(len(sv)), np.zeros(len(hv))]
                s = np.r_[sv, hv]
                auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else .5
                rows.setdefault(name, {})[f'{fam[:2]}_{grp}'] = round(float(auc), 4)
                rows[name][f'{fam[:2]}_{grp}_mean'] = round(float(sv.mean()), 4)
        rows.setdefault('_n', {})[f'{fam[:2]}_inv'] = inv.height
        rows['_n'][f'{fam[:2]}_vis'] = vis.height
    rows['_n']['honest'] = hon.height
    for name in FEATURES:
        rows[name]['honest_mean'] = round(float(hon[name].mean()), 4)
    json.dump(rows, open(OUT + r'\univariate.json', 'w', encoding='utf-8'), indent=1)
    hdr = f'{"feature":26s} {"DT_inv":>7s} {"SP_inv":>7s} {"CI_inv":>7s} | {"DT_vis":>7s} {"SP_vis":>7s} {"CI_vis":>7s} | honest_mean'
    print(hdr)
    for name in FEATURES:
        r = rows[name]
        print(f'{name:26s} {r["di_inv"]:7.4f} {r["so_inv"]:7.4f} {r["co_inv"]:7.4f} | '
              f'{r["di_vis"]:7.4f} {r["so_vis"]:7.4f} {r["co_vis"]:7.4f} | {r["honest_mean"]:.4f}')
    print(rows['_n'])


if __name__ == '__main__':
    main()
