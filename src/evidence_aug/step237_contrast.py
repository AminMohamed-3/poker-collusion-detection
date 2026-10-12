"""STEP237 hypothesis check: invisible listed hands vs honest hands MATCHED on the coarse type key
(players dealt, last street reached, both members acted on the flop, a member folded to the partner).

For each feature: the mean over the invisible listed hands of a family, and the mean over honest
hands re-weighted to the same coarse-key distribution, plus the ratio. This is the conditional
version of the univariate AUC table, so a feature cannot win just by saying 'both members played'.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '1')

import json

import numpy as np
import polars as pl

ROOT = r'C:\Data\Projects\Money Making\work\poker'
OOF = ROOT + r'\Claude Version\experiments\step232_event_detector_20260917\hand_oof.parquet'
OUT = ROOT + r'\Claude Version\experiments\step237_invisible_events_20260917'
INVIS = .0852

from step237_features import FEATURES


def main() -> None:
    oof = pl.read_parquet(OOF)
    fx = pl.read_parquet(OUT + r'\step237_features.parquet')
    ck = pl.read_parquet(OUT + r'\coarse_keys.parquet').select('pair_id', 'hand_id', 'ckey')
    d = oof.join(fx, on=['pair_id', 'hand_id']).join(ck, on=['pair_id', 'hand_id'])
    hon = d.filter(pl.col('label') == 0)
    hon_mean = hon.group_by('ckey').agg(pl.len().alias('n'), *[pl.col(c).mean() for c in FEATURES])
    hm = {r['ckey']: r for r in hon_mean.iter_rows(named=True)}
    res = {}
    for fam in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
        sub = d.filter((pl.col('listed') == 1) & (pl.col('behavior_family') == fam)
                       & (pl.col('p_fmax') < INVIS))
        keys = sub['ckey'].to_list()
        w = np.array([1.0 for _ in keys])
        w /= w.sum()
        row = {}
        for c in FEATURES:
            obs = float(sub[c].mean())
            exp = float(sum(wi * (hm[k][c] if k in hm else 0.) for wi, k in zip(w, keys)))
            row[c] = {'invisible': round(obs, 4), 'honest_matched': round(exp, 4),
                      'ratio': round(obs / exp, 2) if exp > 1e-9 else None}
        res[fam] = {'n_invisible': sub.height, 'features': row}
    json.dump(res, open(OUT + r'\contrast_matched.json', 'w', encoding='utf-8'), indent=1)
    for fam in res:
        print('==', fam, 'n =', res[fam]['n_invisible'])
        for c in FEATURES:
            r = res[fam]['features'][c]
            print(f'  {c:26s} inv={r["invisible"]:9.4f}  honest_matched={r["honest_matched"]:9.4f}'
                  f'  ratio={r["ratio"]}')


if __name__ == '__main__':
    main()
