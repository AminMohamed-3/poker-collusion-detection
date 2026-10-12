"""STEP280: check that the two_sample.py re-implementation of the src/features.py
arithmetic reproduces the PRODUCTION feature cache on real development pair-hands.

Read-only against outputs/hands_development (never written, never modified). Only
label==0 (confirmed non-target) pairs are used, so the check stays inside the honest
population. The four *_residual columns are NOT expected to match: production computes a
player's baseline over that batch's full development history while this study computes it
over its own sampled subset. Everything else must match to float32 precision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_lib as S
import two_sample as T

HERE = Path(__file__).resolve().parent
CACHE = S.ROOT / 'outputs' / 'hands_development'
N_PAIRS = 40


def main():
    lab = pl.read_csv(S.DATA / 'development_labels.csv').filter(pl.col('label') == 0)
    batch = sorted(CACHE.glob('batch_*.parquet'))[0]
    prod = pl.read_parquet(batch)
    keep = [p for p in prod['pair_id'].unique().to_list()
            if p in set(lab['pair_id'].to_list())][:N_PAIRS]
    prod = prod.filter(pl.col('pair_id').is_in(keep))
    pairs = lab.filter(pl.col('pair_id').is_in(keep)).select('pair_id', 'player_1', 'player_2')

    hids = prod['hand_id'].unique().to_list()
    hands = S.load_dev_hands().filter(pl.col('hand_id').is_in(hids))
    seats = S.fetch_rows(hands.select('hand_id'), 'seats')
    acts = S.fetch_rows(hands.select('hand_id'), 'actions')
    s, interactions, _ = T.seat_frame(hands, seats, acts)
    s = s.with_columns([pl.col(c).cast(pl.Float64).fill_null(0) for c in T.PLAYER_COLS])

    left = s.select(['hand_id', 'player_id'] + T.PLAYER_COLS).rename(
        {'player_id': 'player_1', **{c: c + '_1' for c in T.PLAYER_COLS}})
    right = s.select(['hand_id', 'player_id'] + T.PLAYER_COLS).rename(
        {'player_id': 'player_2', **{c: c + '_2' for c in T.PLAYER_COLS}})
    P = pairs.join(left, on='player_1').join(right, on=['player_2', 'hand_id'])
    for side, other in [(1, 2), (2, 1)]:
        inter = interactions.rename(
            {'player_id': f'player_{side}', 'previous_aggressor': f'player_{other}',
             **{c: c + f'_{side}' for c in T.OPP_COLS}})
        P = P.join(inter, on=['hand_id', 'player_1', 'player_2'], how='left')
    P = P.with_columns([pl.col(f'{c}_{i}').cast(pl.Float64).fill_null(0)
                        for c in T.OPP_COLS for i in (1, 2)])
    exprs = []
    for c in T.PLAYER_COLS + T.OPP_COLS:
        c1, c2 = pl.col(c + '_1'), pl.col(c + '_2')
        exprs += [(c1 + c2).alias(c + '_sum'), (c1 - c2).abs().alias(c + '_gap'),
                  pl.min_horizontal(c1, c2).alias(c + '_min'),
                  pl.max_horizontal(c1, c2).alias(c + '_max')]
    exprs += [
        (((pl.col('net_bb_1') > 0) & (pl.col('net_bb_2') < 0)) |
         ((pl.col('net_bb_2') > 0) & (pl.col('net_bb_1') < 0)))
        .cast(pl.Float64).alias('opposite_outcomes'),
        ((pl.col('folded_1') > 0) & (pl.col('folded_2') > 0)).cast(pl.Float64).alias('both_fold'),
        (pl.col('won_share_1') + pl.col('won_share_2')).alias('joint_win_share'),
        (pl.col('hole_strength_1') * pl.col('opp_fold_1') +
         pl.col('hole_strength_2') * pl.col('opp_fold_2')).alias('strong_fold_to_partner'),
        (pl.col('hole_strength_1') * pl.col('late_checks_1') +
         pl.col('hole_strength_2') * pl.col('late_checks_2')).alias('strong_late_checks'),
    ]
    mine = P.with_columns(exprs).select(['pair_id', 'hand_id'] +
                                        [e.meta.output_name() for e in exprs])
    j = prod.join(mine, on=['pair_id', 'hand_id'], suffix='_mine')
    cols = [c for c in mine.columns if c not in ('pair_id', 'hand_id') and c in prod.columns]
    out, mism = {}, []
    for c in cols:
        a = j[c].cast(pl.Float64).to_numpy()
        b = j[c + '_mine'].cast(pl.Float64).to_numpy()
        d = float(np.nanmax(np.abs(a - b)))
        out[c] = round(d, 6)
        if d > 1e-3:
            mism.append(c)
    rep = {'batch': batch.name, 'pairs': len(keep), 'pair_hands_compared': int(j.height),
           'columns_compared': len(cols),
           'columns_matching_to_1e-3': len(cols) - len(mism),
           'columns_not_matching': mism,
           'max_abs_diff_by_column': out}
    print(json.dumps({k: v for k, v in rep.items() if k != 'max_abs_diff_by_column'}, indent=1))
    (HERE / 'feature_parity.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
