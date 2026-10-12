"""Step 23: raw chip-flow ledger for labelled positives vs unlabelled high scorers vs labelled negatives.

Model-free, computed straight from seats.parquet: for every pair, over the hands both were dealt,
  flow_ab = chips lost by A in hands B won, flow_ba = the reverse, imbalance = |flow_ab - flow_ba| / (flow_ab + flow_ba)
  hu_share = fraction of shared hands where the two ended heads-up (both still in at showdown or the last fold)
  partner_win_fold = fraction of shared hands where one folded and the other won the pot
A directed-transfer colluder pair should show a large one-way flow; soft-play should show many heads-up
check-downs / folds to the partner; a false-positive population should look like the negatives.
Output: Claude Version/experiments/pu_pairs_20260911/fold{F}/ledger.json
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
from pu_pairs.build_allpairs_fold import EXP  # noqa: E402

FAM = {1: 'directed_transfer', 2: 'soft_play', 3: 'coordinated_isolation'}


def q(x):
    x = np.asarray(x, dtype=float)
    return [float(v) for v in np.quantile(x, [.1, .5, .9])] if len(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=1)
    args = ap.parse_args()
    out = EXP / f'fold{args.fold}'
    scores = pl.read_parquet(out / 'scores.parquet').filter(pl.col('shared_hands') >= 38)
    pools = sorted(set(scores['table_id']))
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').filter((pl.col('phase') == 'development') & pl.col('table_id').is_in(pools)).select('hand_id', 'big_blind', 'players_at_showdown', 'final_pot')
    seats = pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet').join(hands.lazy(), on='hand_id').collect()
    pairs = scores.select('pair_id', 'player_1', 'player_2')
    a = seats.select('hand_id', 'big_blind', 'players_at_showdown', 'final_pot', pl.col('player_id').alias('player_1'), pl.col('net_chips').alias('net_1'), pl.col('folded').alias('fold_1'), pl.col('won_share').alias('won_1'), pl.col('went_to_showdown').alias('sd_1'))
    b = seats.select('hand_id', pl.col('player_id').alias('player_2'), pl.col('net_chips').alias('net_2'), pl.col('folded').alias('fold_2'), pl.col('won_share').alias('won_2'), pl.col('went_to_showdown').alias('sd_2'))
    ph = pairs.join(a, on='player_1').join(b, on=['player_2', 'hand_id'])
    ph = ph.with_columns(
        pl.when((pl.col('won_2') > 0) & (pl.col('net_1') < 0)).then(-pl.col('net_1') / pl.col('big_blind')).otherwise(0.0).alias('flow_ab'),
        pl.when((pl.col('won_1') > 0) & (pl.col('net_2') < 0)).then(-pl.col('net_2') / pl.col('big_blind')).otherwise(0.0).alias('flow_ba'),
        (~pl.col('fold_1') & ~pl.col('fold_2') & (pl.col('players_at_showdown') <= 2) & (pl.col('sd_1') | pl.col('sd_2'))).cast(pl.Float64).alias('hu_showdown'),
        ((pl.col('fold_1') & (pl.col('won_2') > 0)) | (pl.col('fold_2') & (pl.col('won_1') > 0))).cast(pl.Float64).alias('partner_win_fold'),
        ((pl.col('won_1') > 0) | (pl.col('won_2') > 0)).cast(pl.Float64).alias('either_won'),
        ((pl.col('net_1') + pl.col('net_2')) / pl.col('big_blind')).alias('joint_net_bb'))
    led = ph.group_by('pair_id').agg(pl.len().alias('n'), pl.col('flow_ab').sum(), pl.col('flow_ba').sum(), pl.col('hu_showdown').mean(), pl.col('partner_win_fold').mean(), pl.col('either_won').mean(), pl.col('joint_net_bb').mean(),
                                    (pl.col('flow_ab') > 0).mean().alias('ab_events'), (pl.col('flow_ba') > 0).mean().alias('ba_events'))
    led = led.with_columns(((pl.col('flow_ab') - pl.col('flow_ba')).abs() / (pl.col('flow_ab') + pl.col('flow_ba') + 1e-9)).alias('imbalance'),
                           (pl.max_horizontal('flow_ab', 'flow_ba') / pl.col('n')).alias('max_flow_per_hand_bb'))
    led = led.join(scores.select('pair_id', 'label', 'behavior_family', 'risk', 'predicted_class'), on='pair_id')
    cols = ['imbalance', 'max_flow_per_hand_bb', 'hu_showdown', 'partner_win_fold', 'either_won', 'joint_net_bb', 'ab_events', 'ba_events']
    groups = {'labelled_negatives': led.filter(pl.col('label') == 0), 'unlabelled_lt_0.1': led.filter((pl.col('label') < 0) & (pl.col('risk') < .1))}
    for k, fam in FAM.items():
        groups[f'labelled_pos_{fam}'] = led.filter(pl.col('behavior_family') == fam)
        groups[f'unlabelled_ge_0.9_pred_{fam}'] = led.filter((pl.col('label') < 0) & (pl.col('risk') >= .9) & (pl.col('predicted_class') == k))
        groups[f'unlabelled_0.5_0.9_pred_{fam}'] = led.filter((pl.col('label') < 0) & (pl.col('risk') >= .5) & (pl.col('risk') < .9) & (pl.col('predicted_class') == k))
    res = {g: {'n': f.height, **{c: q(f[c]) for c in cols}} for g, f in groups.items()}
    write_json(out / 'ledger.json', res)
    for g, d in res.items():
        print(g, 'n=', d['n'], 'imbalance q10/50/90=', [round(v, 2) for v in d['imbalance']] if d['imbalance'] else None, 'maxflow/hand=', [round(v, 2) for v in d['max_flow_per_hand_bb']] if d['max_flow_per_hand_bb'] else None,
              'hu_sd=', [round(v, 2) for v in d['hu_showdown']] if d['hu_showdown'] else None, 'pw_fold=', [round(v, 2) for v in d['partner_win_fold']] if d['partner_win_fold'] else None)


if __name__ == '__main__':
    main()
