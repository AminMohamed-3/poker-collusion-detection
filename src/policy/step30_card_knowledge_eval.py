"""Step 30: post-fold card-knowledge statistic on the EVALUATION pairs (label-free hidden-family search).

Motivation (2026-09-11): probe_pair_only scored .65679 public, so public PairAP = .938 against .9999 on labelled
development pairs. Development labels contain only the three disclosed families; the fourth (other_coordination)
has no development examples, so the development nulls of 13.1 and 14.2 say nothing about it. This runs the exact
step-26 statistic (same functions, same buckets, same shrinkage) on the 800k evaluation hands, where the hidden
family must live if it exists, and compares the tail with the N(0,1) null calibrated on development.

Pre-registered before the run:
  flag rule (direction card-sharing predicts): z_fold >= 4.5 (folds more when outs are blocked, rows facing a bet)
     or z_agg <= -4.5 (bets less when outs are blocked). Null expectation per side = n_directed * 3.4e-6.
  population claim requires flagged directed pairs with v13 risk < .5 to exceed 10x the null expectation.
  If claimed, a candidate may raise those pairs' risk; if not, this line is closed for the evaluation set too.
Outputs: experiments/card_knowledge_20260911/eval-directed.parquet, eval-pairs.parquet, eval-summary.json
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
from policy.step26_card_knowledge import outs_mask, CARDS, OUT
import numpy as np
import polars as pl
from scipy.stats import norm


def main():
    tag = 'eval'
    t0 = time.perf_counter()
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').filter(pl.col('phase') == 'evaluation').select('hand_id', 'table_id', 'board_cards')
    acts = pl.scan_parquet(ORIGINAL / 'data' / 'actions.parquet').join(hands.lazy().select('hand_id'), on='hand_id').collect()
    seats = pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet').join(hands.lazy().select('hand_id'), on='hand_id').select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2').collect()
    seats = seats.with_columns(pl.col('hole_card_1').replace_strict(CARDS, return_dtype=pl.Int32).alias('c1'),
                               pl.col('hole_card_2').replace_strict(CARDS, return_dtype=pl.Int32).alias('c2')).drop('hole_card_1', 'hole_card_2')
    dec = acts.filter(pl.col('street').is_in(['flop', 'turn']) & pl.col('action').is_in(['fold', 'check', 'call', 'bet', 'raise', 'all_in']))
    dec = dec.with_columns(pl.when(pl.col('street') == 'flop').then(1).otherwise(2).alias('st'),
                           (pl.col('to_call') / (pl.col('pot_before') + pl.col('to_call')).clip(lower_bound=1)).alias('price'),
                           (pl.col('action') == 'fold').cast(pl.Int8).alias('y_fold'),
                           pl.col('action').is_in(['bet', 'raise', 'all_in']).cast(pl.Int8).alias('y_agg'))
    dec = dec.join(seats, on=['hand_id', 'player_id']).join(hands.select('hand_id', 'table_id', 'board_cards'), on='hand_id')
    u = dec.select('hand_id', 'player_id', 'st', 'c1', 'c2', 'board_cards').unique()
    print(tag, 'decisions', dec.height, 'unique states', u.height, 'load s', round(time.perf_counter() - t0, 1), flush=True)
    masks = [0] * u.height
    nouts = np.zeros(u.height, dtype=np.int16)
    ranks = np.zeros(u.height, dtype=np.int16)
    t1 = time.perf_counter()
    for i, (c1, c2, st, bc) in enumerate(zip(u['c1'].to_list(), u['c2'].to_list(), u['st'].to_list(), u['board_cards'].to_list())):
        board = [CARDS[x] for x in bc.split()][:3 + (st - 1)]
        m, n, r = outs_mask((c1, c2), board)
        masks[i] = m
        nouts[i] = n
        ranks[i] = r
        if i % 100000 == 0:
            print('  outs', i, round(time.perf_counter() - t1, 1), flush=True)
    u = u.with_columns(pl.Series('mask_lo', [m & 0xFFFFFFFF for m in masks], dtype=pl.UInt32), pl.Series('mask_hi', [m >> 32 for m in masks], dtype=pl.UInt32),
                       pl.Series('n_outs', nouts), pl.Series('rank', ranks))
    print(tag, 'outs done', round(time.perf_counter() - t1, 1), flush=True)
    dec = dec.join(u.select('hand_id', 'player_id', 'st', 'mask_lo', 'mask_hi', 'n_outs', 'rank'), on=['hand_id', 'player_id', 'st'])
    dec = dec.with_columns(pl.when(pl.col('price') <= 0).then(0).when(pl.col('price') <= .15).then(1).when(pl.col('price') <= .3).then(2).when(pl.col('price') <= .5).then(3).otherwise(4).alias('pb'),
                           pl.when(pl.col('rank') <= 1609).then(0).when(pl.col('rank') <= 3325).then(1).when(pl.col('rank') <= 6185).then(2).otherwise(3).alias('sb'),
                           pl.when(pl.col('n_outs') == 0).then(0).when(pl.col('n_outs') <= 4).then(1).when(pl.col('n_outs') <= 8).then(2).otherwise(3).alias('ob'))
    keys = ['st', 'pb', 'sb', 'ob']
    for y in ['y_fold', 'y_agg']:
        g = dec.group_by(keys).agg(pl.col(y).mean().alias(f'{y}_glob'))
        o = dec.group_by(['player_id', *keys]).agg(pl.col(y).sum().alias('k'), pl.len().alias('n'))
        dec = dec.join(g, on=keys).join(o, on=['player_id', *keys]).with_columns(((pl.col('k') + 10 * pl.col(f'{y}_glob')) / (pl.col('n') + 10)).alias(f'{y}_hat')).drop('k', 'n')
    folds = acts.filter(pl.col('action') == 'fold').select('hand_id', pl.col('player_id').alias('partner'), pl.col('action_no').alias('fold_no'))
    rows = dec.select('hand_id', 'table_id', 'player_id', 'action_no', 'st', 'price', 'y_fold', 'y_agg', 'y_fold_hat', 'y_agg_hat', 'mask_lo', 'mask_hi', 'n_outs')
    rows = rows.join(folds, on='hand_id').filter((pl.col('fold_no') < pl.col('action_no')) & (pl.col('partner') != pl.col('player_id')))
    rows = rows.join(seats.select('hand_id', pl.col('player_id').alias('partner'), pl.col('c1').alias('b1'), pl.col('c2').alias('b2')), on=['hand_id', 'partner'])
    ml = rows['mask_lo'].to_numpy().astype(np.uint64)
    mh = rows['mask_hi'].to_numpy().astype(np.uint64)
    full = ml | (mh << np.uint64(32))
    b1 = rows['b1'].to_numpy().astype(np.uint64)
    b2 = rows['b2'].to_numpy().astype(np.uint64)
    x = (((full >> b1) & np.uint64(1)) + ((full >> b2) & np.uint64(1))).astype(np.float32)
    rows = rows.with_columns(pl.Series('x', x))
    xm = rows.group_by('n_outs').agg(pl.col('x').mean().alias('x_mean'))
    rows = rows.join(xm, on='n_outs').with_columns((pl.col('x') - pl.col('x_mean')).alias('xc'), (pl.col('y_fold') - pl.col('y_fold_hat')).alias('r_fold'), (pl.col('y_agg') - pl.col('y_agg_hat')).alias('r_agg'))
    print(tag, 'partner rows', rows.height, 'x>0 share', float((rows['x'] > 0).mean()), flush=True)
    facing = rows.filter(pl.col('price') > 0)

    def zstat(df, r):
        return df.group_by('table_id', 'player_id', 'partner').agg(pl.len().alias('n'), (pl.col(r) * pl.col('xc')).sum().alias('num'),
                                                                   ((pl.col(r) ** 2) * (pl.col('xc') ** 2)).sum().alias('den'), (pl.col('x') > 0).sum().alias('n_blocked'))
    zf = zstat(facing, 'r_fold').with_columns((pl.col('num') / pl.col('den').sqrt().clip(lower_bound=1e-9)).alias('z_fold')).rename({'n': 'n_fold', 'n_blocked': 'blocked_fold'}).drop('num', 'den')
    za = zstat(rows, 'r_agg').with_columns((pl.col('num') / pl.col('den').sqrt().clip(lower_bound=1e-9)).alias('z_agg')).rename({'n': 'n_agg', 'n_blocked': 'blocked_agg'}).drop('num', 'den')
    pairs = zf.join(za, on=['table_id', 'player_id', 'partner'], how='full', coalesce=True)
    pairs.write_parquet(OUT / f'{tag}-directed.parquet')
    pairs = pairs.with_columns(pl.min_horizontal('player_id', 'partner').alias('player_1'), pl.max_horizontal('player_id', 'partner').alias('player_2'))
    ev = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').join(
        pl.read_csv(ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv').select('pair_id', 'risk_score', 'predicted_behavior'), on='pair_id')
    p = pairs.join(ev, on=['player_1', 'player_2'], how='inner')
    p.write_parquet(OUT / f'{tag}-pairs.parquet')
    summ = {'tag': tag, 'decisions': dec.height, 'partner_rows': rows.height, 'directed_pairs': pairs.height, 'directed_eval_pairs': p.height, 'seconds': round(time.perf_counter() - t0, 1),
            'flag_rule': 'z_fold >= 4.5 (n_fold >= 20) or z_agg <= -4.5 (n_agg >= 20); null per side = n * 3.4e-6'}
    groups = {'risk_lt_0.1': pl.col('risk_score') < .1, 'risk_0.1_0.5': (pl.col('risk_score') >= .1) & (pl.col('risk_score') < .5),
              'risk_0.5_0.9': (pl.col('risk_score') >= .5) & (pl.col('risk_score') < .9), 'risk_ge_0.9': pl.col('risk_score') >= .9}
    for z in ['z_fold', 'z_agg']:
        summ[z] = {}
        for gname, m in groups.items():
            v = p.filter(m & (pl.col(z.replace('z_', 'n_')) >= 20))[z].drop_nulls().to_numpy()
            summ[z][gname] = {'n': int(len(v)), 'mean': float(v.mean()) if len(v) else None, 'std': float(v.std()) if len(v) else None,
                              'q': [float(q) for q in np.quantile(v, [.001, .01, .5, .99, .999])] if len(v) else None,
                              'abs_ge_3': int((np.abs(v) >= 3).sum()), 'abs_ge_4': int((np.abs(v) >= 4).sum()), 'abs_ge_5': int((np.abs(v) >= 5).sum()),
                              'ge_3': int((v >= 3).sum()), 'le_-3': int((v <= -3).sum()), 'ge_4.5': int((v >= 4.5).sum()), 'le_-4.5': int((v <= -4.5).sum()),
                              'null_expect_abs_ge_4': float(len(v) * 2 * norm.sf(4)), 'null_expect_one_side_ge_4.5': float(len(v) * norm.sf(4.5))}
    flagged = p.filter(((pl.col('n_fold') >= 20) & (pl.col('z_fold') >= 4.5)) | ((pl.col('n_agg') >= 20) & (pl.col('z_agg') <= -4.5)))
    summ['flagged'] = {'n_directed': flagged.height, 'n_directed_risk_lt_0.5': int((flagged['risk_score'] < .5).sum()),
                       'null_expect_total': float(p.filter(pl.col('n_fold') >= 20).height * norm.sf(4.5) + p.filter(pl.col('n_agg') >= 20).height * norm.sf(4.5)),
                       'rows': flagged.sort('risk_score').select('pair_id', 'player_id', 'partner', 'risk_score', 'predicted_behavior', 'n_fold', 'blocked_fold', 'z_fold', 'n_agg', 'blocked_agg', 'z_agg', 'shared_hands').head(60).to_dicts()}
    tail = p.filter((pl.col('n_fold') >= 20) & (pl.col('z_fold').abs() >= 3.5)).sort('z_fold', descending=True).select('pair_id', 'player_id', 'partner', 'risk_score', 'n_fold', 'blocked_fold', 'z_fold', 'z_agg', 'shared_hands')
    summ['tail_fold_abs_ge_3.5'] = tail.head(40).to_dicts()
    tail2 = p.filter((pl.col('n_agg') >= 20) & (pl.col('z_agg').abs() >= 3.5)).sort('z_agg').select('pair_id', 'player_id', 'partner', 'risk_score', 'n_agg', 'blocked_agg', 'z_fold', 'z_agg', 'shared_hands')
    summ['tail_agg_abs_ge_3.5'] = tail2.head(40).to_dicts()
    write_json(OUT / f'{tag}-summary.json', summ)
    print(json.dumps({k: v for k, v in summ.items() if not k.startswith('tail') and k != 'flagged'}, indent=1, default=str))
    print('flagged', {k: v for k, v in summ['flagged'].items() if k != 'rows'})
    for d in summ['flagged']['rows'][:30]:
        print('  ', d)
    for k in [k for k in summ if k.startswith('tail')]:
        print(k)
        for d in summ[k][:12]:
            print('  ', d)


if __name__ == '__main__':
    main()
