"""STEP280 stage 5: the two-sample test.

Detector-style features (the src/features.py arithmetic, re-implemented here and never
edited there) are computed per hand on real honest hands and on simulated hands, and a
classifier is trained to tell the two apart with a split by TABLE.

The pre-registered threshold is applied ONCE:  AUC <= .55 PASS, .55-.65 PARTIAL, > .65 FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_lib as S
from prep_data import SCRATCH

HERE = Path(__file__).resolve().parent

PLAYER_COLS = ['contribution_bb', 'net_bb', 'stack_bb', 'folded', 'went_to_showdown',
               'won_share', 'n_actions', 'n_aggressive', 'n_check', 'n_call', 'late_checks',
               'agg_bb', 'call_bb', 'fold_pot_odds', 'min_active', 'hole_strength',
               'aggression_residual', 'check_residual', 'contribution_residual', 'net_residual']
OPP_COLS = ['opp_fold', 'opp_raise', 'opp_call', 'opp_call_bb', 'opp_fold_odds']


def seat_frame(hands: pl.DataFrame, seats: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    """src/features.py arithmetic, per (hand_id, player_id)."""
    a = actions.join(hands.select('hand_id', 'big_blind'), on='hand_id').sort('hand_id', 'action_no')
    a = a.with_columns(
        (pl.col('action').is_in(['bet', 'raise', 'all_in']) &
         (pl.col('amount') > pl.col('to_call'))).alias('aggressive'),
        (pl.col('amount') / pl.col('big_blind')).alias('amount_bb'),
        (pl.col('to_call') / (pl.col('pot_before') + pl.col('to_call')).clip(lower_bound=1))
        .alias('pot_odds'),
    ).with_columns(
        pl.when('aggressive').then(pl.col('player_id')).otherwise(None)
        .forward_fill().shift(1).over('hand_id', 'street').alias('previous_aggressor'))
    action_stats = a.group_by('hand_id', 'player_id').agg(
        pl.len().alias('n_actions'),
        pl.col('aggressive').sum().alias('n_aggressive'),
        (pl.col('action') == 'check').sum().alias('n_check'),
        (pl.col('action') == 'call').sum().alias('n_call'),
        ((pl.col('action') == 'check') & (pl.col('street') != 'preflop')).sum().alias('late_checks'),
        pl.when('aggressive').then('amount_bb').otherwise(0).sum().alias('agg_bb'),
        pl.when(pl.col('action').is_in(['call', 'all_in']) & ~pl.col('aggressive'))
        .then('amount_bb').otherwise(0).sum().alias('call_bb'),
        pl.when(pl.col('action') == 'fold').then('pot_odds').otherwise(0).max()
        .alias('fold_pot_odds'),
        pl.col('players_active').min().alias('min_active'))
    interactions = a.filter(pl.col('previous_aggressor').is_not_null()).group_by(
        'hand_id', 'player_id', 'previous_aggressor').agg(
        (pl.col('action') == 'fold').sum().alias('opp_fold'),
        pl.col('aggressive').sum().alias('opp_raise'),
        (~pl.col('aggressive') & pl.col('action').is_in(['call', 'all_in'])).sum().alias('opp_call'),
        pl.when(~pl.col('aggressive')).then('amount_bb').otherwise(0).sum().alias('opp_call_bb'),
        pl.when(pl.col('action') == 'fold').then('pot_odds').otherwise(0).max()
        .alias('opp_fold_odds'))
    s = seats.join(hands.select('hand_id', 'big_blind', 'final_pot', 'players_at_showdown'),
                   on='hand_id')
    s = s.join(action_stats, on=['hand_id', 'player_id'], how='left').with_columns(
        (pl.col('total_contribution') / pl.col('big_blind')).alias('contribution_bb'),
        (pl.col('net_chips') / pl.col('big_blind')).alias('net_bb'),
        (pl.col('starting_stack') / pl.col('big_blind')).alias('stack_bb'),
        pl.col('hole_card_1').str.slice(0, 1).replace_strict(
            {r: i + 2 for i, r in enumerate('23456789TJQKA')}).cast(pl.Float64).alias('rank1'),
        pl.col('hole_card_2').str.slice(0, 1).replace_strict(
            {r: i + 2 for i, r in enumerate('23456789TJQKA')}).cast(pl.Float64).alias('rank2'),
        (pl.col('hole_card_1').str.slice(1) == pl.col('hole_card_2').str.slice(1))
        .cast(pl.Float64).alias('suited'),
    ).with_columns(
        ((pl.max_horizontal('rank1', 'rank2') + pl.min_horizontal('rank1', 'rank2') * .5) / 21 +
         (pl.col('rank1') == pl.col('rank2')).cast(pl.Float64) * .7 +
         pl.col('suited') * .1).alias('hole_strength'))
    base = s.group_by('player_id').agg(
        pl.col('n_aggressive').mean().alias('baseline_aggressive'),
        pl.col('n_check').mean().alias('baseline_check'),
        pl.col('contribution_bb').mean().alias('baseline_contribution'),
        pl.col('net_bb').mean().alias('baseline_net'),
        pl.col('folded').mean().alias('baseline_fold'))
    s = s.join(base, on='player_id').with_columns(
        (pl.col('n_aggressive') - pl.col('baseline_aggressive')).alias('aggression_residual'),
        (pl.col('n_check') - pl.col('baseline_check')).alias('check_residual'),
        (pl.col('contribution_bb') - pl.col('baseline_contribution')).alias('contribution_residual'),
        (pl.col('net_bb') - pl.col('baseline_net')).alias('net_residual'))
    return s, interactions, a


def hand_features(hands, seats, actions, tag):
    s, interactions, a = seat_frame(hands, seats, actions)
    s = s.with_columns([pl.col(c).cast(pl.Float64).fill_null(0) for c in PLAYER_COLS])

    # ---- block A: hand level
    per_street = a.group_by('hand_id', 'street').len().pivot(
        on='street', index='hand_id', values='len').fill_null(0)
    for c in ['preflop', 'flop', 'turn', 'river']:
        if c not in per_street.columns:
            per_street = per_street.with_columns(pl.lit(0).alias(c))
    per_street = per_street.rename({c: f'n_{c}_actions' for c in
                                    ['preflop', 'flop', 'turn', 'river']})
    hstat = a.group_by('hand_id').agg(
        pl.len().alias('n_actions_total'),
        pl.col('aggressive').sum().alias('n_aggressive_total'),
        (pl.col('action') == 'fold').sum().alias('n_folds'),
        (pl.col('action') == 'all_in').sum().alias('n_allin'),
        ((pl.col('street') == 'preflop') & pl.col('aggressive')).sum().alias('n_preflop_raises'))
    A = (hands.select('hand_id', 'table_id', 'big_blind', 'final_pot', 'players_at_showdown',
                      'board_cards')
         .with_columns((pl.col('final_pot') / pl.col('big_blind')).alias('pot_bb'),
                       pl.col('board_cards').str.split(' ').list.len().alias('nb_raw'))
         .with_columns(pl.when(pl.col('board_cards') == '').then(0)
                       .otherwise(pl.col('nb_raw')).alias('n_board'))
         .drop('board_cards', 'nb_raw')
         .join(hstat, on='hand_id', how='left').join(per_street, on='hand_id', how='left')
         .with_columns(pl.exclude('hand_id', 'table_id').fill_null(0)))
    vpip = s.filter(pl.col('contribution_bb') > 1.0).group_by('hand_id').len().rename(
        {'len': 'n_voluntary'})
    A = A.join(vpip, on='hand_id', how='left').with_columns(pl.col('n_voluntary').fill_null(0))

    # ---- block B: per-seat aggregation
    aggs = []
    for c in PLAYER_COLS:
        aggs += [pl.col(c).mean().alias(f'B_{c}_mean'), pl.col(c).std().alias(f'B_{c}_std'),
                 pl.col(c).min().alias(f'B_{c}_min'), pl.col(c).max().alias(f'B_{c}_max')]
    B = s.group_by('hand_id').agg(aggs)

    # ---- block C: all 15 seat pairs
    left = s.select(['hand_id', 'player_id'] + PLAYER_COLS).rename(
        {'player_id': 'player_1', **{c: c + '_1' for c in PLAYER_COLS}})
    right = s.select(['hand_id', 'player_id'] + PLAYER_COLS).rename(
        {'player_id': 'player_2', **{c: c + '_2' for c in PLAYER_COLS}})
    P = left.join(right, on='hand_id').filter(pl.col('player_1') < pl.col('player_2'))
    for side, other in [(1, 2), (2, 1)]:
        inter = interactions.rename(
            {'player_id': f'player_{side}', 'previous_aggressor': f'player_{other}',
             **{c: c + f'_{side}' for c in OPP_COLS}})
        P = P.join(inter, on=['hand_id', 'player_1', 'player_2'], how='left')
    P = P.with_columns([pl.col(f'{c}_{i}').cast(pl.Float64).fill_null(0)
                        for c in OPP_COLS for i in (1, 2)])
    pair_exprs = []
    for c in PLAYER_COLS:
        pair_exprs.append((pl.col(c + '_1') - pl.col(c + '_2')).abs().alias('C_' + c + '_gap'))
    for c in OPP_COLS:
        c1, c2 = pl.col(c + '_1'), pl.col(c + '_2')
        pair_exprs += [(c1 + c2).alias('C_' + c + '_sum'),
                       (c1 - c2).abs().alias('C_' + c + '_gap'),
                       pl.min_horizontal(c1, c2).alias('C_' + c + '_min'),
                       pl.max_horizontal(c1, c2).alias('C_' + c + '_max')]
    pair_exprs += [
        (((pl.col('net_bb_1') > 0) & (pl.col('net_bb_2') < 0)) |
         ((pl.col('net_bb_2') > 0) & (pl.col('net_bb_1') < 0)))
        .cast(pl.Float64).alias('C_opposite_outcomes'),
        ((pl.col('folded_1') > 0) & (pl.col('folded_2') > 0)).cast(pl.Float64).alias('C_both_fold'),
        (pl.col('won_share_1') + pl.col('won_share_2')).alias('C_joint_win_share'),
        (pl.col('hole_strength_1') * pl.col('opp_fold_1') +
         pl.col('hole_strength_2') * pl.col('opp_fold_2')).alias('C_strong_fold_to_partner'),
        (pl.col('hole_strength_1') * pl.col('late_checks_1') +
         pl.col('hole_strength_2') * pl.col('late_checks_2')).alias('C_strong_late_checks'),
    ]
    P = P.with_columns(pair_exprs)
    cnames = [e.meta.output_name() for e in pair_exprs]
    C = P.group_by('hand_id').agg(
        [pl.col(c).mean().alias(c + '_pmean') for c in cnames] +
        [pl.col(c).max().alias(c + '_pmax') for c in cnames])

    out = A.join(B, on='hand_id').join(C, on='hand_id')
    return out.with_columns(pl.lit(tag).alias('source'))


def load_real():
    sel = pl.read_parquet(SCRATCH / 'sel_hands.parquet').filter(pl.col('grp') == 'TWO_SAMPLE')
    keep = set(sel['hand_id'].to_list())
    seats = pl.read_parquet(SCRATCH / 'sel_seats.parquet').filter(pl.col('hand_id').is_in(keep))
    acts = pl.read_parquet(SCRATCH / 'sel_actions.parquet').filter(pl.col('hand_id').is_in(keep))
    return sel.drop('grp', strict=False), seats, acts


def load_sim(suffix=''):
    return (pl.read_parquet(SCRATCH / f'sim_hands{suffix}.parquet'),
            pl.read_parquet(SCRATCH / f'sim_seats{suffix}.parquet'),
            pl.read_parquet(SCRATCH / f'sim_actions{suffix}.parquet'))


def main(read_id=1, suffix='', note=''):
    t0 = time.time()
    rh, rs, ra = load_real()
    sh, ss, sa = load_sim(suffix)
    common = sh['hand_id'].str.slice(1).to_list()
    rh = rh.filter(pl.col('hand_id').str.slice(1).is_in(common))
    rk = rh['hand_id'].to_list()
    rs = rs.filter(pl.col('hand_id').is_in(rk))
    ra = ra.filter(pl.col('hand_id').is_in(rk))
    print(f'real {rh.height} sim {sh.height} ({time.time() - t0:.0f}s)', flush=True)
    FR = hand_features(rh, rs, ra, 'real')
    FS = hand_features(sh, ss, sa, 'sim')
    cols = [c for c in FR.columns if c not in ('hand_id', 'table_id', 'source')]
    FS = FS.select(FR.columns)
    D = pl.concat([FR, FS])
    D = D.with_columns(
        (pl.col('source') == 'sim').cast(pl.Int8).alias('y'),
        pl.col('table_id').map_elements(S.disc_side, return_dtype=pl.Utf8).alias('side'))
    print(f'{D.height} rows x {len(cols)} features ({time.time() - t0:.0f}s)', flush=True)

    tr = D.filter(pl.col('side') == 'train')
    te = D.filter(pl.col('side') == 'test')
    Xtr = tr.select(cols).to_numpy().astype(np.float32)
    Xte = te.select(cols).to_numpy().astype(np.float32)
    ytr = tr['y'].to_numpy().astype(np.int64)
    yte = te['y'].to_numpy().astype(np.int64)
    print(f'train {len(ytr)} ({ytr.mean():.3f} sim) test {len(yte)} ({yte.mean():.3f} sim), '
          f'tables train {tr["table_id"].n_unique()} test {te["table_id"].n_unique()}', flush=True)

    settings = [dict(num_boost_round=400, num_leaves=31, learning_rate=.05, feature_fraction=.8),
                dict(num_boost_round=1200, num_leaves=63, learning_rate=.02, feature_fraction=.5)]
    gains = np.zeros(len(cols))
    pte = np.zeros(len(yte))
    nm = 0
    for cfg in settings:
        for seed in range(5):
            p = dict(objective='binary', metric='auc', num_threads=4, verbosity=-1,
                     bagging_fraction=.8, bagging_freq=1, min_child_samples=40,
                     num_leaves=cfg['num_leaves'], learning_rate=cfg['learning_rate'],
                     feature_fraction=cfg['feature_fraction'],
                     seed=90001 + seed, bagging_seed=90001 + seed,
                     feature_fraction_seed=90001 + seed)
            m = lgb.train(p, lgb.Dataset(Xtr, label=ytr, feature_name=cols),
                          num_boost_round=cfg['num_boost_round'])
            pte += m.predict(Xte)
            gains += np.asarray(m.feature_importance('gain'))
            nm += 1
            print(f'  model {nm}/10 ({time.time() - t0:.0f}s)', flush=True)
    pte /= nm
    auc = float(roc_auc_score(yte, pte))

    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(np.nan_to_num((Xtr - mu) / sd), ytr)
    auc_lr = float(roc_auc_score(yte, lr.predict_proba(np.nan_to_num((Xte - mu) / sd))[:, 1]))

    order = np.argsort(-gains)
    top = []
    for i in order[:20]:
        u = float(roc_auc_score(yte, np.nan_to_num(Xte[:, i])))
        top.append({'feature': cols[i], 'gain_share': round(float(gains[i] / gains.sum()), 4),
                    'univariate_auc': round(max(u, 1 - u), 4),
                    'real_mean': round(float(np.nanmean(Xte[yte == 0, i])), 4),
                    'sim_mean': round(float(np.nanmean(Xte[yte == 1, i])), 4)})

    verdict = 'PASS' if auc <= .55 else ('PARTIAL' if auc <= .65 else 'FAIL')
    res = {'read_id': read_id, 'note': note,
           'n_hands_real': int((D['y'] == 0).sum()), 'n_hands_sim': int((D['y'] == 1).sum()),
           'n_features': len(cols),
           'train_rows': int(len(ytr)), 'test_rows': int(len(yte)),
           'train_tables': int(tr['table_id'].n_unique()),
           'test_tables': int(te['table_id'].n_unique()),
           'held_out_auc_primary_10model_lgbm': round(auc, 4),
           'held_out_auc_logistic_control': round(auc_lr, 4),
           'verdict': verdict,
           'threshold': {'PASS': '<= .55', 'PARTIAL': '.55-.65', 'FAIL': '> .65'},
           'top20_features': top,
           'seconds': round(time.time() - t0, 1)}
    (HERE / f'results280_read{read_id}.json').write_text(json.dumps(res, indent=1),
                                                         encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--read', type=int, default=1)
    p.add_argument('--suffix', default='')
    p.add_argument('--note', default='')
    a = p.parse_args()
    main(a.read, a.suffix, a.note)
