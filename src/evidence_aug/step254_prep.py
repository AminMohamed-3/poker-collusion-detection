"""STEP254 prep: (a) the 169-class preflop tables, (b) the chronology / chip-flow block (family 4).

Development phase only (asserted). Writes into the STEP254 experiment directory:
  step254_pf169.npz   eq169[169] (equity vs one random hand), pct169[169] (combo-weighted
                      percentile of that equity), eq169x169[169,169] (Monte-Carlo heads-up equity)
  step254_chrono.parquet  (pair_id, hand_id) -> family (4) running quantities + the A/B role flag

Chronology rule: every running quantity uses ONLY the current hand and EARLIER hands of the same
table / pair, ordered by started_at. The A/B role (A = the member who loses more chips over the
pair's shared hands) is the STEP237 pair-level orientation.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import time

import numpy as np
import polars as pl

import step254_eval7 as E

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
OUT = CV + r'\experiments\step254_mechanical_tensor_20260917'
OOF = CV + r'\experiments\step232_event_detector_20260917\hand_oof.parquet'
N_DEALS = 8_000_000
CHUNK = 2_000_000


def class169(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    r1, r2 = c1 >> 2, c2 >> 2
    hi, lo = np.maximum(r1, r2), np.minimum(r1, r2)
    suited = (c1 & 3) == (c2 & 3)
    return np.where(suited | (hi == lo), hi * 13 + lo, lo * 13 + hi)


def build_pf169(seed: int = 254) -> None:
    rng = np.random.default_rng(seed)
    S = np.zeros(169 * 169)
    C = np.zeros(169 * 169)
    kept = 0
    t0 = time.time()
    while kept < N_DEALS:
        cand = rng.integers(0, 52, size=(CHUNK, 9), dtype=np.int64)
        ok = (np.diff(np.sort(cand, axis=1), axis=1) != 0).all(axis=1)
        d = cand[ok]
        kept += len(d)
        board = d[:, 4:9]
        v1 = E.evaluate(np.concatenate([d[:, 0:2], board], axis=1))
        v2 = E.evaluate(np.concatenate([d[:, 2:4], board], axis=1))
        sc = np.where(v1 < v2, 1.0, np.where(v1 == v2, .5, 0.))
        i1 = class169(d[:, 0], d[:, 1])
        i2 = class169(d[:, 2], d[:, 3])
        f1, f2 = i1 * 169 + i2, i2 * 169 + i1
        S += np.bincount(f1, weights=sc, minlength=169 * 169)
        S += np.bincount(f2, weights=1.0 - sc, minlength=169 * 169)
        C += np.bincount(f1, minlength=169 * 169)
        C += np.bincount(f2, minlength=169 * 169)
        print(f'  deals {kept:,} / {N_DEALS:,}  {time.time() - t0:.0f}s', flush=True)
    eq = (S / np.maximum(C, 1)).reshape(169, 169)
    cnt = C.reshape(169, 169)
    eq_rand = S.reshape(169, 169).sum(axis=1) / np.maximum(cnt.sum(axis=1), 1)
    # combo weights: 6 for a pair, 4 suited, 12 offsuit
    w = np.zeros(169)
    for hi in range(13):
        for lo in range(13):
            i = hi * 13 + lo
            w[i] = 6 if hi == lo else (4 if hi > lo else 12)
    pct = np.zeros(169)
    for i in range(169):
        pct[i] = (w[eq_rand < eq_rand[i]].sum() + .5 * w[i]) / w.sum()
    print('min cell count', cnt.min(), 'median', np.median(cnt))
    np.savez(OUT + r'\step254_pf169.npz', eq169=eq_rand, pct169=pct, eq169x169=eq, cnt=cnt)
    order = np.argsort(-eq_rand)[:8]
    names = []
    for i in order:
        hi, lo = i // 13, i % 13
        names.append(f'{E.RANK_CHARS[max(hi, lo)]}{E.RANK_CHARS[min(hi, lo)]}'
                     f'{"" if hi == lo else ("s" if hi > lo else "o")} {eq_rand[i]:.3f}')
    print('top preflop classes:', names)


def build_chrono() -> None:
    keys = pl.read_parquet(OOF, columns=['pair_id', 'hand_id', 'table_id'])
    lab = pl.read_csv(DATA + r'\development_labels.csv').select('pair_id', 'player_1', 'player_2')
    keys = keys.join(lab, on='pair_id')
    hands = pl.read_parquet(DATA + r'\hands.parquet',
                            columns=['hand_id', 'table_id', 'started_at', 'phase', 'big_blind'])
    tb = keys.select('table_id').unique()
    hd = hands.join(tb, on='table_id')
    hd = hd.filter(pl.col('phase') == 'development')
    assert (hd['phase'] == 'development').all()
    hd = hd.drop('phase').sort('table_id', 'started_at', 'hand_id')
    hd = hd.with_columns(pl.int_range(pl.len()).over('table_id').alias('hidx'))
    # the hands of the labelled pairs must all be inside this development table history
    assert keys.select('hand_id').unique().join(hd.select('hand_id'), on='hand_id').height == \
        keys['hand_id'].n_unique(), 'a pair hand is not a development hand of its table'
    members = pl.concat([lab.select(pl.col('player_1').alias('player_id')),
                         lab.select(pl.col('player_2').alias('player_id'))]).unique()
    s = (pl.scan_parquet(DATA + r'\seats.parquet')
         .select('hand_id', 'player_id', 'starting_stack', 'net_chips', 'total_contribution')
         .join(members.lazy(), on='player_id')
         .join(hd.lazy().select('hand_id', 'table_id', 'hidx', 'big_blind'), on='hand_id')
         .collect())
    print('member seat rows', s.height, flush=True)
    s = s.sort('player_id', 'table_id', 'hidx')
    g = ['player_id', 'table_id']
    s = s.with_columns([
        pl.col('starting_stack').shift(1).over(g).alias('_pstack'),
        pl.col('net_chips').shift(1).over(g).alias('_pnet'),
        pl.col('hidx').shift(1).over(g).alias('_phidx'),
    ])
    s = s.with_columns(
        (pl.col('starting_stack') - (pl.col('_pstack') + pl.col('_pnet'))).alias('topup'))
    s = s.with_columns([
        (pl.col('topup').is_not_null() & (pl.col('topup') != 0)).cast(pl.Int8).alias('rebuy'),
        (pl.col('_phidx').is_null() | (pl.col('hidx') - pl.col('_phidx') != 1)
         | (pl.col('topup') < 0)).cast(pl.Int8).alias('new_sit'),
    ])
    s = s.with_columns(pl.col('new_sit').cum_sum().over(g).alias('sess'))
    g2 = ['player_id', 'table_id', 'sess']
    s = s.with_columns([
        pl.int_range(pl.len()).over(g2).alias('session_age'),
        (pl.col('net_chips').cum_sum().over(g2) - pl.col('net_chips')).alias('sess_profit_before'),
        pl.col('rebuy').shift(1).fill_null(0).rolling_sum(5, min_periods=1).over(g).alias('rebuy_prev5'),
        (pl.col('starting_stack') - pl.col('starting_stack').shift(5)).over(g).alias('stackchg5'),
    ])
    bb = pl.col('big_blind')
    s = s.select('hand_id', 'player_id',
                 pl.col('session_age').cast(pl.Float32),
                 (pl.col('sess_profit_before') / bb).cast(pl.Float32).alias('sess_profit_bb'),
                 pl.col('rebuy').cast(pl.Float32),
                 (pl.col('topup').fill_null(0) / bb).cast(pl.Float32).alias('topup_bb'),
                 pl.col('rebuy_prev5').cast(pl.Float32),
                 (pl.col('stackchg5').fill_null(0) / bb).cast(pl.Float32).alias('stackchg5_bb'),
                 (pl.col('starting_stack') / bb).cast(pl.Float32).alias('stack_bb'),
                 pl.col('net_chips'), pl.col('total_contribution'))
    COLS = ['session_age', 'sess_profit_bb', 'rebuy', 'topup_bb', 'rebuy_prev5', 'stackchg5_bb',
            'stack_bb']
    k = keys.join(hd.select('hand_id', 'started_at', 'big_blind'), on='hand_id')
    for who, pcol in (('1', 'player_1'), ('2', 'player_2')):
        k = k.join(s.rename({c: c + '_' + who for c in COLS + ['net_chips', 'total_contribution']}),
                   left_on=['hand_id', pcol], right_on=['hand_id', 'player_id'], how='left')
    for c in COLS:
        assert k[c + '_1'].null_count() == 0 and k[c + '_2'].null_count() == 0, c
    n1, n2 = pl.col('net_chips_1'), pl.col('net_chips_2')
    k = k.with_columns(
        pl.when((n1 < 0) & (n2 > 0)).then(pl.min_horizontal(-n1, n2))
        .when((n2 < 0) & (n1 > 0)).then(-pl.min_horizontal(-n2, n1))
        .otherwise(0).cast(pl.Float64).alias('flow_1to2'))
    k = k.sort('pair_id', 'started_at', 'hand_id')
    k = k.with_columns([
        ((pl.col('flow_1to2').cum_sum().over('pair_id') - pl.col('flow_1to2'))
         / pl.col('big_blind')).alias('cumflow_before_1to2'),
        ((pl.col('flow_1to2').abs().cum_sum().over('pair_id') - pl.col('flow_1to2').abs())
         / pl.col('big_blind')).alias('cumabsflow_before'),
        pl.int_range(pl.len()).over('pair_id').alias('shared_idx'),
        (pl.col('flow_1to2') / pl.col('big_blind')).alias('flow_bb_1to2'),
    ])
    # role: A = the member who loses more chips over the pair's shared hands
    role = k.group_by('pair_id').agg(pl.col('net_chips_1').sum().alias('tot1'),
                                     pl.col('net_chips_2').sum().alias('tot2'))
    role = role.with_columns((pl.col('tot1') <= pl.col('tot2')).cast(pl.Int8).alias('p1_is_A'))
    k = k.join(role.select('pair_id', 'p1_is_A'), on='pair_id')
    A = pl.col('p1_is_A') == 1

    def ab(name: str):
        a = pl.when(A).then(pl.col(name + '_1')).otherwise(pl.col(name + '_2')).cast(pl.Float32)
        b = pl.when(A).then(pl.col(name + '_2')).otherwise(pl.col(name + '_1')).cast(pl.Float32)
        return a, b

    exprs = [pl.col('p1_is_A')]
    for c in COLS:
        a, b = ab(c)
        exprs += [a.alias(f't4_{c}_A'), b.alias(f't4_{c}_B'),
                  pl.max_horizontal(a, b).alias(f't4_{c}_mx'),
                  pl.min_horizontal(a, b).alias(f't4_{c}_mn'),
                  (a + b).alias(f't4_{c}_sm')]
    sgn = pl.when(A).then(1.0).otherwise(-1.0)
    exprs += [
        (pl.col('cumflow_before_1to2') * sgn).cast(pl.Float32).alias('t4_cumflow_before_A2B'),
        pl.col('cumabsflow_before').cast(pl.Float32).alias('t4_cumabsflow_before'),
        pl.col('shared_idx').cast(pl.Float32).alias('t4_shared_idx'),
        (pl.col('flow_bb_1to2') * sgn).cast(pl.Float32).alias('t4_flow_bb_A2B'),
        (pl.col('cumflow_before_1to2') * sgn / (pl.col('shared_idx') + 1)).cast(pl.Float32)
        .alias('t4_cumflow_rate_A2B'),
    ]
    out = k.select('pair_id', 'hand_id', *exprs)
    assert out.height == keys.height
    out.write_parquet(OUT + r'\step254_chrono.parquet')
    print('chrono', out.shape, flush=True)
    print(out.select([c for c in out.columns if c.startswith('t4_')][:8]).describe())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if not os.path.exists(OUT + r'\step254_pf169.npz'):
        build_pf169()
    print('pf169 done', round(time.time() - t0, 1), 's', flush=True)
    build_chrono()
    print('prep done', round(time.time() - t0, 1), 's')
