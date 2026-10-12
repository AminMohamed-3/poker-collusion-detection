"""STEP257 prep: the STEP254 chronology / chip-flow block (family 4) for ONE phase, parameterised.

This is src/evidence_aug/step254_prep.py:build_chrono with exactly two things changed - the phase and
the pair list - as REPORT.md section 8 prescribes.  Nothing else differs: the same sort keys, the same
shift / cumulative expressions, the same A/B role rule, the same column names and dtypes.
step254_pf169.npz is phase-independent and is reused unchanged (never rebuilt here).

Every running quantity uses ONLY the current hand and EARLIER hands of the same table / pair inside
the SAME phase, ordered by started_at.  The phase is asserted on every hand touched.

Stage A (once per phase)  : hands table with the within-table chronological index, and the per
                            (hand, player) session block for the member players.
Stage B (per pair batch)  : the (pair_id, hand_id) chronology rows, written one file per batch.

Usage:  python step257_prep_eval.py evaluation
        python step257_prep_eval.py development         (the parity-gate copy)
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import glob
import json
import sys
import time

import polars as pl

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
SCRATCH = (r'C:\Users\USER\AppData\Local\Temp\claude\C--Data-Projects-Money-Making-work-poker'
           r'\f2bb4f17-323f-4806-98b3-984023c70212\scratchpad\step257')
COLS = ['session_age', 'sess_profit_bb', 'rebuy', 'topup_bb', 'rebuy_prev5', 'stackchg5_bb',
        'stack_bb']
PAIRFILE = {'evaluation': DATA + r'\evaluation_pairs.csv',
            'development': DATA + r'\development_labels.csv'}
BATCHDIR = {'evaluation': ROOT + r'\outputs_v2\hands_evaluation',
            'development': ROOT + r'\outputs_v2\hands_development'}


def pair_list(phase: str) -> pl.DataFrame:
    return pl.read_csv(PAIRFILE[phase]).select('pair_id', 'player_1', 'player_2')


def batch_files(phase: str) -> list[str]:
    return sorted(glob.glob(BATCHDIR[phase] + r'\batch_*.parquet'))


def stage_a(phase: str, keys_tables: pl.DataFrame, lab: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """hd (hand chronology inside the phase) and s (per hand/player session block)."""
    hands = pl.read_parquet(DATA + r'\hands.parquet',
                            columns=['hand_id', 'table_id', 'started_at', 'phase', 'big_blind'])
    hd = hands.join(keys_tables, on='table_id')
    hd = hd.filter(pl.col('phase') == phase)
    assert (hd['phase'] == phase).all()
    hd = hd.drop('phase').sort('table_id', 'started_at', 'hand_id')
    hd = hd.with_columns(pl.int_range(pl.len()).over('table_id').alias('hidx'))
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
    return hd, s


def chrono_for(keys: pl.DataFrame, hd: pl.DataFrame, s: pl.DataFrame) -> pl.DataFrame:
    """keys: pair_id, hand_id, player_1, player_2 for a set of COMPLETE pairs."""
    k = keys.join(hd.select('hand_id', 'started_at', 'big_blind'), on='hand_id')
    assert k.height == keys.height, 'a pair hand is not a hand of its table inside this phase'
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
    return out


def main() -> None:
    phase = sys.argv[1]
    assert phase in ('evaluation', 'development')
    outdir = SCRATCH + '\\chrono_' + phase
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    lab = pair_list(phase)
    files = batch_files(phase)
    allkeys = [pl.read_parquet(f, columns=['pair_id', 'hand_id']) for f in files]
    tabs = (pl.read_parquet(DATA + r'\hands.parquet', columns=['hand_id', 'table_id', 'phase'])
            .filter(pl.col('phase') == phase).drop('phase'))
    ktab = pl.concat(allkeys).join(tabs, on='hand_id').select('table_id').unique()
    hd, s = stage_a(phase, ktab, lab)
    print('stage A done', hd.height, 'hands', s.height, 'seat rows',
          round(time.time() - t0, 1), 's', flush=True)
    tot = 0
    for b, kk in enumerate(allkeys):
        kk = kk.join(lab, on='pair_id')
        c = chrono_for(kk, hd, s)
        c.write_parquet(outdir + f'\\chrono_b{b:02d}.parquet')
        tot += c.height
        print(f'  batch {b:02d}  {c.height} rows  {time.time() - t0:.0f}s', flush=True)
    json.dump({'phase': phase, 'rows': tot, 'batches': len(allkeys), 'hands_in_phase': hd.height,
               'member_seat_rows': s.height, 'seconds': round(time.time() - t0, 1)},
              open(outdir + r'\receipt_chrono.json', 'w', encoding='utf-8'), indent=1)
    print('chrono total', tot, round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
