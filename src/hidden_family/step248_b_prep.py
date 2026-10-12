"""STEP248 Part B prep: normalise ONE phase into compact per-chunk numeric files.

Phase is filtered and asserted here and nowhere else touches the other phase, so no artefact
downstream can mix phases.  Development can be truncated to the first N hands of each table
(started_at order) to length-match evaluation.

Writes to the session scratchpad, not into the project tree.
"""
import os

os.environ.setdefault('POLARS_MAX_THREADS', '6')
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ORIGINAL = Path('C:/Data/Projects/Money Making/work/poker')
SCRATCH = Path(os.environ.get('STEP248_SCRATCH',
                              'C:/Users/USER/AppData/Local/Temp/claude/'
                              'C--Data-Projects-Money-Making-work-poker/'
                              'f2bb4f17-323f-4806-98b3-984023c70212/scratchpad/step248'))
TABLES_PER_CHUNK = 10
STREETS = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
ACTS = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all_in': 5}


def main(phase='evaluation', limit=None):
    t0 = time.time()
    tag = phase if limit is None else f'{phase}{limit}'
    dest = SCRATCH / tag
    dest.mkdir(parents=True, exist_ok=True)

    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet')
    hands = hands.filter(pl.col('phase') == phase)
    assert hands.height > 0 and (hands['phase'] == phase).all(), 'phase assertion failed'
    hands = hands.with_columns(pl.col('started_at').dt.epoch('us').alias('ts')).drop(
        'started_at', 'phase')
    hands = hands.sort(['table_id', 'ts'])
    if limit is not None:
        hands = hands.group_by('table_id', maintain_order=True).head(limit)
    tabs = sorted(hands['table_id'].unique().to_list())
    tmap = {t: i for i, t in enumerate(tabs)}
    hands = hands.with_columns(
        pl.col('table_id').replace_strict(tmap, return_dtype=pl.Int32).alias('tidx'))
    hands = hands.sort(['tidx', 'ts']).with_row_index('hidx').with_columns(
        pl.col('hidx').cast(pl.Int32),
        pl.col('hidx').cum_count().over('tidx').cast(pl.Int32).alias('hloc') - 1,
        (pl.col('tidx') // TABLES_PER_CHUNK).alias('chunk'))
    hid_map = dict(zip(hands['hand_id'].to_list(), hands['hidx'].to_list()))
    print(f'{phase} limit={limit}: {hands.height} hands, {len(tabs)} tables '
          f'[{time.time() - t0:.0f}s]', flush=True)

    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet')
    seats = seats.filter(pl.col('hand_id').is_in(pl.Series(list(hid_map.keys()))))
    seats = seats.with_columns(
        pl.col('hand_id').replace_strict(hid_map, return_dtype=pl.Int32).alias('hidx'))
    seats = seats.join(hands.select('hidx', 'tidx', 'chunk'), on='hidx', how='inner')
    # local player index inside the table (0..29), stable and table-scoped
    ptab = seats.select('tidx', 'player_id').unique().sort(['tidx', 'player_id'])
    ptab = ptab.with_columns(
        (pl.col('player_id').cum_count().over('tidx').cast(pl.Int32) - 1).alias('ploc'))
    n_per = ptab.group_by('tidx').len()
    assert n_per['len'].min() == 30 and n_per['len'].max() == 30, n_per['len'].min()
    seats = seats.join(ptab, on=['tidx', 'player_id'], how='inner')
    print(f'seats {seats.height}  [{time.time() - t0:.0f}s]', flush=True)

    lf = pl.scan_parquet(ORIGINAL / 'data' / 'actions.parquet').join(
        hands.lazy().select('hand_id', 'hidx', 'tidx', 'chunk'), on='hand_id', how='inner')
    acts = lf.select('hidx', 'tidx', 'chunk', 'action_no', 'street', 'player_id', 'action',
                     'amount', 'amount_to', 'pot_before', 'to_call').collect(engine='streaming')
    acts = acts.with_columns(
        pl.col('street').replace_strict(STREETS, return_dtype=pl.Int8),
        pl.col('action').replace_strict(ACTS, return_dtype=pl.Int8))
    acts = acts.join(ptab, on=['tidx', 'player_id'], how='inner').drop('player_id')
    acts = acts.sort(['hidx', 'action_no'])
    print(f'actions {acts.height}  [{time.time() - t0:.0f}s]', flush=True)

    ptab.write_parquet(dest / 'players.parquet')
    pl.DataFrame({'table_id': tabs, 'tidx': list(range(len(tabs)))}).write_parquet(
        dest / 'tables.parquet')
    hands.select('hidx', 'hand_id', 'tidx', 'hloc', 'chunk', 'button_seat', 'big_blind',
                 'board_cards', 'final_pot', 'ts').write_parquet(dest / 'hands.parquet')
    for ch, part in hands.partition_by('chunk', as_dict=True).items():
        c = ch[0] if isinstance(ch, tuple) else ch
        part.select('hidx', 'tidx', 'hloc', 'button_seat', 'big_blind', 'board_cards',
                    'final_pot').write_parquet(dest / f'h_{c:03d}.parquet')
    for ch, part in seats.partition_by('chunk', as_dict=True).items():
        c = ch[0] if isinstance(ch, tuple) else ch
        part.select('hidx', 'tidx', 'ploc', 'seat_no', 'hole_card_1', 'hole_card_2',
                    'total_contribution', 'net_chips', 'folded', 'went_to_showdown',
                    'won_share').sort(['hidx', 'seat_no']).write_parquet(
            dest / f's_{c:03d}.parquet')
    for ch, part in acts.partition_by('chunk', as_dict=True).items():
        c = ch[0] if isinstance(ch, tuple) else ch
        part.select('hidx', 'tidx', 'ploc', 'action_no', 'street', 'action', 'amount',
                    'amount_to', 'pot_before', 'to_call').sort(
            ['hidx', 'action_no']).write_parquet(dest / f'a_{c:03d}.parquet')
    print(f'wrote {dest}  [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    ph = sys.argv[1] if len(sys.argv) > 1 else 'evaluation'
    lim = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'none' else None
    main(ph, lim)
