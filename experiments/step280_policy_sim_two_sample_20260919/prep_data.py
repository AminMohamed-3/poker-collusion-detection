"""STEP280 stage 1: select honest development hands and cache the rows we need.

Development phase only (asserted). players.parquet is never opened.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_lib as S

SCRATCH = Path(r'C:\Users\USER\AppData\Local\Temp\claude'
               r'\C--Data-Projects-Money-Making-work-poker'
               r'\f2bb4f17-323f-4806-98b3-984023c70212\scratchpad\step280')
SCRATCH.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent

N_TRAIN, N_HELD, N_SAMPLE = 500, 375, 800
SEED = 280019


def main():
    t0 = time.time()
    hands = S.load_dev_hands()
    print(f'dev hands {hands.height}, tables {hands["table_id"].n_unique()}', flush=True)
    posp = S.positive_players()
    print(f'positive players {len(posp)}', flush=True)

    flag_path = SCRATCH / 'honest_flags.parquet'
    if flag_path.exists():
        flags = pl.read_parquet(flag_path)
    else:
        seats = (pl.scan_parquet(S.DATA / 'seats.parquet').select('hand_id', 'player_id')
                 .join(hands.lazy().select('hand_id'), on='hand_id', how='inner')
                 .with_columns(pl.col('player_id').is_in(list(posp)).alias('bad'))
                 .group_by('hand_id').agg(pl.col('bad').any().alias('any_pos'))
                 .collect(engine='streaming'))
        flags = seats
        flags.write_parquet(flag_path)
    hands = hands.join(flags, on='hand_id', how='inner')
    honest = hands.filter(~pl.col('any_pos')).drop('any_pos')
    print(f'honest dev hands {honest.height} of {hands.height} '
          f'({honest.height / hands.height:.3f})', flush=True)

    tables = sorted(hands['table_id'].unique().to_list())
    split = S.split_tables(tables)
    print({k: len(v) for k, v in split.items()}, flush=True)

    rng = np.random.default_rng(SEED)
    picks = []
    per_table = honest.group_by('table_id').len().to_dict(as_series=False)
    avail = dict(zip(per_table['table_id'], per_table['len']))
    for grp, k in [('POLICY_TRAIN', N_TRAIN), ('POLICY_HELD', N_HELD), ('TWO_SAMPLE', N_SAMPLE)]:
        sub = honest.filter(pl.col('table_id').is_in(split[grp]))
        sub = sub.with_columns(pl.Series('u', rng.random(sub.height)))
        sub = (sub.sort('u').group_by('table_id', maintain_order=True).head(k)
               .with_columns(pl.lit(grp).alias('grp')).drop('u'))
        picks.append(sub)
        print(f'{grp}: {sub.height} hands from {sub["table_id"].n_unique()} tables', flush=True)
    sel = pl.concat(picks)
    sel.write_parquet(SCRATCH / 'sel_hands.parquet')

    ids = sel.select('hand_id')
    seats = S.fetch_rows(ids, 'seats')
    seats.write_parquet(SCRATCH / 'sel_seats.parquet')
    print(f'seats {seats.height}  ({time.time() - t0:.0f}s)', flush=True)
    acts = S.fetch_rows(ids, 'actions')
    acts.write_parquet(SCRATCH / 'sel_actions.parquet')
    print(f'actions {acts.height}  ({time.time() - t0:.0f}s)', flush=True)

    meta = {
        'dev_hands': hands.height, 'honest_dev_hands': honest.height,
        'positive_players': len(posp),
        'tables': {k: len(v) for k, v in split.items()},
        'selected': sel.group_by('grp').len().to_dict(as_series=False),
        'seed': SEED, 'per_table': {'POLICY_TRAIN': N_TRAIN, 'POLICY_HELD': N_HELD,
                                    'TWO_SAMPLE': N_SAMPLE},
        'seconds': round(time.time() - t0, 1),
    }
    (HERE / 'prep_meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == '__main__':
    main()
