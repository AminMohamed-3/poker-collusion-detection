"""STEP280: validity check on the SIMULATED output (not a read).

Feeds the emitted hands/seats/actions back through the same engine replay used to
validate against real data. This checks the emission code and the legality of every
simulated action; it says nothing about realism.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_lib as S
from build_decisions import replay_hand, Mismatch
from prep_data import SCRATCH

HERE = Path(__file__).resolve().parent


def main(suffix=''):
    h = pl.read_parquet(SCRATCH / f'sim_hands{suffix}.parquet')
    s = pl.read_parquet(SCRATCH / f'sim_seats{suffix}.parquet')
    a = pl.read_parquet(SCRATCH / f'sim_actions{suffix}.parquet').sort('hand_id', 'action_no')
    pf = S.build_preflop_equity()
    sm, am = defaultdict(list), defaultdict(list)
    for r in s.iter_rows(named=True):
        sm[r['hand_id']].append(r)
    for r in a.iter_rows(named=True):
        am[r['hand_id']].append(r)
    rng = np.random.default_rng(1)
    bad, nbad, ex = 0, 0, []
    for row in h.iter_rows(named=True):
        try:
            *_, problems = replay_hand(row, sm[row['hand_id']], am[row['hand_id']], pf, rng, False)
        except (Mismatch, S.IllegalAction, StopIteration) as exc:
            bad += 1
            if len(ex) < 6:
                ex.append(f'{row["hand_id"]}: {type(exc).__name__}: {exc}')
            continue
        if problems:
            nbad += 1
            if len(ex) < 6:
                ex.append(f'{row["hand_id"]}: {problems[:3]}')
    cardsets = s.group_by('hand_id').agg(
        pl.concat_list('hole_card_1', 'hole_card_2').flatten().n_unique().alias('u'))
    rep = {
        'hands': h.height, 'illegal_or_inconsistent': bad, 'outcome_mismatch': nbad,
        'self_consistent_rate': round((h.height - bad - nbad) / max(h.height, 1), 6),
        'hands_with_12_distinct_hole_cards': int((cardsets['u'] == 12).sum()),
        'null_cells': {c: int(h[c].null_count()) for c in h.columns if h[c].null_count()},
        'examples': ex,
    }
    print(json.dumps(rep, indent=1), flush=True)
    (HERE / f'sim_validity{suffix}.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--suffix', default='')
    main(p.parse_args().suffix)
