"""STEP246 features: the STEP237 invisible-event block (25 per-hand pair features) for any (pair, hand) keys
of ONE phase, so the block can be carried into a production build.

STEP237 built the block for every hand of every labelled DEVELOPMENT pair
(experiments/step237_invisible_events_20260917/step237_features.parquet, 223,749 x 27) with the phase hard
wired.  A production file needs the same 25 numbers for the 45,129 development top-20 rows that stage 2 fits on
and the 321,200 evaluation top-20 rows that stage 2 scores.  This module is only plumbing: every number is
produced by STEP237's own functions, imported read-only --

    step237_features.hand_tables          -> per-player street strengths and the final-board strength
    step237_features.pairhand_features    -> the 25 features of one (pair, hand)

so the two builds cannot diverge numerically.  The phase is asserted on every hand, and the pair membership of a
phase comes only from that phase's own pair list (development_labels.csv / evaluation_pairs.csv); nothing is
ever joined across phases by player or pair identity.

GATE (features phase): the development build reproduces step237_features.parquet value-for-value (max abs
exactly 0.0) on all 45,129 production development rows before any evaluation key is touched.

Usage: python step246_invisible_features.py features
"""
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True

CV = POKER / 'Claude Version'
DATA = POKER / 'data'
OUT = CV / 'outputs_step246_20260917'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
S237_TABLE = CV / 'experiments/step237_invisible_events_20260917/step237_features.parquet'
FEAT_DEV = OUT / 'invisible_dev.parquet'
KEYS = ['pair_id', 'hand_id']


def _mod():
    from evidence_aug import step237_features as S
    return S


FEATURES = None


def names():
    return list(_mod().FEATURES)


def _task(task):
    S = _mod()
    meta, seat_rows, act_rows, jobs = task
    hid = meta['hand_id']
    m = {hid: {'bb': meta['big_blind'], 'board': (meta['board_cards'] or '').split(),
               'final_pot': meta['final_pot'], 'players_dealt': meta['players_dealt']}}
    sdf = pl.DataFrame({'hand_id': [r[0] for r in seat_rows], 'player_id': [r[1] for r in seat_rows],
                        'hole_card_1': [r[2] for r in seat_rows], 'hole_card_2': [r[3] for r in seat_rows]})
    strength, final = S.hand_tables(sdf, m)
    srows = {r[1]: {'hand_id': r[0], 'player_id': r[1], 'hole_card_1': r[2], 'hole_card_2': r[3],
                    'total_contribution': r[4], 'net_chips': r[5], 'folded': r[6],
                    'went_to_showdown': r[7], 'won_share': r[8]} for r in seat_rows}
    out = []
    for pid, pair in jobs:
        if act_rows:
            f = S.pairhand_features(pid, hid, pair[0], pair[1], m[hid], srows, act_rows,
                                    strength[hid], final[hid])
        else:  # a hand with no action row cannot produce an event; STEP237 skipped these
            f = {k: 0. for k in S.FEATURES}
            f['fp_price_min'] = 1.0
        out.append((pid, hid, [float(f[k]) for k in S.FEATURES], int(not act_rows)))
    return out


def build(keys, pair_players, phase, out_path, workers=4, chunk_hands=40000):
    """keys: (pair_id, hand_id) rows of ONE phase; pair_players: pair_id -> (player_1, player_2)."""
    S = _mod()
    nm = list(S.FEATURES)
    t0 = time.perf_counter()
    keys = keys.select(KEYS)
    hids = keys['hand_id'].unique().to_list()
    meta_all = (pl.scan_parquet(DATA / 'hands.parquet')
                .filter(pl.col('hand_id').is_in(hids))
                .select('hand_id', 'phase', 'big_blind', 'board_cards', 'final_pot', 'players_dealt').collect())
    assert meta_all.height == len(hids), 'hands missing'
    bad = meta_all.filter(pl.col('phase') != phase).height
    assert bad == 0, f'{bad} hands are not {phase}-phase'
    jm = {}
    for pid, hid in keys.iter_rows():
        jm.setdefault(hid, []).append((pid, tuple(pair_players[pid])))
    res, no_action = [], 0
    bounds = list(range(0, len(hids), chunk_hands)) + [len(hids)]
    for ci in range(len(bounds) - 1):
        part = pl.DataFrame({'hand_id': hids[bounds[ci]:bounds[ci + 1]]})
        meta = meta_all.join(part, on='hand_id')
        seats = (pl.scan_parquet(DATA / 'seats.parquet').join(part.lazy(), on='hand_id')
                 .select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2', 'total_contribution',
                         'net_chips', 'folded', 'went_to_showdown', 'won_share').collect())
        acts = (pl.scan_parquet(DATA / 'actions.parquet').join(part.lazy(), on='hand_id')
                .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount',
                        'amount_to', 'pot_before', 'to_call').collect().sort('hand_id', 'action_no'))
        sm, am = {}, {}
        for r in seats.iter_rows():
            sm.setdefault(r[0], []).append(r)
        for r in acts.iter_rows(named=True):
            am.setdefault(r['hand_id'], []).append(r)
        tasks = [(mr, sm[mr['hand_id']], am.get(mr['hand_id'], []), jm[mr['hand_id']])
                 for mr in meta.iter_rows(named=True)]
        with Pool(workers) as pool:
            for r in pool.imap_unordered(_task, tasks, chunksize=64):
                res.extend(r)
        print(f'  chunk {ci + 1}/{len(bounds) - 1}: {len(res)} rows  {round(time.perf_counter() - t0)}s',
              flush=True)
    no_action = sum(z for _, _, _, z in res)
    out = pl.DataFrame({'pair_id': [p for p, _, _, _ in res], 'hand_id': [h for _, h, _, _ in res],
                        **{c: [v[i] for _, _, v, _ in res] for i, c in enumerate(nm)}},
                       schema_overrides={c: pl.Float64 for c in nm})
    out = keys.join(out, on=KEYS, how='left', validate='1:1', maintain_order='left')
    assert out.height == keys.height and out.select(nm).null_count().sum_horizontal().item() == 0
    out = out.with_columns(pl.col(nm).cast(pl.Float32))
    v = out.select(nm).to_numpy()
    assert np.isfinite(v).all()
    out.write_parquet(out_path)
    print(f'wrote {Path(out_path).name} {out.shape}  rows_with_no_action={no_action}  '
          f'{round(time.perf_counter() - t0)}s', flush=True)
    return out


def phase_features():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not FEAT_DEV.exists(), 'preserve a built feature file'
    keys = pl.read_parquet(FEAT4).select(KEYS)
    assert keys.height == 45129
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r[0]: (r[1], r[2]) for r in lab.iter_rows()}
    feat = build(keys, pp, 'development', FEAT_DEV, workers=int(os.environ.get('INV_WORKERS', '4')))
    nm = names()
    ref = pl.read_parquet(S237_TABLE)
    j = feat.join(ref, on=KEYS, how='inner', validate='1:1', suffix='_ref')
    assert j.height == feat.height == 45129, 'gate rows missing'
    gate = {c: float(np.abs(j[c].to_numpy().astype('float64')
                            - j[c + '_ref'].to_numpy().astype('float64')).max()) for c in nm}
    print('gate vs step237_features.parquet (max abs):', max(gate.values()), 'rows', j.height, flush=True)
    assert max(gate.values()) == 0.0, 'does not reproduce step237_features.parquet'
    rec = {'rows': feat.height, 'phase_asserted': 'development', 'gate_rows': j.height,
           'gate_max_abs': max(gate.values()), 'gate_per_column': gate,
           'step237_table_sha256': hashlib.sha256(S237_TABLE.read_bytes()).hexdigest(),
           'step237_builder_sha256': hashlib.sha256(
               (CV / 'src/evidence_aug/step237_features.py').read_bytes()).hexdigest(),
           'columns': nm, 'nonzero_share': {c: round(float((feat[c] != 0).mean()), 4) for c in nm},
           'sha256': hashlib.sha256(FEAT_DEV.read_bytes()).hexdigest()}
    (OUT / 'receipt_features246.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ('columns', 'nonzero_share', 'gate_per_column')}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
