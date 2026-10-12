"""STEP224 features: Codex's LAYERED marginal-payout block (mp_*, 32 fields) for any (pair, hand) keys of one phase.

Codex built the block write-once for the 3,240 audit rows only (Codex Resume/research/marginal_payout_20260912/
full-layered.parquet; `layered_payout` was the protocol's primary arm). A production file needs it for every
development pool row and every evaluation top-20 row, so this module drives Codex's own code (read-only imports,
bytecode off): marginal_payout_engine.HandPayouts(cards, board) once per hand, then
marginal_payout_features.extract(hand, seats, actions, pair, engine)[0]['layered'] per pair.
The per-hand inputs are read as marginal_payout_features.inputs reads them, with the phase's own pair list.
GATE (features phase): the 3,240 development audit rows reproduce Codex's full-layered.parquet exactly.
Usage: python step224_C_marginal_payout.py features
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
OUT = CV / 'outputs_step224_20260917'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
CODEX_MP = POKER / 'Codex Resume/research/marginal_payout_20260912/full-layered.parquet'
FEAT_MP = OUT / 'marginal_payout_layered_dev.parquet'
KEYS = ['pair_id', 'hand_id']
SOURCES = ['marginal_payout_features.py', 'marginal_payout_engine.py', 'contestable_state_features.py']


def _mods():
    import marginal_payout_features as MPF
    import marginal_payout_engine as MPE
    return MPF, MPE


def names():
    return list(_mods()[0].FEATURES)


def _task(task):
    MPF, MPE = _mods()
    from phevaluator import Card
    hand, seats, acts, jobs = task
    cards = {s['player_id']: (Card.to_id(s['hole_card_1']), Card.to_id(s['hole_card_2'])) for s in seats}
    board = [Card.to_id(c) for c in (hand['board_cards'] or '').split()]
    engine = MPE.HandPayouts(cards, board)
    out = []
    for pid, pair in jobs:
        feats, _ = MPF.extract(hand, seats, acts, pair, engine)
        out.append((pid, hand['hand_id'], [float(feats['layered'][f]) for f in MPF.FEATURES]))
    return out


def build(keys, pair_players, phase, out_path, workers=12):
    """keys: (pair_id, hand_id) rows of one phase; pair_players: pair_id -> (player_1, player_2)."""
    t0 = time.perf_counter()
    hids = keys['hand_id'].unique().to_list()
    hands = (pl.scan_parquet(DATA / 'hands.parquet').filter(pl.col('hand_id').is_in(hids))
             .select('hand_id', 'phase', 'button_seat', 'small_blind', 'big_blind', 'players_dealt', 'board_cards').collect())
    assert hands.height == len(hids), 'hands missing'
    bad = hands.filter(pl.col('phase') != phase).height
    assert bad == 0, f'{bad} hands are not {phase}-phase'
    seats = (pl.scan_parquet(DATA / 'seats.parquet').filter(pl.col('hand_id').is_in(hids))
             .select('hand_id', 'player_id', 'seat_no', 'starting_stack', 'total_contribution', 'hole_card_1', 'hole_card_2').collect())
    acts = pl.scan_parquet(DATA / 'actions.parquet').filter(pl.col('hand_id').is_in(hids)).sort('hand_id', 'action_no').collect()
    hm = {x['hand_id']: x for x in hands.drop('phase').to_dicts()}
    sm = {k[0]: g.to_dicts() for k, g in seats.group_by('hand_id')}
    am = {k[0]: g.to_dicts() for k, g in acts.group_by('hand_id')}
    assert len(sm) == len(am) == len(hids)
    jm = {}
    for pid, hid in keys.select(KEYS).iter_rows():
        jm.setdefault(hid, []).append((pid, tuple(pair_players[pid])))
    # longest hands first spreads the slow postflop enumerations across workers
    tasks = sorted(((hm[h], sm[h], am[h], jm[h]) for h in hids), key=lambda t: -len(t[2]) * len(t[3]))
    print(f'{keys.height} rows, {len(hids)} {phase} hands, {acts.height} actions  {round(time.perf_counter() - t0)}s', flush=True)
    res = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=16)):
            res.extend(r)
            if (i + 1) % 20000 == 0:
                print(f'  hands {i + 1}/{len(tasks)}  {round(time.perf_counter() - t0)}s', flush=True)
    nm = names()
    out = pl.DataFrame({'pair_id': [p for p, _, _ in res], 'hand_id': [h for _, h, _ in res],
                        **{c: [v[i] for _, _, v in res] for i, c in enumerate(nm)}},
                       schema_overrides={c: pl.Float64 for c in nm})
    out = keys.select(KEYS).join(out, on=KEYS, how='left', validate='1:1', maintain_order='left')
    assert out.height == keys.height and out.select(nm).null_count().sum_horizontal().item() == 0
    v = out.select(nm).to_numpy()
    assert np.isfinite(v).all() and (v >= 0).all()
    out.write_parquet(out_path)
    print(f'wrote {out_path.name} {out.shape}  {round(time.perf_counter() - t0)}s', flush=True)
    return out


def phase_features():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not FEAT_MP.exists(), 'preserve a built feature file'
    keys = pl.read_parquet(FEAT4).select(KEYS)
    assert keys.height == 45129
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    feat = build(keys, pp, 'development', FEAT_MP, workers=int(os.environ.get('MP_WORKERS', '12')))
    codex = pl.read_parquet(CODEX_MP)
    nm = names()
    j = codex.join(feat, on=KEYS, how='inner', validate='1:1', suffix='_mine')
    assert j.height == codex.height == 3240, 'gate rows missing'
    gate = {c: float(np.abs(j[c].to_numpy() - j[c + '_mine'].to_numpy()).max()) for c in nm}
    print('gate vs Codex full-layered.parquet (max abs):', max(gate.values()), 'rows', j.height, flush=True)
    assert max(gate.values()) == 0.0, 'does not reproduce Codex full-layered.parquet'
    rec = {'rows': feat.height, 'phase_asserted': 'development', 'gate_rows': j.height, 'gate_max_abs': max(gate.values()),
           'codex_layered_sha256': hashlib.sha256(CODEX_MP.read_bytes()).hexdigest(),
           'codex_source_sha256': {s: hashlib.sha256((POKER / 'Codex Resume/src' / s).read_bytes()).hexdigest() for s in SOURCES},
           'columns': nm, 'nonzero_share': {c: round(float((feat[c] != 0).mean()), 4) for c in nm},
           'sha256': hashlib.sha256(FEAT_MP.read_bytes()).hexdigest()}
    (OUT / 'receipt_features224.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('columns', 'nonzero_share')}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
