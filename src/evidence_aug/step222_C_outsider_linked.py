"""STEP222 features: Codex's LINKED outsider-card block (od_*, 24 fields) for any (pair, hand) keys of one phase.

Codex built the block write-once for the 3,240 audit rows only (Codex Resume/research/outsider_decisions_20260912/
linked.parquet). STEP221 found it lifts directed_transfer stage 2 (+.0105 on both halves). A production file needs
it for every development pool row and every evaluation top-20 row, so this module drives Codex's own functions
(read-only imports, bytecode off):
  outsider_decision_coverage.events_for_hand  -> outsider action rows of one hand
  (facing_pair rows only, sorted by action_no, as outsider_card_build.main does)
  outsider_card_build.arrays                   -> actor / pressure / action / strengths / aggressor / weights
  outsider_card_features.aggregate(...)['linked']
The per-hand inputs are read like marginal_payout_features.inputs, with the phase's own pair list.
GATE (features phase): the 3,240 development audit rows reproduce Codex's linked.parquet exactly.
Usage: python step222_C_outsider_linked.py features
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
OUT = CV / 'outputs_step222_20260917'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
CODEX_LINKED = POKER / 'Codex Resume/research/outsider_decisions_20260912/linked.parquet'
FEAT_OD = OUT / 'outsider_linked_dev.parquet'
KEYS = ['pair_id', 'hand_id']
SOURCES = ['outsider_decision_coverage.py', 'outsider_card_build.py', 'outsider_card_features.py',
           'retrospective_strength_features.py', 'contestable_state_features.py']


def _mods():
    import outsider_decision_coverage as ODC
    import outsider_card_build as OCB
    import outsider_card_features as OCF
    return ODC, OCB, OCF


def names():
    return list(_mods()[2].FEATURES)


def _task(task):
    ODC, OCB, OCF = _mods()
    hand, seats, acts, jobs = task
    out = []
    empty = None
    for pid, pair in jobs:
        rows, _ = ODC.events_for_hand(hand, seats, acts, pair)
        ev = pl.DataFrame(rows) if rows else None
        q = ev.filter(pl.col('facing_pair')).sort('action_no') if ev is not None else None
        if q is None or q.height == 0:
            if empty is None:
                empty = pl.DataFrame(schema={'action_no': pl.Int64, 'actor': pl.String, 'street': pl.String,
                                             'action': pl.String, 'pair_aggressors_this_street': pl.Int64,
                                             'own_strength': pl.Float64, 'last_aggressor_strength': pl.Float64,
                                             'log_pot_bb': pl.Float64})
            q = empty
        res = OCF.aggregate(*OCB.arrays(hand, seats, pair, q))['linked']
        out.append((pid, hand['hand_id'], [float(res[f]) for f in OCF.FEATURES]))
    return out


def build(keys, pair_players, phase, out_path, workers=6):
    """keys: (pair_id, hand_id) rows of one phase; pair_players: pair_id -> (player_1, player_2)."""
    ODC, OCB, OCF = _mods()
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
    jm = {}
    for pid, hid in keys.select(KEYS).iter_rows():
        jm.setdefault(hid, []).append((pid, tuple(pair_players[pid])))
    tasks = [(hm[h], sm[h], am.get(h, []), jm[h]) for h in hids]
    print(f'{keys.height} rows, {len(hids)} {phase} hands, {acts.height} actions  {round(time.perf_counter() - t0)}s', flush=True)
    res = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=32)):
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
    assert not FEAT_OD.exists(), 'preserve a built feature file'
    keys = pl.read_parquet(FEAT4).select(KEYS)
    assert keys.height == 45129
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    feat = build(keys, pp, 'development', FEAT_OD, workers=int(os.environ.get('OD_WORKERS', '6')))
    codex = pl.read_parquet(CODEX_LINKED)
    nm = names()
    j = codex.join(feat, on=KEYS, how='inner', validate='1:1', suffix='_mine')
    assert j.height == codex.height == 3240, 'gate rows missing'
    gate = {c: float(np.abs(j[c].to_numpy() - j[c + '_mine'].to_numpy()).max()) for c in nm}
    print('gate vs Codex linked.parquet (max abs):', max(gate.values()), 'rows', j.height, flush=True)
    assert max(gate.values()) == 0.0, 'does not reproduce Codex linked.parquet'
    rec = {'rows': feat.height, 'phase_asserted': 'development', 'gate_rows': j.height, 'gate_max_abs': max(gate.values()),
           'codex_linked_sha256': hashlib.sha256(CODEX_LINKED.read_bytes()).hexdigest(),
           'codex_source_sha256': {s: hashlib.sha256((POKER / 'Codex Resume/src' / s).read_bytes()).hexdigest() for s in SOURCES},
           'columns': nm, 'nonzero_share': {c: round(float((feat[c] != 0).mean()), 4) for c in nm},
           'sha256': hashlib.sha256(FEAT_OD.read_bytes()).hexdigest()}
    (OUT / 'receipt_features222.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('columns', 'nonzero_share')}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
