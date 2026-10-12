"""STEP225 features: the TERMINAL-board version of STEP208's role-split strength block.

STEP208 (DTs16, the largest DT gain so far) splits production's prefix-mode rs_ accumulators by pool role. Codex's
extractor (retrospective_strength_features.extract) computes the same 16 per-player accumulators in a second
mode, `terminal`: every decision is weighted by the player's strength on the FINAL board instead of the board
prefix at the decision ("did the donor call the partner's bet with what turned out to be the losing hand").
Production carries no terminal field at all. This module keeps the per-player terminal accumulators and orders
them by STEP206's pool direction, exactly as STEP208 does for prefix mode.

The loop is Codex's `extract`, retyped only to keep per-player terminal values (read-only import of its
constants and `strength`). GATE: donor + recipient and max(donor, recipient) reproduce Codex's write-once
terminal.parquet (research/retrospective_strength_20260912, 3,240 audit rows) EXACTLY. Phase is asserted on
every hand. Board and hole cards are public hand data; no label is read.

Columns: dt_donor_rt_<16 base>, dt_recip_rt_<16 base>  (32).
Usage: python step225_C_dt_role_terminal.py features
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True

CV = POKER / 'Claude Version'
DATA = POKER / 'data'
OUT = CV / 'outputs_step225_20260917'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
FEAT_ROLE = CV / 'outputs_step206_20260917/dt_roles_dev.parquet'
FEAT_RS = CV / 'outputs_step208_20260917/dt_role_strength_dev.parquet'
CODEX_TERM = POKER / 'Codex Resume/research/retrospective_strength_20260912/terminal.parquet'
FEAT_RT = OUT / 'dt_role_terminal_dev.parquet'
KEYS = ['pair_id', 'hand_id']


def _rsf():
    import retrospective_strength_features as RSF
    return RSF


def names():
    RSF = _rsf()
    return (['dt_donor_rt_' + b for b in RSF.BASE] + ['dt_recip_rt_' + b for b in RSF.BASE])


def per_player(hand, seats, events, pair):
    """Codex extract(), terminal mode only, returning the per-player accumulators {player: {base: value}}."""
    RSF = _rsf()
    from phevaluator import Card
    cards = {s['player_id']: (Card.to_id(s['hole_card_1']), Card.to_id(s['hole_card_2'])) for s in seats}
    board = [Card.to_id(c) for c in (hand['board_cards'] or '').split()]
    terminal = {p: RSF.strength(cards[p], board) for p in pair}
    acc = {p: dict.fromkeys(RSF.BASE, 0.) for p in pair}
    live = set(cards)
    street = None
    previous = None
    for a in events:
        if a['street'] != street:
            street = a['street']
            previous = None
        n = RSF.STREET[street]
        assert len(board) >= n
        actor = a['player_id']
        kind = a['action']
        amount = int(a['amount'])
        tocall = int(a['to_call'])
        aggressive = kind in ('bet', 'raise', 'all_in') and amount > tocall
        calling = kind in ('call', 'all_in') and not aggressive
        if actor in pair and set(pair).issubset(live):
            partner = pair[1] if actor == pair[0] else pair[0]
            action = 'aggr' if aggressive else 'call' if calling else kind
            assert action in ('aggr', 'call', 'fold', 'check')
            context = 'partner' if previous == partner else 'other'
            w = math.log1p(int(a['pot_before']) / int(hand['big_blind']))
            s = terminal
            acc[actor][f'{context}_{action}_strength'] += s[actor] * w
            acc[actor][f'{context}_{action}_ahead_gap'] += max(0., s[actor] - s[partner]) * w
        if aggressive:
            previous = actor
        if kind == 'fold':
            live.remove(actor)
    return acc


def build(keys_sign, pair_players, phase, out_path, check_rt=None):
    """keys_sign: (pair_id, hand_id, dt_sign) rows to write; dt_sign from STEP206c (+1: player_2 receives).
    check_rt: optional frame (pair_id, hand_id, Codex terminal rs_* columns) to gate the sums against."""
    RSF = _rsf()
    t0 = time.perf_counter()
    hids = keys_sign['hand_id'].unique().to_list()
    hands = (pl.scan_parquet(DATA / 'hands.parquet').filter(pl.col('hand_id').is_in(hids))
             .select('hand_id', 'phase', 'big_blind', 'board_cards').collect())
    assert hands.height == len(hids), 'hands missing'
    bad = hands.filter(pl.col('phase') != phase).height
    assert bad == 0, f'{bad} hands are not {phase}-phase'
    seats = (pl.scan_parquet(DATA / 'seats.parquet').filter(pl.col('hand_id').is_in(hids))
             .select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2').collect())
    acts = (pl.scan_parquet(DATA / 'actions.parquet').filter(pl.col('hand_id').is_in(hids))
            .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call', 'pot_before')
            .sort('hand_id', 'action_no').collect())
    hm = {x['hand_id']: x for x in hands.to_dicts()}
    sm = {k[0]: g.to_dicts() for k, g in seats.group_by('hand_id')}
    am = {k[0]: g.sort('action_no').to_dicts() for k, g in acts.group_by('hand_id')}
    print(f'{keys_sign.height} rows, {len(hids)} {phase} hands, {acts.height} actions  '
          f'{round(time.perf_counter() - t0)}s', flush=True)
    rows = []
    for pid, hid, sgn in keys_sign.select(KEYS + ['dt_sign']).iter_rows():
        p1, p2 = pair_players[pid]
        acc = per_player(hm[hid], sm[hid], am.get(hid, []), (p1, p2))
        donor, recip = (p1, p2) if sgn > 0 else (p2, p1)
        r = {'pair_id': pid, 'hand_id': hid}
        for b in RSF.BASE:
            r['dt_donor_rt_' + b] = acc[donor][b]
            r['dt_recip_rt_' + b] = acc[recip][b]
        rows.append(r)
    out = pl.DataFrame(rows).select(KEYS + names())
    assert out.height == keys_sign.height
    assert np.isfinite(out.select(names()).to_numpy()).all()
    gate = None
    if check_rt is not None:
        j = check_rt.join(out, on=KEYS, how='inner', validate='1:1')
        assert j.height == check_rt.height, 'gate rows missing'
        gate = {}
        for b in RSF.BASE:
            d, r_ = j['dt_donor_rt_' + b].to_numpy(), j['dt_recip_rt_' + b].to_numpy()
            gate['rs_' + b + '_sum'] = float(np.abs((d + r_) - j['rs_' + b + '_sum'].to_numpy()).max())
            gate['rs_' + b + '_max'] = float(np.abs(np.maximum(d, r_) - j['rs_' + b + '_max'].to_numpy()).max())
        print('gate vs Codex terminal rs_ (max abs):', max(gate.values()), 'rows', j.height, flush=True)
        assert max(gate.values()) == 0.0, 'role split does not reproduce Codex terminal.parquet'
    out.write_parquet(out_path)
    print(f'wrote {out_path.name} {out.shape}  {round(time.perf_counter() - t0)}s', flush=True)
    return out, gate


def phase_features():
    RSF = _rsf()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not FEAT_RT.exists(), 'preserve a built feature file'
    ks = pl.read_parquet(FEAT_ROLE).select(KEYS + ['dt_sign'])
    assert ks.height == 45129
    assert ks.select(KEYS).equals(pl.read_parquet(FEAT4).select(KEYS))
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    check = pl.read_parquet(CODEX_TERM)
    assert sorted(c for c in check.columns if c not in KEYS) == sorted(RSF.FEATURES) and check.height == 3240
    feat, gate = build(ks, pp, 'development', FEAT_RT, check_rt=check)
    # label-free comparison with the prefix block: where the board never grew past the decision, the two agree
    pre = pl.read_parquet(FEAT_RS)
    j = pre.join(feat, on=KEYS, validate='1:1')
    same = {b: float((j['dt_donor_rs_' + b] == j['dt_donor_rt_' + b]).mean()) for b in RSF.BASE}
    rec = {'rows': feat.height, 'phase_asserted': 'development', 'gate_rows': check.height,
           'gate_vs_codex_terminal_max_abs': max(gate.values()), 'gate_detail': gate,
           'codex_terminal_sha256': hashlib.sha256(CODEX_TERM.read_bytes()).hexdigest(),
           'codex_source_sha256': hashlib.sha256((POKER / 'Codex Resume/src/retrospective_strength_features.py').read_bytes()).hexdigest(),
           'donor_share_equal_to_prefix': same,
           'nonzero_share': {c: round(float((feat[c] != 0).mean()), 4) for c in names()},
           'sha256': hashlib.sha256(FEAT_RT.read_bytes()).hexdigest()}
    (OUT / 'receipt_features225.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('gate_detail', 'nonzero_share')}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
