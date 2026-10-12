"""STEP208 features: production's 32 action-prefix strength fields (Codex retrospective_strength_features, the
`rs_*` block that stage-2 already uses) split by the pair's ROLE instead of summed over the two players.

Production reports, per hand, sum and max over the two players of 16 per-player accumulators
({partner, other} x {call, fold, check, aggr} x {strength, ahead_gap}, prefix mode). Both are symmetric, so
stage-2 cannot ask "did the DONOR call the partner's bet with a weak hand" -- the chip-dump signature -- only
"did either of them". The per-player accumulators exist inside the extractor; this keeps them and orders them
by STEP206's pool direction.

The accumulation loop is Codex's `extract` (read-only import of its constants and `strength`), retyped only
to keep the per-player values; the GATE is that donor + recipient and max(donor, recipient) reproduce
production's rs_*_sum and rs_*_max EXACTLY on every audit row. Phase is ASSERTED on every hand.

Columns: dt_donor_rs_<16 base>, dt_recip_rs_<16 base>  (32).
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
OUT = CV / 'outputs_step208_20260917'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
FEAT_ROLE = CV / 'outputs_step206_20260917/dt_roles_dev.parquet'
FEAT_RS = OUT / 'dt_role_strength_dev.parquet'
KEYS = ['pair_id', 'hand_id']


def _rsf():
    import retrospective_strength_features as RSF
    return RSF


def names():
    RSF = _rsf()
    return (['dt_donor_rs_' + b for b in RSF.BASE] + ['dt_recip_rs_' + b for b in RSF.BASE])


def per_player(hand, seats, events, pair):
    """Codex extract(), prefix mode only, returning the per-player accumulators {player: {base: value}}."""
    RSF = _rsf()
    from phevaluator import Card
    cards = {s['player_id']: (Card.to_id(s['hole_card_1']), Card.to_id(s['hole_card_2'])) for s in seats}
    board = [Card.to_id(c) for c in (hand['board_cards'] or '').split()]
    acc = {p: dict.fromkeys(RSF.BASE, 0.) for p in pair}
    live = set(cards)
    street = None
    previous = None
    cache = {}
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
            if n not in cache:
                cache[n] = {p: RSF.strength(cards[p], board[:n]) for p in pair}
            w = math.log1p(int(a['pot_before']) / int(hand['big_blind']))
            s = cache[n]
            acc[actor][f'{context}_{action}_strength'] += s[actor] * w
            acc[actor][f'{context}_{action}_ahead_gap'] += max(0., s[actor] - s[partner]) * w
        if aggressive:
            previous = actor
        if kind == 'fold':
            live.remove(actor)
    return acc


def build(keys_sign, pair_players, phase, out_path, check_rs=None):
    """keys_sign: (pair_id, hand_id, dt_sign) rows to write; dt_sign from STEP206c (+1: player_2 receives).
    check_rs: optional frame (pair_id, hand_id, rs_* production columns) to gate the sums against."""
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
            r['dt_donor_rs_' + b] = acc[donor][b]
            r['dt_recip_rs_' + b] = acc[recip][b]
        rows.append(r)
    out = pl.DataFrame(rows).select(KEYS + names())
    assert out.height == keys_sign.height
    assert np.isfinite(out.select(names()).to_numpy()).all()
    gate = None
    if check_rs is not None:
        j = check_rs.join(out, on=KEYS, how='inner', validate='1:1')
        assert j.height == check_rs.height, 'gate rows missing'
        gate = {}
        for b in RSF.BASE:
            d, r_ = j['dt_donor_rs_' + b].to_numpy(), j['dt_recip_rs_' + b].to_numpy()
            gate['rs_' + b + '_sum'] = float(np.abs((d + r_) - j['rs_' + b + '_sum'].to_numpy()).max())
            gate['rs_' + b + '_max'] = float(np.abs(np.maximum(d, r_) - j['rs_' + b + '_max'].to_numpy()).max())
        print('gate vs production rs_ (max abs):', max(gate.values()), 'rows', j.height, flush=True)
        assert max(gate.values()) == 0.0, 'role split does not reproduce production rs_ columns'
    out.write_parquet(out_path)
    print(f'wrote {out_path.name} {out.shape}  {round(time.perf_counter() - t0)}s', flush=True)
    return out, gate


def phase_features():
    from evidence_aug import step165_C_e2e_audit as S165
    RSF = _rsf()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not FEAT_RS.exists(), 'preserve a built feature file'
    ks = pl.read_parquet(FEAT_ROLE).select(KEYS + ['dt_sign'])
    assert ks.height == 45129
    assert ks.select(KEYS).equals(pl.read_parquet(FEAT4).select(KEYS))
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    D = S165.load()
    rs = [c for c in D['pf_cols']]
    assert sorted(rs) == sorted(RSF.FEATURES)
    check = D['Q'].select(KEYS + rs)
    feat, gate = build(ks, pp, 'development', FEAT_RS, check_rs=check)
    print(feat.select(names()).describe(), flush=True)
    rec = {'rows': feat.height, 'phase_asserted': 'development', 'gate_rows': check.height,
           'gate_vs_production_rs_max_abs': max(gate.values()), 'gate_detail': gate,
           'sha256': hashlib.sha256(FEAT_RS.read_bytes()).hexdigest()}
    (OUT / 'receipt_features208.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k != 'gate_detail'}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
