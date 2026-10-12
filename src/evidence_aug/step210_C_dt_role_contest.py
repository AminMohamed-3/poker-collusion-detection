"""STEP210 features: production's contestable-state ACTION block (Codex contestable_state_features, `cs_*`)
split by the pair's pool role, the way STEP208 split the strength block.

Production counts, per hand, the pair members' folds / checks / calls / raises while the partner is still in
the hand (and the bb put in by calls and raises), SUMMED over the two members. The split keeps the member.
Only the partner-ACTIVE variants and the decision count are split (the partner-all-in variants are rare).

The event walk is Codex `state_features` reduced to what these fields need (fold / all-in tracking and the
action classification), retyped; the GATE is that donor + recipient reproduces production's
cs_decisions and cs_{fold,check,call,aggr}_partner_active_count EXACTLY and the two bb sums to 1e-9
(float addition order differs) on every audit row. Phase is ASSERTED on every hand.

Columns (14): dt_{donor,recip}_cs_{decisions, fold_pa, check_pa, call_pa, aggr_pa, call_pa_bb, aggr_pa_bb}
"""
import hashlib
import json
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
OUT = CV / 'outputs_step210_20260917'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
FEAT_ROLE = CV / 'outputs_step206_20260917/dt_roles_dev.parquet'
FEAT_CS = OUT / 'dt_role_contest_dev.parquet'
KEYS = ['pair_id', 'hand_id']
Q7 = ['decisions', 'fold_pa', 'check_pa', 'call_pa', 'aggr_pa', 'call_pa_bb', 'aggr_pa_bb']
PROD = {'decisions': 'cs_decisions', 'fold_pa': 'cs_fold_partner_active_count',
        'check_pa': 'cs_check_partner_active_count', 'call_pa': 'cs_call_partner_active_count',
        'aggr_pa': 'cs_aggr_partner_active_count', 'call_pa_bb': 'cs_call_partner_active_bb_sum',
        'aggr_pa_bb': 'cs_aggr_partner_active_bb_sum'}


def names():
    return ['dt_donor_cs_' + q for q in Q7] + ['dt_recip_cs_' + q for q in Q7]


def per_player(hand, seats, events, pair):
    p1, p2 = pair
    big, small, button = int(hand['big_blind']), int(hand['small_blind']), int(hand['button_seat'])
    seat_player = {int(r['seat_no']): r['player_id'] for r in seats}
    starting = {r['player_id']: int(r['starting_stack']) for r in seats}
    contribution = {p: 0 for p in starting}
    contribution[seat_player[(button + 1) % 6]] = small
    contribution[seat_player[(button + 2) % 6]] = big
    folded = set()
    all_in = {p for p in starting if starting[p] == contribution[p] and starting[p] > 0}
    acc = {p: dict.fromkeys(Q7, 0.0) for p in pair}
    for e in events:
        actor = e['player_id']
        amount = int(e['amount'])
        if actor in pair:
            partner = p2 if actor == p1 else p1
            if partner not in folded:
                partner_allin = partner in all_in
                a = acc[actor]
                a['decisions'] += 1
                if not partner_allin:
                    act = e['action']
                    tc = int(e['to_call'])
                    if act == 'fold':
                        a['fold_pa'] += 1
                    elif act == 'check':
                        a['check_pa'] += 1
                    elif act == 'call' or (act == 'all_in' and amount <= tc):
                        a['call_pa'] += 1
                        a['call_pa_bb'] += amount / big
                    elif act in ('bet', 'raise') or (act == 'all_in' and amount > tc):
                        a['aggr_pa'] += 1
                        a['aggr_pa_bb'] += amount / big
                    else:
                        raise ValueError('unknown action ' + act)
        contribution[actor] += amount
        if e['action'] == 'fold':
            folded.add(actor)
        if contribution[actor] == starting[actor] and starting[actor] > 0 and actor not in folded:
            all_in.add(actor)
    return acc


def build(keys_sign, pair_players, phase, out_path, check_cs=None):
    t0 = time.perf_counter()
    hids = keys_sign['hand_id'].unique().to_list()
    hands = (pl.scan_parquet(DATA / 'hands.parquet').filter(pl.col('hand_id').is_in(hids))
             .select('hand_id', 'phase', 'big_blind', 'small_blind', 'button_seat').collect())
    assert hands.height == len(hids), 'hands missing'
    bad = hands.filter(pl.col('phase') != phase).height
    assert bad == 0, f'{bad} hands are not {phase}-phase'
    seats = (pl.scan_parquet(DATA / 'seats.parquet').filter(pl.col('hand_id').is_in(hids))
             .select('hand_id', 'player_id', 'seat_no', 'starting_stack').collect())
    acts = (pl.scan_parquet(DATA / 'actions.parquet').filter(pl.col('hand_id').is_in(hids))
            .select('hand_id', 'action_no', 'player_id', 'action', 'amount', 'to_call')
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
        for q in Q7:
            r['dt_donor_cs_' + q] = acc[donor][q]
            r['dt_recip_cs_' + q] = acc[recip][q]
        rows.append(r)
    out = pl.DataFrame(rows).select(KEYS + names())
    assert out.height == keys_sign.height
    assert np.isfinite(out.select(names()).to_numpy()).all()
    gate = None
    if check_cs is not None:
        j = check_cs.join(out, on=KEYS, how='inner', validate='1:1')
        assert j.height == check_cs.height, 'gate rows missing'
        gate = {}
        for q, pc in PROD.items():
            s = j['dt_donor_cs_' + q].to_numpy() + j['dt_recip_cs_' + q].to_numpy()
            gate[pc] = float(np.abs(s - j[pc].to_numpy()).max())
        print('gate vs production cs_ (max abs):', json.dumps(gate), 'rows', j.height, flush=True)
        exact = [PROD[q] for q in Q7 if not q.endswith('_bb')]
        assert max(gate[c] for c in exact) == 0.0, 'counts do not reproduce production'
        assert max(gate.values()) < 1e-9, 'bb sums do not reproduce production'
    out.write_parquet(out_path)
    print(f'wrote {out_path.name} {out.shape}  {round(time.perf_counter() - t0)}s', flush=True)
    return out, gate


def phase_features():
    from evidence_aug import step165_C_e2e_audit as S165
    OUT.mkdir(parents=True, exist_ok=True)
    assert not FEAT_CS.exists(), 'preserve a built feature file'
    ks = pl.read_parquet(FEAT_ROLE).select(KEYS + ['dt_sign'])
    assert ks.height == 45129
    assert ks.select(KEYS).equals(pl.read_parquet(FEAT4).select(KEYS))
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    D = S165.load()
    assert set(PROD.values()) <= set(D['cs_cols'])
    check = D['Q'].select(KEYS + list(PROD.values()))
    feat, gate = build(ks, pp, 'development', FEAT_CS, check_cs=check)
    print(feat.select(names()).describe(), flush=True)
    rec = {'rows': feat.height, 'phase_asserted': 'development', 'gate_rows': check.height,
           'gate_vs_production_cs_max_abs': gate,
           'sha256': hashlib.sha256(FEAT_CS.read_bytes()).hexdigest()}
    (OUT / 'receipt_features210.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
