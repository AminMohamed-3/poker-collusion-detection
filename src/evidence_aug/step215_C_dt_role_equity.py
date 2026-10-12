"""STEP215 features: production's 12 heads-up-equity accumulators (root `src/equity_evidence_features.py`, the
`eq_*` block that DT and SP stage 2 already use) split by the pair's pool ROLE instead of summed over the two
members.

Production reports, per hand, sum and max over the two members (`extract_pair_hand`). These are the equity
cousins of the rs_ block whose role split carried STEP208 (+.027 on DT). The per-member values exist inside
the extractor; `per_member` retypes its loop only to keep them. The equity function itself, the card map and
the preflop sample table are the production module's own (read-only import, sha asserted).

GATE: donor + recipient and max(donor, recipient) reproduce production's eq_*_sum and eq_*_max (float32
storage) on every audit row; rows production could not extract (ValueError -> -999) must fail here too.
Phase is ASSERTED on every hand. Member order is (player_1, player_2), as in production, which matters for
the bit pattern of the complemented equity.

Columns: dt_donor_eq_<12 base>, dt_recip_eq_<12 base>  (24).
Usage: python step215_C_dt_role_equity.py features
"""
import hashlib
import importlib.util
import json
import math
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
OUT = CV / 'outputs_step215_20260917'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
FEAT_ROLE = CV / 'outputs_step206_20260917/dt_roles_dev.parquet'
FEAT_EQ = OUT / 'dt_role_equity_dev.parquet'
EQ_SRC = POKER / 'src/equity_evidence_features.py'
EQ_SHA = '9ff36fdee8f965fb32fba111a53fc006382141f62e2fb27320937a2d3f6678f8'
KEYS = ['pair_id', 'hand_id']
_EQ = None


def eqmod():
    global _EQ
    if _EQ is None:
        assert hashlib.sha256(EQ_SRC.read_bytes()).hexdigest() == EQ_SHA, 'production equity module changed'
        spec = importlib.util.spec_from_file_location('production_equity_evidence_features', EQ_SRC)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _EQ = m
    return _EQ


def names():
    E = eqmod()
    return ['dt_donor_eq_' + n for n in E.NAMES] + ['dt_recip_eq_' + n for n in E.NAMES]


def per_member(first, second, big_blind, seats, board, actions):
    """production extract_pair_hand, returning {player: {name: value}} instead of sum/max."""
    E = eqmod()
    NAMES = E.NAMES
    pair = {first, second}
    if len(pair) != 2 or not pair.issubset(seats) or big_blind <= 0:
        raise ValueError('Invalid pair, seats or blind')
    live = set(seats)
    values = {p: dict.fromkeys(NAMES, 0.) for p in pair}
    street, previous = None, None
    equity = {}
    prior_equity = {}
    ordered = sorted(actions, key=lambda a: a['action_no'])
    if len({a['action_no'] for a in ordered}) != len(ordered):
        raise ValueError('Duplicate action number')
    for action in ordered:
        if action['street'] != street:
            street, previous = action['street'], None
        player = action['player_id']
        if player not in live:
            raise ValueError('Action after fold')
        kind, amount, to_call = action['action'], action['amount'], action['to_call']
        aggressive = kind in ('bet', 'raise', 'all_in') and amount > to_call
        calling = kind in ('call', 'all_in') and not aggressive
        if player in pair and pair.issubset(live):
            partner = second if player == first else first
            to_partner = previous == partner
            relevant = (to_partner and (calling or kind == 'fold' or aggressive)) or kind == 'check' or aggressive
            if relevant:
                if street not in equity:
                    n = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}[street]
                    if len(board) < n:
                        raise ValueError('Betting street exceeds disclosed board')
                    equity[street] = E.heads_up_equity(seats[first], seats[second], board[:n])
                q = equity[street] if player == first else 1 - equity[street]
                cost = min(to_call, action.get('stack_before', to_call))
                pot = action['pot_before']
                price = cost / max(pot + cost, 1.)
                amount_bb = amount / big_blind
                surplus = max(0., q * (pot + cost) - cost) / big_blind
                deficit = max(0., cost - q * (pot + cost)) / big_blind
                v = values[player]
                past = prior_equity.get(player, q)
                if to_partner and calling:
                    v['partner_call_deficit_bb'] += deficit
                    v['partner_call_surplus_bb'] += surplus
                    v['partner_call_equity_cost_bb'] += q * amount_bb
                    v['partner_call_equity_drop_bb'] += max(0., past - q) * amount_bb
                if to_partner and kind == 'fold':
                    v['partner_fold_surplus_pot_bb'] += surplus
                    v['partner_fold_equity'] += q * (1 - price)
                    v['partner_fold_underdog'] += (1 - q) * (1 - price)
                    v['partner_fold_equity_gain'] += max(0., q - past)
                if kind == 'check':
                    v['late_check_equity'] += q * (street in ('turn', 'river'))
                    v['heads_up_check_equity'] += q * (len(live) == 2)
                if aggressive:
                    v['partner_raise_equity_cost_bb'] += q * amount_bb * to_partner
                    v['multiway_raise_low_equity_logcost'] += (1 - q) * math.log1p(amount_bb) * (len(live) >= 3)
                prior_equity[player] = q
        if aggressive:
            previous = player
        if kind == 'fold':
            live.remove(player)
    for p in pair:
        if not all(math.isfinite(x) and x >= 0 for x in values[p].values()):
            raise ValueError('Invalid equity feature')
    return values


def _hand_task(task):
    """task: (hand_id, big_blind, board_str, seats [(player, c1, c2)], actions [dict], [(pair_id, p1, p2, sign)])"""
    E = eqmod()
    hid, bb, board_str, seat_rows, acts, jobs = task
    seats = {p: (E.CARDS[c1], E.CARDS[c2]) for p, c1, c2 in seat_rows}
    board = [E.CARDS[c] for c in (board_str or '').split()]
    out = []
    for pid, p1, p2, sgn in jobs:
        try:
            vals = per_member(p1, p2, bb, seats, board, acts)
        except ValueError:
            out.append((pid, hid, None))
            continue
        donor, recip = (p1, p2) if sgn > 0 else (p2, p1)
        out.append((pid, hid, [vals[donor][n] for n in E.NAMES] + [vals[recip][n] for n in E.NAMES]))
    return out


def build(keys_sign, pair_players, phase, out_path, check_eq=None, workers=6):
    """keys_sign: (pair_id, hand_id, dt_sign) rows to write; dt_sign from STEP206c (+1: player_2 receives).
    check_eq: optional frame (pair_id, hand_id, eq_* production columns, -999 = not extracted) for the gate."""
    E = eqmod()
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
            .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call',
                    'pot_before', 'stack_before').sort('hand_id', 'action_no').collect())
    sm = {}
    for h, p, c1, c2 in seats.iter_rows():
        sm.setdefault(h, []).append((p, c1, c2))
    am = {}
    for row in acts.iter_rows(named=True):
        am.setdefault(row.pop('hand_id'), []).append(row)
    jm = {}
    for pid, hid, sgn in keys_sign.select(KEYS + ['dt_sign']).iter_rows():
        p1, p2 = pair_players[pid]
        jm.setdefault(hid, []).append((pid, p1, p2, sgn))
    tasks = [(h, bb, bc, sm[h], am.get(h, []), jm[h]) for h, _, bb, bc in hands.iter_rows()]
    print(f'{keys_sign.height} rows, {len(hids)} {phase} hands, {acts.height} actions  '
          f'{round(time.perf_counter() - t0)}s', flush=True)
    res = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_hand_task, tasks, chunksize=64)):
            res.extend(r)
            if (i + 1) % 10000 == 0:
                print(f'  hands {i + 1}/{len(tasks)}  {round(time.perf_counter() - t0)}s', flush=True)
    nm = names()
    failed = [(p, h) for p, h, v in res if v is None]
    out = (pl.DataFrame({'pair_id': [p for p, _, _ in res], 'hand_id': [h for _, h, _ in res],
                         **{c: [(v[i] if v is not None else None) for _, _, v in res] for i, c in enumerate(nm)}},
                        schema_overrides={c: pl.Float64 for c in nm})
           .sort(KEYS))
    ref_order = keys_sign.select(KEYS).sort(KEYS)
    assert out.select(KEYS).equals(ref_order), 'row set differs from the requested keys'
    out = keys_sign.select(KEYS).join(out, on=KEYS, how='left', validate='1:1', maintain_order='left')
    print(f'extracted {out.height - len(failed)} rows, {len(failed)} not extractable (production raises)', flush=True)
    gate = None
    if check_eq is not None:
        j = check_eq.join(out, on=KEYS, how='inner', validate='1:1')
        assert j.height == check_eq.height, 'gate rows missing'
        prod_missing = (j['eq_' + E.NAMES[0] + '_sum'] <= -998).to_numpy()
        mine_missing = j[nm[0]].is_null().to_numpy()
        gate = {'missing_rows_production': int(prod_missing.sum()), 'missing_rows_here': int(mine_missing.sum()),
                'missing_pattern_equal': bool((prod_missing == mine_missing).all())}
        ok = ~prod_missing
        dif = {}
        for n in E.NAMES:
            d_, r_ = j['dt_donor_eq_' + n].to_numpy()[ok], j['dt_recip_eq_' + n].to_numpy()[ok]
            ps, pm = j['eq_' + n + '_sum'].to_numpy()[ok], j['eq_' + n + '_max'].to_numpy()[ok]
            dif[n + '_sum'] = float(np.abs((d_ + r_).astype(np.float32) - ps).max())
            dif[n + '_max'] = float(np.abs(np.maximum(d_, r_).astype(np.float32) - pm).max())
        gate['max_abs_float32'] = max(dif.values())
        gate['detail'] = dif
        print('gate vs production eq_:', {k: v for k, v in gate.items() if k != 'detail'}, 'rows', j.height, flush=True)
        assert gate['missing_pattern_equal'], 'extraction failures differ from production'
        assert gate['max_abs_float32'] == 0.0, 'role split does not reproduce production eq_ columns'
    if failed:
        out = out.with_columns(pl.col(nm).fill_null(-999.))
    assert np.isfinite(out.select(nm).to_numpy()).all()
    out.write_parquet(out_path)
    print(f'wrote {out_path.name} {out.shape}  {round(time.perf_counter() - t0)}s', flush=True)
    return out, gate, len(failed)


def phase_features(limit=None):
    from evidence_aug import step165_C_e2e_audit as S165
    E = eqmod()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = FEAT_EQ if limit is None else OUT / f'_probe_{limit}.parquet'
    assert not dest.exists(), 'preserve a built feature file'
    ks = pl.read_parquet(FEAT_ROLE).select(KEYS + ['dt_sign'])
    assert ks.height == 45129
    assert ks.select(KEYS).equals(pl.read_parquet(FEAT4).select(KEYS))
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    D = S165.load()
    eqc = ['eq_' + n + s for s in ('_sum', '_max') for n in E.NAMES]
    assert set(eqc) <= set(D['rcols'])
    check = D['Q'].select(KEYS + eqc)
    if limit is not None:
        check = check.head(limit)
        ks = ks.join(check.select(KEYS), on=KEYS, how='semi')
    feat, gate, nfail = build(ks, pp, 'development', dest, check_eq=check, workers=int(os.environ.get('EQ_WORKERS', '6')))
    if limit is not None:
        print('probe ok', gate['max_abs_float32'], flush=True)
        return
    nz = {c: round(float((feat[c] > 0).mean()), 4) for c in names()}
    rec = {'rows': feat.height, 'phase_asserted': 'development', 'not_extractable_rows': nfail,
           'gate_rows': check.height, 'gate': gate, 'equity_module_sha256': EQ_SHA,
           'columns': names(), 'positive_share': nz,
           'sources': {str(FEAT_ROLE.relative_to(CV)): hashlib.sha256(FEAT_ROLE.read_bytes()).hexdigest()},
           'sha256': hashlib.sha256(FEAT_EQ.read_bytes()).hexdigest()}
    (OUT / 'receipt_features215.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('gate', 'columns', 'positive_share')}), flush=True)
    print(json.dumps(nz), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    elif sys.argv[1] == 'probe':
        phase_features(int(sys.argv[2]))
    else:
        raise SystemExit('usage: features | probe N')
