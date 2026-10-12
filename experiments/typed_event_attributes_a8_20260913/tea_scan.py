"""poker-a8 typed-event attributes (STEP112 interface): label-free scan of one pair-hand from raw hands/seats/actions.
Members come from the pair definition only. W/L follow step104/106 (W = higher net_chips, member a on ties); net_tie flags ties."""
import sys
sys.dont_write_bytecode = True
from collections import defaultdict
import polars as pl

DATA = 'C:/Data/Projects/Money Making/work/poker/data/'
SO = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
NM = ['pre', 'flop', 'turn', 'river']
MEMBER = ['folded', 'fold_street', 'fold_facing', 'fold_pot_bb', 'fold_to_call_bb', 'partner_aggr_before_fold', 'called_before_fold_same_street',
          'called_before_fold_earlier', 'n_calls_facing_partner', 'max_call_bb', 'max_call_street', 'max_call_pot_share', 'n_raises_facing_partner']


def aggressive(x):
    return x['action'] in ('bet', 'raise') or (x['action'] == 'all_in' and x['amount'] > x['to_call'])


def passive_call(x):
    return x['action'] == 'call' or (x['action'] == 'all_in' and x['amount'] <= x['to_call'])


def load(hids):
    hands = {r['hand_id']: r for r in pl.scan_parquet(DATA + 'hands.parquet').filter(pl.col('hand_id').is_in(hids)).drop('started_at').collect().iter_rows(named=True)}
    seats = defaultdict(dict)
    for r in pl.scan_parquet(DATA + 'seats.parquet').filter(pl.col('hand_id').is_in(hids)).collect().iter_rows(named=True):
        seats[r['hand_id']][r['player_id']] = r
    acts = defaultdict(list)
    for r in pl.scan_parquet(DATA + 'actions.parquet').filter(pl.col('hand_id').is_in(hids)).collect().sort('hand_id', 'action_no').iter_rows(named=True):
        acts[r['hand_id']].append(r)
    return hands, seats, acts


def scan(hand, se, ac, a, b):
    """Cheap fields plus member events (role, street, kind fold/br/calls/checks, facing_partner, amount_bb, pot_before_bb, partner_in)."""
    bb = max(1, hand['big_blind'])
    na, nb = se[a]['net_chips'], se[b]['net_chips']
    w, l = (a, b) if na >= nb else (b, a)
    opp = na * nb < 0
    r = dict(players_dealt=hand['players_dealt'], players_at_showdown=hand['players_at_showdown'], pot_bb=hand['final_pot'] / bb, net_bb_a=na / bb,
             net_bb_b=nb / bb, W_net_bb=se[w]['net_chips'] / bb, L_net_bb=se[l]['net_chips'] / bb, net_tie=na == nb, W_is_a=w == a, opp_sign=opp,
             transfer_bb=min(abs(na), abs(nb)) / bb if opp else 0., winner='a' if na > nb else ('b' if nb > na else 'none'),
             both_showdown=bool(se[a]['went_to_showdown'] and se[b]['went_to_showdown']), ended_street=-1)
    r['transfer_share'] = r['transfer_bb'] / r['pot_bb'] if r['pot_bb'] > 0 else None
    role = {w: 'W', l: 'L'}
    for R in 'WL':
        r.update({f'{R}_{k}': None for k in MEMBER})
        r.update({f'{R}_folded': False, f'{R}_n_calls_facing_partner': 0, f'{R}_n_raises_facing_partner': 0})
        r.update({f'{R}_{s}_{k}': 0 for s in NM for k in ('br', 'calls', 'checks', 'fold')})
    r.update({f'pot_bb_start_{s}': None for s in NM})
    ev, street, last, folded, hu = [], None, None, set(), {}
    called, p_aggr = defaultdict(set), defaultdict(int)
    for x in ac:
        s, p = SO[x['street']], x['player_id']
        if x['street'] != street:
            street, last = x['street'], None
            r[f'pot_bb_start_{NM[s]}'], r['ended_street'] = x['pot_before'] / bb, max(r['ended_street'], s)
            hu[s] = a not in folded and b not in folded
        if p not in (a, b):
            hu[s] = False
        else:
            R, o = role[p], (b if p == a else a)
            fac, amt, pot = last == o, (x['amount'] or 0) / bb, x['pot_before'] / bb
            k = 'fold' if x['action'] == 'fold' else ('br' if aggressive(x) else ('calls' if passive_call(x) else 'checks'))
            r[f'{R}_{NM[s]}_{k}'] += 1
            if k == 'fold':
                r.update({f'{R}_folded': True, f'{R}_fold_street': s, f'{R}_fold_facing': 'partner' if fac else ('outsider' if last else 'none'),
                          f'{R}_fold_pot_bb': pot, f'{R}_fold_to_call_bb': x['to_call'] / bb, f'{R}_partner_aggr_before_fold': p_aggr[o],
                          f'{R}_called_before_fold_same_street': s in called[p], f'{R}_called_before_fold_earlier': any(u < s for u in called[p])})
            elif k == 'calls':
                called[p].add(s)
                if fac:
                    r[f'{R}_n_calls_facing_partner'] += 1
                    if r[f'{R}_max_call_bb'] is None or amt > r[f'{R}_max_call_bb']:
                        r.update({f'{R}_max_call_bb': amt, f'{R}_max_call_street': s, f'{R}_max_call_pot_share': amt / (pot + amt) if pot + amt > 0 else None})
            elif k == 'br' and fac:
                r[f'{R}_n_raises_facing_partner'] += 1
            ev.append((R, s, k, fac, amt, pot, o not in folded))
            if k == 'fold':
                folded.add(p)
        if aggressive(x):
            last = p
            p_aggr[p] += 1
    r['hu_streets'] = int(sum(hu.values()))
    r['facing_partner_any'] = any(e[3] and e[2] != 'checks' for e in ev)
    return r, ev, w, l
