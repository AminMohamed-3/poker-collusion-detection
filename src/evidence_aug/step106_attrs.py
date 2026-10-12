"""Step 106 (Claude): label-free gameplay attributes of pair shared hands. Equity against the partner with per-member event
indicators (the step-105 descriptive build), strength against one random hand (canonical preflop classes), and outsider
involvement. Members come from player columns only; no labels and no scores are read here."""
import sys
sys.dont_write_bytecode = True
import hashlib
import random
from collections import defaultdict
import polars as pl
from phevaluator import evaluate_cards

DATA = 'C:/Data/Projects/Money Making/work/poker/data/'
SO = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
BL = [0, 3, 4, 5]
NM = ['preflop', 'flop', 'turn', 'river']
DECK = [r + s for r in '23456789TJQKA' for s in 'cdhs']
RANKS = '23456789TJQKA'
PRE = {}


def load(hids):
    hands = {r['hand_id']: r for r in pl.scan_parquet(DATA + 'hands.parquet').filter(pl.col('hand_id').is_in(hids)).drop('started_at').collect().iter_rows(named=True)}
    seats = defaultdict(dict)
    for r in pl.scan_parquet(DATA + 'seats.parquet').filter(pl.col('hand_id').is_in(hids)).collect().iter_rows(named=True):
        seats[r['hand_id']][r['player_id']] = r
    acts = defaultdict(list)
    for r in pl.scan_parquet(DATA + 'actions.parquet').filter(pl.col('hand_id').is_in(hids)).collect().sort('hand_id', 'action_no').iter_rows(named=True):
        acts[r['hand_id']].append(r)
    return hands, seats, acts


def aggressive(x):
    return x['action'] in ('bet', 'raise') or (x['action'] == 'all_in' and x['amount'] > x['to_call'])


def passive_call(x):
    return x['action'] == 'call' or (x['action'] == 'all_in' and x['amount'] <= x['to_call'])


def compare(a, b, board):
    u, v = evaluate_cards(*a, *board), evaluate_cards(*b, *board)
    return 1. if u < v else (.5 if u == v else 0.)


def seeded(key):
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def equity(a, b, board, key):
    rest = [c for c in DECK if c not in set(a) | set(b) | set(board)]
    need = 5 - len(board)
    if need == 0:
        return compare(a, b, board)
    if need == 1:
        return sum(compare(a, b, board + [c]) for c in rest) / len(rest)
    rng = seeded(key)
    return sum(compare(a, b, board + rng.sample(rest, need)) for _ in range(300)) / 300


def scan(hand, se, ac, a, b):
    """Per-member event indicators within one hand; equities are member a against member b at each street both started."""
    hid = hand['hand_id']
    cards = {m: [se[m]['hole_card_1'], se[m]['hole_card_2']] for m in (a, b)}
    board, bb = (hand['board_cards'] or '').split(), max(1, hand['big_blind'])
    per = {m: dict(fold_s=-1, fold_facing='nofold', fold_eq=None, fold_ahead_partner=0, fold_to_call_bb=0., fold_pot_bb=0., inv_bb=0.,
                   bet_fold_partner=0, weak_call_partner=0, passive_strong=0, aggr=0, calls=0, hu_aggr=0) for m in (a, b)}
    st = {'s': None, 'last': None, 'agg': defaultdict(int)}
    folded, eq = set(), {}

    def close():
        s = SO[st['s']]
        if s in eq:
            for m in (a, b):
                if (eq[s] if m == a else 1 - eq[s]) > .7 and st['agg'][m] == 0:
                    per[m]['passive_strong'] += 1
    for x in ac:
        if x['street'] != st['s']:
            if st['s'] is not None:
                close()
            st.update(s=x['street'], last=None, agg=defaultdict(int))
            s = SO[x['street']]
            if a not in folded and b not in folded and len(board) >= BL[s]:
                eq[s] = equity(cards[a], cards[b], board[:BL[s]], f'{hid}|{a}|{b}|{s}')
        p, s, last = x['player_id'], SO[st['s']], st['last']
        if p in (a, b):
            o = b if p == a else a
            em = None if (s not in eq or o in folded) else (eq[s] if p == a else 1 - eq[s])
            if x['action'] == 'fold':
                per[p].update(fold_s=s, fold_facing='partner' if last == o else ('outsider' if last else 'none'), fold_eq=em,
                              fold_to_call_bb=x['to_call'] / bb, fold_pot_bb=x['pot_before'] / bb,
                              fold_ahead_partner=int(last == o and em is not None and em > .5), bet_fold_partner=int(last == o and st['agg'][p] > 0))
                folded.add(p)
            else:
                per[p]['inv_bb'] += (x['amount'] or 0) / bb
                if passive_call(x):
                    per[p]['calls'] += 1
                    if last == o and em is not None and em < .35:
                        per[p]['weak_call_partner'] = 1
        if aggressive(x):
            st['last'] = p
            st['agg'][p] += 1
            if p in (a, b):
                per[p]['aggr'] += 1
                if x['players_active'] == 2 and a not in folded and b not in folded:
                    per[p]['hu_aggr'] += 1
    if st['s'] is not None:
        close()
    return per, eq


def equity_row(pid, hand, se, ac, a, b):
    per, eq = scan(hand, se, ac, a, b)
    w, l = (a, b) if se[a]['net_chips'] >= se[b]['net_chips'] else (b, a)
    both_sd = bool(se[a]['went_to_showdown'] and se[b]['went_to_showdown'])
    board, bb = (hand['board_cards'] or '').split(), max(1, hand['big_blind'])
    fin = compare([se[l]['hole_card_1'], se[l]['hole_card_2']], [se[w]['hole_card_1'], se[w]['hole_card_2']], board) if both_sd and len(board) == 5 else None
    r = dict(pair_id=pid, hand_id=hand['hand_id'], players_dealt=hand['players_dealt'], sd_players=hand['players_at_showdown'], pot_bb=hand['final_pot'] / bb,
             pair_net_bb=(se[a]['net_chips'] + se[b]['net_chips']) / bb, W_net_bb=se[w]['net_chips'] / bb, L_net_bb=se[l]['net_chips'] / bb,
             both_sd=int(both_sd), L_beats_W_sd=fin, out_won=int(any((v['won_share'] or 0) > 0 for k, v in se.items() if k not in (a, b))))
    for s, name in enumerate(('pre', 'flop', 'turn', 'river')):
        r['L_eq_' + name] = None if s not in eq else (eq[s] if l == a else 1 - eq[s])
    for role, m in (('W', w), ('L', l)):
        for k, v in per[m].items():
            r[f'{role}_{k}'] = v
    return r, w, l


def strength_mc(hole, board, key, n):
    rest = [c for c in DECK if c not in set(hole) | set(board)]
    need = 5 - len(board)
    rng, total = seeded(key), 0.
    for _ in range(n):
        draw = rng.sample(rest, 2 + need)
        full = board + draw[2:]
        u, v = evaluate_cards(*hole, *full), evaluate_cards(*draw[:2], *full)
        total += 1. if u < v else (.5 if u == v else 0.)
    return total / n


def preflop_strength(hole):
    hi, lo = sorted(hole, key=lambda c: RANKS.index(c[0]), reverse=True)
    cls = hi[0] + lo[0] + ('s' if hi[1] == lo[1] else 'o')
    if cls not in PRE:
        PRE[cls] = strength_mc([hi[0] + 'h', lo[0] + ('h' if cls[2] == 's' else 's')], [], 'pre|' + cls, 3000)
    return PRE[cls]


def strength_row(hand, se, ac, w, l):
    board = (hand['board_cards'] or '').split()
    acted = {(x['player_id'], x['street']) for x in ac}
    r = {}
    for role, m in (('W', w), ('L', l)):
        hole = [se[m]['hole_card_1'], se[m]['hole_card_2']]
        for i, s in enumerate(NM):
            ok = (m, s) in acted and len(board) >= BL[i]
            r[f'{role}_S_{s}'] = None if not ok else (preflop_strength(hole) if i == 0 else strength_mc(hole, board[:BL[i]], f"{hand['hand_id']}|{m}|{s}", 200))
    return r

OUTSIDER = ['fold_partner_hu', 'fold_partner_multi', 'fold_partner_after_outfold', 'pair_reraise', 'sandwich', 'caught', 'out_fold_to_pair', 'out_aggr', 'out_call_pair']


def outsider_row(hand, se, ac, a, b):
    pair, f, street = (a, b), defaultdict(int), None
    for x in ac:
        if x['street'] != street:
            street, last, lpa, mid, outfold = x['street'], None, None, None, False
        p, agg = x['player_id'], aggressive(x)
        if p in pair:
            if x['action'] == 'fold' and last in pair and last != p:
                f['fold_partner_hu' if x['players_active'] == 2 else 'fold_partner_multi'] += 1
                if outfold:
                    f['fold_partner_after_outfold'] += 1
            if agg:
                if last in pair and last != p:
                    f['pair_reraise'] += 1
                if lpa is not None and lpa != p and mid is not None:
                    f['sandwich'] += 1
                    if mid:
                        f['caught'] += 1
                lpa, mid = p, None
        else:
            if x['action'] == 'fold' and last in pair:
                f['out_fold_to_pair'] += 1
                outfold = True
            elif agg:
                f['out_aggr'] += 1
            elif x['action'] != 'fold' and x['action'] != 'check' and last in pair:
                f['out_call_pair'] += 1
            if lpa is not None:
                mid = bool(mid) or x['action'] not in ('fold', 'check')
        if agg:
            last = p
    bb = max(1, hand['big_blind'])
    contrib = {k: (v['total_contribution'] or 0) for k, v in se.items()}
    return {**{k: f[k] for k in OUTSIDER}, 'out_contrib_bb': sum(v for k, v in contrib.items() if k not in pair) / bb,
            'pair_contrib_bb': (contrib.get(a, 0) + contrib.get(b, 0)) / bb}


def build(keys, members, chunk=4000):
    """keys: frame with pair_id and hand_id; members: pair_id -> (player_1, player_2). One attribute row per key."""
    keys = keys.select('pair_id', 'hand_id').unique().sort('pair_id', 'hand_id')
    hids, out = keys['hand_id'].unique().sort().to_list(), []
    for i in range(0, len(hids), chunk):
        part = hids[i:i + chunk]
        hands, seats, acts = load(part)
        for pid, hid in keys.filter(pl.col('hand_id').is_in(part)).iter_rows():
            a, b = members[pid]
            se, ac = seats[hid], acts[hid]
            r, w, l = equity_row(pid, hands[hid], se, ac, a, b)
            r.update(strength_row(hands[hid], se, ac, w, l))
            r.update(outsider_row(hands[hid], se, ac, a, b))
            out.append(r)
    return pl.DataFrame(out, infer_schema_length=None).sort('pair_id', 'hand_id')
