"""poker-a8 typed-event attributes, equity stage (top-20 keys). Street-start equity of member a vs b and strength vs one random hand use the
step106 functions and seed keys, so overlaps with the step106 reranker attributes are exact; value-transfer magnitudes; R1 as step106_features."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Claude Version/src')
from evidence_aug import step106_attrs as A6

SO = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
NM = ['pre', 'flop', 'turn', 'river']
N6 = ['preflop', 'flop', 'turn', 'river']
BL = [0, 3, 4, 5]


def equity_fields(hand, se, ac, a, b, w, l, ev):
    hid, board = hand['hand_id'], (hand['board_cards'] or '').split()
    cards = {m: [se[m]['hole_card_1'], se[m]['hole_card_2']] for m in (a, b)}
    streets, fold = set(), {}
    for x in ac:
        streets.add(SO[x['street']])
        if x['action'] == 'fold' and x['player_id'] in (a, b):
            fold[x['player_id']] = SO[x['street']]
    inn = lambda m, s: s in streets and fold.get(m, 9) >= s and len(board) >= BL[s]
    eq, S, mem, r = {}, {}, {'W': w, 'L': l}, {}
    for s in range(4):
        if inn(a, s) and inn(b, s):
            eq[s] = A6.equity(cards[a], cards[b], board[:BL[s]], f'{hid}|{a}|{b}|{s}')
        for m in (a, b):
            if inn(m, s):
                S[m, s] = A6.preflop_strength(cards[m]) if s == 0 else A6.strength_mc(cards[m], board[:BL[s]], f'{hid}|{m}|{N6[s]}', 200)
        r[f'L_eq_{NM[s]}'] = None if s not in eq else (eq[s] if l == a else 1 - eq[s])
        for R, m in mem.items():
            r[f'{R}_S_{NM[s]}'] = S.get((m, s))
    em = lambda R, s: None if s not in eq else (eq[s] if mem[R] == a else 1 - eq[s])
    for R in 'WL':
        r.update({f'{R}_fold_eq': None, f'{R}_vt_call_sum': 0., f'{R}_vt_call_max': 0., f'{R}_vt_fold': 0., f'{R}_ps_n': 0, f'{R}_ps_pot_sum': 0.,
                  f'{R}_ps_pot_max': 0.})
    r1 = False
    for R, s, k, fac, amt, pot, pin in ev:
        e, sm = em(R, s), S.get((mem[R], s))
        if k == 'fold' and pin and e is not None:
            r[f'{R}_fold_eq'] = e
            r[f'{R}_vt_fold'] = pot * e if fac else 0.
            r1 = r1 or (fac and e > .5 and (sm or 0) > .6)
        elif k == 'calls' and fac and e is not None:
            r[f'{R}_vt_call_sum'] += amt * (1 - e)
            r[f'{R}_vt_call_max'] = max(r[f'{R}_vt_call_max'], amt * (1 - e))
        if k in ('calls', 'checks') and pin and (sm or 0) > .7:
            r[f'{R}_ps_n'] += 1
            r[f'{R}_ps_pot_sum'] += pot
            r[f'{R}_ps_pot_max'] = max(r[f'{R}_ps_pot_max'], pot)
    for R in 'WL':
        r[f'{R}_vt_total'] = r[f'{R}_vt_call_sum'] + r[f'{R}_vt_fold']
    r['R1'] = bool(r1)
    return r
