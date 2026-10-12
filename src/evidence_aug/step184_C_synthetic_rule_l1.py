"""Step 184 (Claude): mechanise RULE-L1 over the OpenCode synthetic tables, so the three complete-but-unannotated tables get evidence
annotations without hand judgement. RULE-L1 (tasks/synthetic/annotate_T002.md): listed = the first min(5, eligible) partner postflop
folds in started_at order, where eligible = partner folds postflop AND the other partner showed heat (bet or raise) on the same street
before the fold AND the fold is cheap (to_call / pot_before <= .5) AND (the folder's hole is live -- pocket pair, hole card pairing the
board, four-flush or open-ended straight draw -- OR the folder is ahead of the partner right now by score7).
Gate: re-deriving T002_SP and T004_DT must reproduce their hand-written `annotations.json` listings, or the mechanisation is wrong and
the output for the other tables is not trusted. Reads the synthetic records read-only; writes only annotations_rule_l1.json per table
and a summary. No uploads, no production effect. Usage: step184_C_synthetic_rule_l1.py"""
import os, sys, json, glob
sys.dont_write_bytecode = True
POKER = 'C:/Data/Projects/Money Making/work/poker'
sys.path.insert(0, POKER + '/synthetic')
from engine import score7                                        # noqa: E402

TABLES = POKER + '/data/synthetic/tables/'
OUT = POKER + '/Claude Version/experiments/synthetic_rule_l1_20260916/'
RANKS = '23456789TJQKA'
STREETS = ('preflop', 'flop', 'turn', 'river')
NBOARD = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}


def live(hole, board):
    """pocket pair, hole card pairing the board, four-flush or open-ended straight draw."""
    if hole[0][0] == hole[1][0]:
        return 'pocket_pair'
    if any(h[0] == b[0] for h in hole for b in board):
        return 'hole_pairs_board'
    for s in 'cdhs':
        if sum(c[1] == s for c in hole + board) == 4 and any(c[1] == s for c in hole):
            return 'four_flush'
    vals = sorted({RANKS.index(c[0]) for c in hole + board})
    hv = {RANKS.index(c[0]) for c in hole}
    for i in range(len(vals) - 3):
        w = vals[i:i + 4]
        if w[3] - w[0] == 3 and w[0] > 0 and w[3] < 12 and (set(w) & hv):
            return 'oesd'
    return None


def table_rows(t):
    cfg = json.load(open(t + '/config.json'))
    seats = json.load(open(t + '/seats.json'))
    hands = {h['hand_id']: h for h in json.load(open(t + '/hands.json'))}
    acts = json.load(open(t + '/actions.json'))
    import re
    pseats = [int(re.search(r'seat(\d+)', p).group(1)) for p in cfg['partners']]   # DT tables append -sender-A / -receiver-B
    seat_of = {}
    hole = {}
    for s in seats:
        seat_of.setdefault(s['hand_id'], {})[s['player_id']] = s['seat_no']
        hole[(s['hand_id'], s['player_id'])] = [s['hole_card_1'], s['hole_card_2']]
    pid_by_seat = {}
    for s in seats:
        pid_by_seat.setdefault(s['hand_id'], {})[s['seat_no']] = s['player_id']
    by_hand = {}
    for a in acts:
        by_hand.setdefault(a['hand_id'], []).append(a)
    rows = []
    for hid, aa in by_hand.items():
        h = hands[hid]
        board = (h.get('board_cards') or '').split()
        pm = pid_by_seat.get(hid, {})
        pu = [pm.get(s) for s in pseats]
        if not all(pu):
            continue
        aa = sorted(aa, key=lambda x: x['action_no'])
        for i, a in enumerate(aa):
            if a['action'] != 'fold' or a['street'] == 'preflop' or a['player_id'] not in pu:
                continue
            partner = pu[0] if a['player_id'] == pu[1] else pu[1]
            heat = [b for b in aa[:i] if b['street'] == a['street'] and b['player_id'] == partner
                    and b['action'] in ('bet', 'raise', 'all_in')]
            if not heat:
                continue
            pot = a['pot_before'] or 1
            if a['to_call'] / pot > .5:
                continue
            b = board[:NBOARD[a['street']]]
            mine, theirs = hole[(hid, a['player_id'])], hole[(hid, partner)]
            lv = live(mine, b) if len(b) >= 3 else None
            ahead = None
            if len(b) >= 3:
                ahead = score7(mine, b) > score7(theirs, b)      # engine.score7(hole, board), best-5 comparable
            if not (lv or ahead):
                continue
            rows.append({'hand_id': hid, 'hand_idx': h.get('hand_idx'), 'started_at': h['started_at'],
                         'street': a['street'], 'to_call': a['to_call'], 'pot_before': a['pot_before'],
                         'folder_seat': seat_of[hid][a['player_id']], 'live': lv, 'ahead': bool(ahead),
                         'hole': ' '.join(mine), 'partner_hole': ' '.join(theirs), 'board': ' '.join(b)})
    rows.sort(key=lambda r: (r['started_at'], r['hand_id']))
    return cfg, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for t in sorted(glob.glob(TABLES + 'T0*')):
        name = os.path.basename(t)
        if not os.path.exists(t + '/HANDS_COMPLETE'):
            summary[name] = {'skipped': 'incomplete'}
            continue
        cfg, rows = table_rows(t)
        listed = rows[:5]
        rec = {'table': name, 'table_id': cfg['table_id'], 'partners': cfg['partners'], 'rule': 'RULE-L1, mechanised (step184)',
               'eligible_n': len(rows), 'listed': [dict(r, rank=i + 1) for i, r in enumerate(listed)],
               'overflow': rows[5:], 'method': 'first min(5, eligible) in started_at order'}
        ref = t + '/annotations.json'
        if os.path.exists(ref):
            a = json.load(open(ref))
            want = [e['hand_id'] for e in sorted(a.get('eligible_rank1_first5', []), key=lambda e: e.get('rank', 99))]
            got = [r['hand_id'] for r in listed]
            rec['reference_listed'] = want
            rec['reproduces_reference'] = bool(want) and set(want) == set(got)
            rec['overlap_with_reference'] = len(set(want) & set(got))
        json.dump(rec, open(t + '/annotations_rule_l1.json', 'w', encoding='utf-8'), indent=1)
        summary[name] = {k: rec[k] for k in ('eligible_n', 'reproduces_reference', 'overlap_with_reference') if k in rec}
        summary[name]['listed'] = [r['hand_id'] for r in listed]
        print(name, json.dumps(summary[name]), flush=True)
    json.dump(summary, open(OUT + 'summary184.json', 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
