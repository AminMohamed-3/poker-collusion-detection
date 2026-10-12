"""STEP280 shared library: honest-play policy simulator and its two-sample harness.

Everything here is mechanical. No LLM output, no organiser simulator source, no
evaluation-phase record, no players.parquet. The betting rules were inferred from the
public columns of hands/actions/seats.parquet (see registration280.json, P4_engine)
and validated by replaying real hands through this engine and comparing the derived
pot_before / stack_before / to_call / players_active against the recorded values.

The same state machine (`run_hand`) both replays recorded hands and generates new
ones, so the decision features a policy is FITTED on are computed by exactly the same
code path as the features it is SAMPLED with.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from phevaluator import Card, evaluate_cards

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
DATA = ROOT / 'data'
HERE = Path(__file__).resolve().parent

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
CARD_ID = {r + s: Card.to_id(r + s) for r in RANKS for s in SUITS}
ID_CARD = {v: k for k, v in CARD_ID.items()}
RANK_OF = {v: RANKS.index(k[0]) for k, v in CARD_ID.items()}      # 0 = deuce .. 12 = ace
SUIT_OF = {v: SUITS.index(k[1]) for k, v in CARD_ID.items()}
RANK_ARR = np.array([RANK_OF[i] for i in range(52)], dtype=np.int64)
SUIT_ARR = np.array([SUIT_OF[i] for i in range(52)], dtype=np.int64)

STREETS = ['preflop', 'flop', 'turn', 'river']
STREET_IDX = {s: i for i, s in enumerate(STREETS)}
BOARD_LEN = [0, 3, 4, 5]
ACTION_CLASSES = ['fold', 'check', 'call', 'raise', 'all_in']
CLASS_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}

# amount / pot_before buckets for the size head
SIZE_EDGES = [0.0, .35, .55, .80, 1.05, 1.50, 2.20, 3.50, 1e9]
N_SIZE = len(SIZE_EDGES) - 1

# phevaluator rank boundaries (1 = best). Used only to name a made-hand category.
CAT_EDGES = [10, 166, 322, 1599, 1609, 2467, 3325, 6185, 7462]
# 0 straight flush, 1 quads, 2 boat, 3 flush, 4 straight, 5 trips, 6 two pair, 7 pair, 8 high


# ----------------------------------------------------------------------------- splits
def table_key(table_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(table_id.encode('utf-8'), digest_size=8).digest(), 'big')


def split_tables(tables):
    """buckets 0-5 POLICY_TRAIN, 6-7 POLICY_HELD, 8-9 TWO_SAMPLE (registration P1)."""
    out = {'POLICY_TRAIN': [], 'POLICY_HELD': [], 'TWO_SAMPLE': []}
    for t in sorted(tables):
        b = table_key(t) % 10
        out['POLICY_TRAIN' if b <= 5 else 'POLICY_HELD' if b <= 7 else 'TWO_SAMPLE'].append(t)
    return out


def policy_bag(table_id: str) -> int:
    return (table_key(table_id) // 10) % 5


def disc_side(table_id: str) -> str:
    return 'train' if (table_key(table_id) // 100) % 4 < 3 else 'test'


# ----------------------------------------------------------------------------- loading
def load_dev_hands() -> pl.DataFrame:
    h = pl.read_parquet(DATA / 'hands.parquet',
                        columns=['hand_id', 'table_id', 'phase', 'button_seat', 'small_blind',
                                 'big_blind', 'board_cards', 'final_pot', 'players_dealt',
                                 'players_at_showdown'])
    h = h.filter(pl.col('phase') == 'development')
    phases = h['phase'].unique().to_list()
    assert phases == ['development'], f'PHASE ASSERTION FAILED: {phases}'
    return h.drop('phase')


def positive_players() -> set:
    lab = pl.read_csv(DATA / 'development_labels.csv')
    pos = lab.filter(pl.col('label') == 1)
    return set(pos['player_1'].to_list()) | set(pos['player_2'].to_list())


def fetch_rows(hand_ids: pl.DataFrame, which: str) -> pl.DataFrame:
    """One streaming pass over seats/actions restricted to the selected hands."""
    lazy = pl.scan_parquet(DATA / f'{which}.parquet')
    return lazy.join(hand_ids.lazy(), on='hand_id', how='inner').collect(engine='streaming')


# ----------------------------------------------------------------------------- cards
def hand_category(rank: int) -> int:
    for i, e in enumerate(CAT_EDGES):
        if rank <= e:
            return i
    return 8


def _straight_draw(rank_bits: int, nboard: int) -> int:
    """0 none, 1 gutshot, 2 open-ended (>=2 filling ranks). Ace counts low as well."""
    if nboard >= 5:
        return 0
    def has_straight(bits):
        b = bits | ((bits >> 12) & 1)  # ace low at position -1 -> fold into bit 0 trick below
        for lo in range(0, 9):
            if all((bits >> (lo + k)) & 1 for k in range(5)):
                return True
        # wheel A-2-3-4-5
        if (bits >> 12) & 1 and all((bits >> k) & 1 for k in range(4)):
            return True
        return False
    if has_straight(rank_bits):
        return 0
    fills = 0
    for r in range(13):
        if (rank_bits >> r) & 1:
            continue
        if has_straight(rank_bits | (1 << r)):
            fills += 1
    return 0 if fills == 0 else (1 if fills == 1 else 2)


def card_shape(cards):
    """(rank_bits, suit_counts) for a list of card ids."""
    bits = 0
    sc = [0, 0, 0, 0]
    for c in cards:
        bits |= 1 << RANK_OF[c]
        sc[SUIT_OF[c]] += 1
    return bits, sc


def board_texture(board):
    n = len(board)
    if n == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    rc = {}
    sc = [0, 0, 0, 0]
    for c in board:
        rc[RANK_OF[c]] = rc.get(RANK_OF[c], 0) + 1
        sc[SUIT_OF[c]] += 1
    paired = 1.0 if max(rc.values()) >= 2 else 0.0
    top = max(rc) / 12.0
    rs = sorted(rc)
    conn = max(sum(1 for x in rs if lo <= x < lo + 5) for lo in range(13))
    return (n, paired, float(max(sc)), top, float(conn))


# preflop 169-bucket equity vs 5 random opponents, Monte Carlo, fixed seed, computed once
PF_CACHE = HERE / 'preflop_equity169.json'


def preflop_bucket(c1: int, c2: int):
    r1, r2 = RANK_OF[c1], RANK_OF[c2]
    hi, lo = max(r1, r2), min(r1, r2)
    suited = SUIT_OF[c1] == SUIT_OF[c2]
    kind = 'p' if hi == lo else ('s' if suited else 'o')
    return f'{hi}_{lo}_{kind}'


def build_preflop_equity(sims_per_bucket=1500, seed=280001):
    if PF_CACHE.exists():
        return json.loads(PF_CACHE.read_text(encoding='utf-8'))
    rng = np.random.default_rng(seed)
    table = {}
    for hi in range(13):
        for lo in range(hi + 1):
            for kind in (['p'] if hi == lo else ['s', 'o']):
                if hi == lo and kind != 'p':
                    continue
                if kind == 'p':
                    c1 = CARD_ID[RANKS[hi] + 'c']; c2 = CARD_ID[RANKS[lo] + 'd']
                elif kind == 's':
                    c1 = CARD_ID[RANKS[hi] + 'c']; c2 = CARD_ID[RANKS[lo] + 'c']
                else:
                    c1 = CARD_ID[RANKS[hi] + 'c']; c2 = CARD_ID[RANKS[lo] + 'd']
                rest = np.array([c for c in range(52) if c not in (c1, c2)])
                wins = 0.0
                for _ in range(sims_per_bucket):
                    draw = [int(x) for x in rng.choice(rest, size=15, replace=False)]
                    board = draw[:5]
                    mine = evaluate_cards(c1, c2, *board)
                    best = min(evaluate_cards(draw[5 + 2 * k], draw[6 + 2 * k], *board)
                               for k in range(5))
                    wins += 1.0 if mine < best else (0.5 if mine == best else 0.0)
                table[f'{hi}_{lo}_{kind}'] = wins / sims_per_bucket
    PF_CACHE.write_text(json.dumps(table, sort_keys=True), encoding='utf-8')
    return table


# ----------------------------------------------------------------------------- features
FEAT_NAMES = [
    'street', 'position', 'players_live', 'players_not_allin',
    'to_call_bb', 'pot_bb', 'pot_odds', 'stack_bb', 'spr',
    'committed_bb', 'street_committed_bb', 'facing_bet', 'n_raises_street',
    'actor_idx_street', 'yet_to_act', 'is_street_aggressor', 'is_hand_aggressor',
    'aggressor_rel_pos', 'can_raise', 'bb_size',
    'pf_equity', 'hole_hi', 'hole_lo', 'hole_pair', 'hole_suited', 'hole_gap',
    'hs', 'made_cat', 'flush_draw', 'straight_draw', 'n_hole_board_suit',
    'board_n', 'board_paired', 'board_maxsuit', 'board_top', 'board_conn',
]
N_FEAT = len(FEAT_NAMES)


def state_features(st, hole, board, hs, pf_eq):
    """One decision -> feature vector. Used identically by replay and by rollout."""
    bb = st['bb']
    to_call = st['to_call']
    pot = st['pot']
    stack = st['stack']
    c1, c2 = hole
    r1, r2 = RANK_OF[c1], RANK_OF[c2]
    hi, lo = max(r1, r2), min(r1, r2)
    suited = 1.0 if SUIT_OF[c1] == SUIT_OF[c2] else 0.0
    allc = [c1, c2] + list(board)
    bits, sc = card_shape(allc)
    nb = len(board)
    if nb == 0:
        made, fd, sd, nhb = -1.0, 0.0, 0.0, 0.0
    else:
        made = float(hand_category(evaluate_cards(*allc)))
        fd = 1.0 if (max(sc) == 4 and nb < 5) else 0.0
        sd = float(_straight_draw(bits, nb))
        nhb = float(max(sc))
    bt = board_texture(board)
    return [
        float(st['street']), float(st['position']), float(st['players_live']),
        float(st['players_not_allin']),
        to_call / bb, pot / bb, to_call / max(pot + to_call, 1.0), stack / bb,
        stack / max(pot, 1.0),
        st['committed'] / bb, st['street_committed'] / bb,
        1.0 if to_call > 0 else 0.0, float(st['n_raises_street']),
        float(st['actor_idx_street']), float(st['yet_to_act']),
        float(st['is_street_aggressor']), float(st['is_hand_aggressor']),
        float(st['aggressor_rel_pos']), float(st['can_raise']), float(bb),
        float(pf_eq), float(hi), float(lo), 1.0 if hi == lo else 0.0, suited,
        float(hi - lo),
        float(hs), made, fd, sd, nhb,
        float(bt[0]), bt[1], bt[2], bt[3], bt[4],
    ]


# ----------------------------------------------------------------------------- engine
class IllegalAction(Exception):
    pass


def side_pots(contrib, folded):
    """Standard layered side pots. Returns [(amount, [eligible seats])]."""
    levels = sorted(set(c for c in contrib if c > 0))
    pots, prev = [], 0
    for lv in levels:
        amt = sum(min(c, lv) - min(c, prev) for c in contrib)
        elig = [s for s in range(len(contrib)) if contrib[s] >= lv and not folded[s]]
        if amt > 0:
            pots.append((amt, elig))
        prev = lv
    return pots


def run_hand(cfg, decide, rng=None, cards=None, record=False):
    """Callback driver around `hand_gen` (used by the replay/validation path)."""
    g = hand_gen(cfg, rng=rng, cards=cards, record=record)
    try:
        req = next(g)
        while True:
            req = g.send(decide(*req))
    except StopIteration as stop:
        return stop.value


def hand_gen(cfg, rng=None, cards=None, record=False):
    """One hand of 6-max NLHE, as a coroutine.

    Yields (state, legal, ctx) at every decision and is sent (action_name, amount_to).
    Returns (via StopIteration.value) the hands/seats/actions rows.
    cfg: button_seat, small_blind, big_blind, starting_stack (list of 6)
    cards: optional {'hole': [(c1,c2)]*6, 'board': [5 ids]}; otherwise dealt from rng.
    """
    n = 6
    btn = cfg['button_seat']
    sb, bb = cfg['small_blind'], cfg['big_blind']
    stacks = [int(x) for x in cfg['starting_stack']]
    start_stacks = list(stacks)
    if cards is None:
        deck = rng.permutation(52)
        hole = [(int(deck[2 * i]), int(deck[2 * i + 1])) for i in range(n)]
        board_full = [int(x) for x in deck[12:17]]
    else:
        hole = cards['hole']
        board_full = cards['board']

    contrib = [0] * n            # total in the hand
    street_contrib = [0] * n
    folded = [False] * n
    allin = [False] * n
    actions = []
    decisions = []

    def post(seat, amt):
        amt = min(amt, stacks[seat])
        stacks[seat] -= amt
        contrib[seat] += amt
        street_contrib[seat] += amt
        if stacks[seat] == 0:
            allin[seat] = True

    post((btn + 1) % n, sb)
    post((btn + 2) % n, bb)

    street = 0
    hand_aggressor = (btn + 2) % n
    board_dealt = 0
    action_no = 0
    pf_cache = {}

    while True:
        live = [s for s in range(n) if not folded[s]]
        if len(live) < 2:
            break
        if street > 0:
            street_contrib = [0] * n
        board = board_full[:BOARD_LEN[street]]
        board_dealt = max(board_dealt, BOARD_LEN[street])
        level = max(street_contrib)
        last_raise_size = bb
        street_aggressor = hand_aggressor if street == 0 else -1
        can_raise = [True] * n
        actor_idx = 0
        contenders = [s for s in live if not allin[s]]
        if len(contenders) >= 2 or (len(contenders) == 1 and street_contrib[contenders[0]] < level):
            first = (btn + 3) % n if street == 0 else (btn + 1) % n
            needs = set(s for s in live if not allin[s])
            order = [(first + k) % n for k in range(n)]
            pos_ptr = 0
            guard = 0
            while needs:
                guard += 1
                if guard > 200:
                    raise IllegalAction('betting round did not terminate')
                seat = None
                for k in range(n):
                    s = order[(pos_ptr + k) % n]
                    if s in needs:
                        seat = s
                        pos_ptr = (pos_ptr + k + 1) % n
                        break
                if seat is None:
                    break
                if len([s for s in range(n) if not folded[s]]) < 2:
                    break
                stack = stacks[seat]
                to_call_raw = level - street_contrib[seat]
                # the source data reports to_call CAPPED at the actor's remaining stack
                to_call = min(to_call_raw, stack)
                legal = []
                if to_call_raw <= 0:
                    legal.append('check')
                    if stack > 0 and can_raise[seat]:
                        if stack > max(last_raise_size, bb):
                            legal.append('raise')
                        legal.append('all_in')
                else:
                    legal.append('fold')
                    if stack > to_call_raw:
                        legal.append('call')
                        if can_raise[seat] and stack > to_call_raw + last_raise_size:
                            legal.append('raise')
                        legal.append('all_in')
                    else:
                        legal.append('all_in')
                pot_before = sum(contrib)
                st = {
                    'street': street, 'seat': seat, 'position': (seat - btn) % n,
                    'players_live': len([s for s in range(n) if not folded[s]]),
                    'players_not_allin': len([s for s in range(n) if not folded[s] and not allin[s]]),
                    'to_call': to_call, 'pot': pot_before, 'stack': stack,
                    'committed': contrib[seat], 'street_committed': street_contrib[seat],
                    'n_raises_street': sum(1 for a in actions
                                           if a['street'] == street and a['aggressive']),
                    'actor_idx_street': actor_idx, 'yet_to_act': len(needs) - 1,
                    'is_street_aggressor': 1 if seat == street_aggressor else 0,
                    'is_hand_aggressor': 1 if seat == hand_aggressor else 0,
                    'aggressor_rel_pos': ((seat - street_aggressor) % n) if street_aggressor >= 0 else -1,
                    'can_raise': 1 if can_raise[seat] else 0,
                    'bb': bb, 'level': level, 'last_raise_size': last_raise_size,
                    'min_raise_to': level + last_raise_size,
                    'max_to': street_contrib[seat] + stack,
                }
                ctx = {'hole': hole[seat], 'board': board, 'all_hole': hole,
                       'cache': pf_cache}
                act, amt_to = yield (st, legal, ctx)
                if act not in legal:
                    raise IllegalAction(f'{act} not in {legal}')
                aggressive = False
                if act == 'fold':
                    folded[seat] = True
                    amount = 0
                    amount_to = street_contrib[seat]
                elif act == 'check':
                    amount = 0
                    amount_to = street_contrib[seat]
                elif act == 'call':
                    amount = to_call_raw
                    stacks[seat] -= amount
                    contrib[seat] += amount
                    street_contrib[seat] += amount
                    amount_to = street_contrib[seat]
                elif act == 'all_in':
                    amount = stack
                    stacks[seat] = 0
                    contrib[seat] += amount
                    street_contrib[seat] += amount
                    allin[seat] = True
                    amount_to = street_contrib[seat]
                    aggressive = amount_to > level
                else:  # raise / bet
                    target = int(amt_to)
                    lo_t = level + last_raise_size
                    hi_t = street_contrib[seat] + stack
                    target = max(lo_t, min(target, hi_t))
                    amount = target - street_contrib[seat]
                    if amount >= stack:
                        amount = stack
                        allin[seat] = True
                        act = 'all_in'
                    stacks[seat] -= amount
                    contrib[seat] += amount
                    street_contrib[seat] += amount
                    amount_to = street_contrib[seat]
                    aggressive = amount_to > level
                actions.append({
                    'action_no': action_no, 'street': street, 'seat': seat,
                    'action': act, 'amount': amount, 'amount_to': amount_to,
                    'pot_before': pot_before, 'stack_before': stack, 'to_call': to_call,
                    'players_active': st['players_live'], 'aggressive': aggressive,
                })
                if record:
                    decisions.append((st, ctx, act, amount, amount_to, pot_before))
                action_no += 1
                actor_idx += 1
                needs.discard(seat)
                if aggressive:
                    inc = amount_to - level
                    full = inc >= last_raise_size
                    level = amount_to
                    if full:
                        last_raise_size = inc
                    street_aggressor = seat
                    hand_aggressor = seat
                    for t in range(n):
                        if t == seat or folded[t] or allin[t]:
                            continue
                        needs.add(t)
                        if not full and street_contrib[t] >= level - inc:
                            can_raise[t] = False
                if len([s for s in range(n) if not folded[s]]) < 2:
                    break
        live = [s for s in range(n) if not folded[s]]
        if len(live) < 2:
            break
        if street == 3:
            break
        contenders = [s for s in live if not allin[s]]
        street += 1
        if len(contenders) < 2:
            # no more betting: run the board out to the river and go to showdown
            street = 3
            board_dealt = 5
            break

    live = [s for s in range(n) if not folded[s]]
    board_dealt = max(board_dealt, BOARD_LEN[street] if len(live) >= 2 else board_dealt)
    if len(live) >= 2:
        board_dealt = 5 if street == 3 else BOARD_LEN[street]
    board_out = board_full[:board_dealt]
    final_pot = sum(contrib)
    won = [0] * n
    showdown = 0
    if len(live) == 1:
        won[live[0]] = final_pot
    else:
        showdown = len(live)
        ranks = {s: evaluate_cards(hole[s][0], hole[s][1], *board_out) for s in live}
        for amt, elig in side_pots(contrib, folded):
            if not elig:
                continue
            best = min(ranks[s] for s in elig)
            winners = sorted((s for s in elig if ranks[s] == best),
                             key=lambda s: (s - btn - 1) % n)   # odd chips to the first seat left of the button
            share = amt // len(winners)
            rem = amt - share * len(winners)
            for i, w in enumerate(winners):
                won[w] += share + (1 if i < rem else 0)
    return {
        'button_seat': btn, 'small_blind': sb, 'big_blind': bb,
        'board_cards': ' '.join(ID_CARD[c] for c in board_out),
        'final_pot': final_pot, 'players_dealt': n, 'players_at_showdown': showdown,
        'hole': hole, 'contrib': contrib, 'won': won, 'folded': folded,
        'start_stacks': start_stacks, 'actions': actions, 'decisions': decisions,
        'showdown_seats': live if len(live) >= 2 else [],
    }


# ----------------------------------------------------------------------------- strengths
def street_hs(board, hole_by_seat, seats, rng, K=128):
    """HS = share of K sampled opponent holdings beaten on this board. Same K on both sides."""
    if not board:
        return {s: 0.0 for s in seats}
    dead = set(board)
    rest = np.array([c for c in range(52) if c not in dead])
    pick = rng.choice(rest, size=(K, 2), replace=True)
    opp = []
    for i in range(K):
        a, b = int(pick[i, 0]), int(pick[i, 1])
        if a == b:
            continue
        opp.append(evaluate_cards(a, b, *board))
    opp = np.array(opp)
    out = {}
    for s in seats:
        c1, c2 = hole_by_seat[s]
        if c1 in dead or c2 in dead:
            out[s] = 0.0
            continue
        r = evaluate_cards(c1, c2, *board)
        out[s] = float((opp > r).mean() + 0.5 * (opp == r).mean())
    return out
