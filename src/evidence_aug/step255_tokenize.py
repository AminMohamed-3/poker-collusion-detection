"""STEP255 step 1 -- turn every (pair, hand) into a raw-ACTION-SEQUENCE token stream from the
pair's point of view, and from that build (a) a sparse n-gram design matrix and (b) an integer
token-id sequence for a neural encoder.

Development phase only (asserted). No player ids anywhere in the token strings: the two members are
A (the member who loses more chips over the pair's shared hands; ties broken by player_id order)
and B, every other seat is O. Outsiders are NOT individually numbered in the action tokens, so no
identity can leak across hands.

Views (each n-gram key carries its view, so the same surface form in two views is two features):
  P  ROLE.ACT.FACING.PSTATE                       n = 1..4   (the primary ordering view)
  R  ROLE.ACT                                     n = 2..4   (coarse, for generalisation)
  Q  ROLE.STREET.ACT.SIZE.STR                     n = 1..2   (rich, local)
  F  ROLE.STREET.ACT.SIZE.FACING.PSTATE.STR.BL    n = 1      (full composite)
  S  SKIP.street/adj . P_i . P_j                  member x other-member ordered pairs, same street
  H  header / card / board / end tokens           n = 1

Key packing (exact and invertible, no hashing): key = (view*8 + n) << 48 | symbols, where a
unigram uses the low 48 bits for one symbol id and an n-gram (n>=2) packs up to four 12-bit ids.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '6')

import json
import time

import numpy as np
import polars as pl
from phevaluator import Card, evaluate_cards

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
OOF = ROOT + r'\Claude Version\experiments\step232_event_detector_20260917\hand_oof.parquet'
OUT = ROOT + r'\Claude Version\experiments\step255_sequence_view_20260917'
TMP = (r'C:\Users\USER\AppData\Local\Temp\claude'
       r'\C--Data-Projects-Money-Making-work-poker'
       r'\f2bb4f17-323f-4806-98b3-984023c70212\scratchpad\step255')

CARDS = {r + s: Card.to_id(r + s) for r in '23456789TJQKA' for s in 'cdhs'}
RANKS = {r: i + 2 for i, r in enumerate('23456789TJQKA')}
SIDX = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
SNAME = ('pf', 'fl', 'tu', 'ri')
NB = (0, 3, 4, 5)
AGG = ('bet', 'raise', 'all_in')
NCHUNK = 24
MAXLEN = 96          # neural sequence length
MIN_DF = 25          # document-frequency floor for the n-gram vocabulary
VIEWS = ('P', 'R', 'Q', 'F', 'S', 'H')


# ---------------------------------------------------------------- strength helpers (step237 convention)
def preflop_strength(c1: int, c2: int) -> float:
    r1, r2 = c1 // 4 + 2, c2 // 4 + 2
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return .55 + hi / 32
    return min(.95, max(.05, (hi + lo) / 32 - max(hi - lo - 1, 0) / 40 + .07 * (c1 % 4 == c2 % 4)))


def strengths_for(c1: str, c2: str, board: list[str]) -> list[float | None]:
    a, b = CARDS[c1], CARDS[c2]
    ids = [CARDS[c] for c in board]
    out: list[float | None] = [preflop_strength(a, b)]
    for n in NB[1:]:
        out.append((7463 - evaluate_cards(a, b, *ids[:n])) / 7462 if len(ids) >= n else None)
    return out


def hole_class(c1: str, c2: str) -> str:
    r1, r2 = RANKS[c1[0]], RANKS[c2[0]]
    hi, lo = max(r1, r2), min(r1, r2)
    suited = c1[1] == c2[1]
    if hi == lo:
        return 'pp_hi' if hi >= 11 else ('pp_mid' if hi >= 7 else 'pp_lo')
    tag = 's' if suited else 'o'
    if hi == 14:
        return f'{tag}_A{"broad" if lo >= 10 else "x"}'
    if lo >= 10:
        return f'{tag}_broad'
    if hi - lo <= 2:
        return f'{tag}_conn{"hi" if hi >= 10 else "lo"}'
    return f'{tag}_junk{"hi" if hi >= 10 else "lo"}'


def board_tokens(board: list[str]) -> list[str]:
    out = []
    if len(board) >= 3:
        rs = sorted((RANKS[c[0]] for c in board[:3]), reverse=True)
        ss = [c[1] for c in board[:3]]
        pairing = 'trip' if rs[0] == rs[2] else ('pair' if rs[0] == rs[1] or rs[1] == rs[2] else 'unp')
        suit = 'mono' if len(set(ss)) == 1 else ('two' if len(set(ss)) == 2 else 'rain')
        high = 'A' if rs[0] == 14 else ('K' if rs[0] >= 12 else ('M' if rs[0] >= 9 else 'L'))
        conn = 'conn' if (rs[0] - rs[2]) <= 4 else 'dry'
        out.append(f'B.fl.{pairing}.{suit}.{high}.{conn}')
    for i, st in ((3, 'tu'), (4, 'ri')):
        if len(board) > i:
            r = RANKS[board[i][0]]
            prev = [RANKS[c[0]] for c in board[:i]]
            sprev = [c[1] for c in board[:i]]
            kind = 'pairs' if r in prev else ('over' if r > max(prev) else 'under')
            fl = 'flush' if sprev.count(board[i][1]) >= 2 and len(set(sprev + [board[i][1]])) <= 2 else 'nofl'
            out.append(f'B.{st}.{kind}.{fl}')
    return out


def bucket(x: float, cuts: tuple, names: tuple) -> str:
    for c, nm in zip(cuts, names):
        if x < c:
            return nm
    return names[-1]


# ---------------------------------------------------------------- symbol tables / key packing
class Sym:
    def __init__(self):
        self.tab = [dict() for _ in VIEWS]
        self.inv = [[] for _ in VIEWS]

    def id(self, view: int, s: str) -> int:
        t = self.tab[view]
        v = t.get(s)
        if v is None:
            v = len(t)
            t[s] = v
            self.inv[view].append(s)
        return v

    def dump(self) -> dict:
        return {VIEWS[i]: self.inv[i] for i in range(len(VIEWS))}


def pack(view: int, n: int, ids: tuple) -> int:
    head = (view * 8 + n) << 48
    if n == 1:
        return head | ids[0]
    k = 0
    for s in ids:
        k = (k << 12) | s
    return head | k


def unpack(key: int) -> tuple[int, int, tuple]:
    head = key >> 48
    view, n = head // 8, head % 8
    body = key & ((1 << 48) - 1)
    if n == 1:
        return view, n, (body,)
    ids = []
    for _ in range(n):
        ids.append(body & 4095)
        body >>= 12
    return view, n, tuple(reversed(ids))


def decode(key: int, inv: dict) -> str:
    view, n, ids = unpack(key)
    vn = VIEWS[view]
    if vn == 'S':
        return 'S|' + inv['S'][ids[0]] + ' | ' + ' >> '.join(inv['P'][i] for i in ids[1:])
    return f'{vn}{n}|' + ' > '.join(inv[vn][i] for i in ids)


# ---------------------------------------------------------------- one (pair, hand) -> tokens
SIZE_AGG = ((1 / 3, '<t'), (2 / 3, 't-2t'), (1.0, '2t-p'), (1e9, '>p'))
ODDS_C = ((.2, 'cheap'), (.35, 'mid'), (1e9, 'dear'))


def tok_hand(m1: str, m2: str, a_is_1: bool, meta: dict, seat: dict, acts: list,
             st: dict, cuts: list, sym: Sym):
    """Return (ngram keys as a list[int], neural token id sequence as list[str])."""
    A = m1 if a_is_1 else m2
    B = m2 if a_is_1 else m1
    role = {p: ('A' if p == A else ('B' if p == B else 'O')) for p in seat}
    bb = meta['bb']
    keys: list[int] = []
    seq: list[str] = []

    def hkey(s: str):
        keys.append(pack(5, 1, (sym.id(5, s),)))
        seq.append(s)

    # ---- header
    pos = meta['pos']            # player -> position label
    order = meta['order']        # preflop action order (players)
    hkey(f'H.posA.{pos[A]}')
    hkey(f'H.posB.{pos[B]}')
    ia, ib = order.index(A), order.index(B)
    hkey('H.order.' + ('A_first' if ia < ib else 'B_first') + f'.gap{abs(ia - ib)}')
    stkA, stkB = seat[A]['starting_stack'] / bb, seat[B]['starting_stack'] / bb
    eff = min(stkA, stkB)
    hkey('H.eff.' + bucket(eff, (20, 50, 100, 200, 1e9), ('lt20', '20_50', '50_100', '100_200', 'gt200')))
    hkey('H.stkA.' + bucket(stkA, (50, 100, 1e9), ('sh', 'md', 'dp')))
    hkey('H.stkB.' + bucket(stkB, (50, 100, 1e9), ('sh', 'md', 'dp')))
    for nm, p in (('A', A), ('B', B)):
        hkey(f'H.blind{nm}.' + ('sb' if pos[p] == 'SB' else ('bb' if pos[p] == 'BB' else 'no')))
    # ---- cards
    hkey('C.A.' + hole_class(seat[A]['hole_card_1'], seat[A]['hole_card_2']))
    hkey('C.B.' + hole_class(seat[B]['hole_card_1'], seat[B]['hole_card_2']))
    others = [p for p in seat if p not in (A, B)]
    for s in range(4):
        for nm, p in (('A', A), ('B', B)):
            v = st[p][s]
            if v is not None:
                hkey(f'C.str{nm}.{SNAME[s]}.' + terc(v, cuts[s]))
        ov = [st[p][s] for p in others if st[p][s] is not None]
        if ov:
            hkey(f'C.strO.{SNAME[s]}.' + terc(max(ov), cuts[s]))
    for t in board_tokens(meta['board']):
        hkey(t)

    # ---- action sequence
    live = set(seat)
    street = None
    prev_agg = None
    acted = set()
    later: dict = {}
    for i, a in enumerate(acts):
        later.setdefault((a['street'], a['player_id']), []).append(i)
    Pseq: list[int] = []     # P symbol ids in order
    Rseq: list[int] = []
    Qseq: list[int] = []
    per_street: dict = {}    # street -> list of (index_in_Pseq, role)
    for i, a in enumerate(acts):
        p, act0 = a['player_id'], a['action']
        if a['street'] != street:
            street = a['street']
            prev_agg = None
            acted = set()
        s = SIDX[street]
        amt, pot, to_call = a['amount'] / bb, a['pot_before'] / bb, a['to_call'] / bb
        stack0 = a['stack_before'] / bb
        aggressive = act0 in AGG and a['amount'] > a['to_call']
        allin = act0 == 'all_in' or (a['amount'] >= a['stack_before'] and a['amount'] > 0)
        if act0 == 'fold':
            act = 'fold'
        elif act0 == 'check':
            act = 'check'
        elif aggressive:
            act = 'jam' if allin else ('bet' if to_call <= 0 else 'raise')
        else:
            act = 'callin' if allin else 'call'
        # size bucket
        if act in ('bet', 'raise'):
            size = bucket(amt / pot, *zip(*SIZE_AGG)) if pot > 0 else 'nopot'
        elif act == 'jam':
            size = 'jam' + bucket(stack0 / max(pot, 1e-9), (1.0, 3.0, 1e9), ('sm', 'md', 'big'))
        elif act in ('call', 'callin'):
            size = 'c_' + (bucket(to_call / (pot + to_call), *zip(*ODDS_C)) if (pot + to_call) > 0 else 'free')
        elif act == 'fold':
            size = 'p_' + (bucket(to_call / (pot + to_call), *zip(*ODDS_C)) if (pot + to_call) > 0 else 'free')
        else:
            size = 'na'
        rl = role[p]
        # facing
        if prev_agg is None:
            facing = 'none'
        elif rl in ('A', 'B'):
            facing = 'partner' if role[prev_agg] in ('A', 'B') and prev_agg != p else 'out'
        else:
            facing = 'member' if role[prev_agg] in ('A', 'B') else 'out'
        # partner / pair state
        if rl in ('A', 'B'):
            partner = B if rl == 'A' else A
            if partner not in live:
                pstate = 'p_gone'
            elif partner in acted:
                pstate = 'p_acted'
            else:
                pstate = 'p_toact'
        else:
            nlive = sum(1 for q in (A, B) if q in live)
            pstate = f'pair{nlive}'
        v = st[p][s]
        stg = terc(v, cuts[s]) if v is not None else 'tx'
        others_live = [st[q][s] for q in live if q != p and st[q][s] is not None]
        bl = 'bl1' if (v is not None and others_live and v > max(others_live)) else 'bl0'

        psym = sym.id(0, f'{rl}.{act}.{facing}.{pstate}')
        rsym = sym.id(1, f'{rl}.{act}')
        qsym = sym.id(2, f'{rl}.{SNAME[s]}.{act}.{size}.{stg}')
        fstr = f'{rl}.{SNAME[s]}.{act}.{size}.{facing}.{pstate}.{stg}.{bl}'
        keys.append(pack(3, 1, (sym.id(3, fstr),)))
        seq.append('F.' + fstr)
        Pseq.append(psym)
        Rseq.append(rsym)
        Qseq.append(qsym)
        per_street.setdefault(s, []).append((len(Pseq) - 1, rl))

        acted.add(p)
        if aggressive:
            prev_agg = p
        if act0 == 'fold':
            live.discard(p)

    # n-grams
    for n in (1, 2, 3, 4):
        for i in range(len(Pseq) - n + 1):
            keys.append(pack(0, n, tuple(Pseq[i:i + n])))
    for n in (2, 3, 4):
        for i in range(len(Rseq) - n + 1):
            keys.append(pack(1, n, tuple(Rseq[i:i + n])))
    for n in (1, 2):
        for i in range(len(Qseq) - n + 1):
            keys.append(pack(2, n, tuple(Qseq[i:i + n])))
    # skip-bigrams A-action .. B-action on the same street (ordered, both directions by position)
    for s, lst in per_street.items():
        mem = [(i, r) for i, r in lst if r in ('A', 'B')]
        for x in range(len(mem)):
            for y in range(x + 1, len(mem)):
                i, ri = mem[x]
                j, rj = mem[y]
                if ri == rj:
                    continue
                adj = 1 if j == i + 1 else 0
                s1 = sym.id(4, f'{SNAME[s]}.adj{adj}')
                keys.append(pack(4, 3, (s1, Pseq[i], Pseq[j])))

    # ---- end tokens
    nets = {p: seat[p]['net_chips'] / bb for p in seat}
    ws = max(seat, key=lambda p: seat[p]['won_share'])
    hkey('E.winner.' + (role[ws] if seat[ws]['won_share'] > 0 else 'none'))
    hkey('E.sdA.' + ('1' if seat[A]['went_to_showdown'] else '0'))
    hkey('E.sdB.' + ('1' if seat[B]['went_to_showdown'] else '0'))
    hkey('E.nsd.' + str(min(meta['n_sd'], 4)))
    for nm, p in (('A', A), ('B', B)):
        hkey(f'E.net{nm}.' + netb(nets[p]))
    n1, n2 = nets[A], nets[B]
    tr = 'none'
    if (n1 > 0) != (n2 > 0) and min(n1, n2) < 0 and max(n1, n2) > 0:
        win, lose = (A, B) if n1 > n2 else (B, A)
        if seat[lose]['total_contribution'] / bb > 1.0:
            v = min(-nets[lose], nets[win])
            tr = ('A2B' if lose == A else 'B2A') + '.' + bucket(v, (2, 10, 30, 1e9), ('lt2', '2_10', '10_30', 'gt30'))
    hkey('E.transfer.' + tr)
    hkey('E.last.' + SNAME[SIDX[acts[-1]['street']]] if acts else 'E.last.none')
    hkey('E.pairshare.' + bucket(
        (seat[A]['total_contribution'] + seat[B]['total_contribution']) / max(meta['final_pot'], 1),
        (.2, .5, .8, 1e9), ('lo', 'mid', 'hi', 'all')))
    return keys, seq


def terc(v: float, c: tuple) -> str:
    return 't0' if v < c[0] else ('t1' if v < c[1] else 't2')


def netb(x: float) -> str:
    if abs(x) < .5:
        return 'zero'
    if x > 0:
        return 'win_sm' if x < 5 else ('win_md' if x < 25 else 'win_big')
    return 'los_sm' if x > -5 else ('los_md' if x > -25 else 'los_big')


# ---------------------------------------------------------------- driver
def load_keys():
    keys = pl.read_parquet(OOF, columns=['pair_id', 'hand_id', 'fold', 'label', 'listed',
                                         'behavior_family', 'table_id'])
    lab = pl.read_csv(DATA + r'\development_labels.csv').select('pair_id', 'player_1', 'player_2')
    keys = keys.join(lab, on='pair_id')
    return keys


def chunk_meta(part: pl.DataFrame):
    m = (pl.scan_parquet(DATA + r'\hands.parquet').join(part.lazy(), on='hand_id')
         .select('hand_id', 'phase', 'big_blind', 'board_cards', 'final_pot', 'players_dealt',
                 'players_at_showdown', 'button_seat').collect())
    assert (m['phase'] == 'development').all(), 'non-development hand in step255'
    seats = (pl.scan_parquet(DATA + r'\seats.parquet').join(part.lazy(), on='hand_id')
             .select('hand_id', 'player_id', 'seat_no', 'starting_stack', 'hole_card_1',
                     'hole_card_2', 'total_contribution', 'net_chips', 'folded',
                     'went_to_showdown', 'won_share').collect())
    acts = (pl.scan_parquet(DATA + r'\actions.parquet').join(part.lazy(), on='hand_id')
            .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount',
                    'amount_to', 'pot_before', 'stack_before', 'to_call')
            .collect().sort('hand_id', 'action_no'))
    srows, arows = {}, {}
    for r in seats.iter_rows(named=True):
        srows.setdefault(r['hand_id'], {})[r['player_id']] = r
    for r in acts.iter_rows(named=True):
        arows.setdefault(r['hand_id'], []).append(r)
    meta = {}
    for r in m.iter_rows(named=True):
        h = r['hand_id']
        board = (r['board_cards'] or '').split()
        sr = srows[h]
        seats_sorted = sorted(sr.values(), key=lambda x: x['seat_no'])
        nos = [x['seat_no'] for x in seats_sorted]
        bi = max([i for i, s in enumerate(nos) if s <= r['button_seat']], default=len(nos) - 1)
        ring = [seats_sorted[(bi + 1 + k) % len(nos)]['player_id'] for k in range(len(nos))]
        names = ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN'][:len(nos)]
        pos = {p: nm for p, nm in zip(ring, names)}
        order = ring[2:] + ring[:2] if len(nos) > 2 else ring
        meta[h] = {'bb': r['big_blind'], 'board': board, 'final_pot': r['final_pot'],
                   'n_sd': r['players_at_showdown'], 'pos': pos, 'order': order}
    strength = {}
    for h, sr in srows.items():
        board = meta[h]['board']
        strength[h] = {p: strengths_for(v['hole_card_1'], v['hole_card_2'], board)
                       for p, v in sr.items()}
    return meta, srows, arows, strength


def estimate_cuts(keys: pl.DataFrame) -> list[tuple]:
    rng = np.random.default_rng(255)
    hl = keys['hand_id'].unique().sort().to_list()
    samp = [hl[i] for i in rng.choice(len(hl), size=min(8000, len(hl)), replace=False)]
    part = pl.DataFrame({'hand_id': samp})
    meta, srows, _, strength = chunk_meta(part)
    vals = [[], [], [], []]
    for h, d in strength.items():
        for p, v in d.items():
            for s in range(4):
                if v[s] is not None:
                    vals[s].append(v[s])
    cuts = [tuple(np.quantile(np.array(v), [1 / 3, 2 / 3]).tolist()) for v in vals]
    return cuts


def main() -> None:
    t0 = time.time()
    keys = load_keys()
    lim = int(os.environ.get('STEP255_LIMIT', '0'))
    if lim:   # smoke mode: a deterministic subset of hands, outputs go nowhere important
        hl = keys['hand_id'].unique().sort().to_list()[:lim]
        keys = keys.join(pl.DataFrame({'hand_id': hl}), on='hand_id')
    print('rows', keys.height, 'hands', keys['hand_id'].n_unique(), flush=True)
    cuts = estimate_cuts(keys)
    print('strength terciles', [[round(x, 4) for x in c] for c in cuts], flush=True)

    # role A/B per pair: A = lower summed net_chips over the pair's shared hands
    hid_all = keys.select('hand_id').unique()
    nets = (pl.scan_parquet(DATA + r'\seats.parquet').join(hid_all.lazy(), on='hand_id')
            .select('hand_id', 'player_id', 'net_chips').collect())
    ph = keys.select('pair_id', 'hand_id', 'player_1', 'player_2')
    n1 = (ph.join(nets.rename({'player_id': 'player_1', 'net_chips': 'n1'}),
                  on=['hand_id', 'player_1'], how='left')
          .group_by('pair_id').agg(pl.col('n1').sum()))
    n2 = (ph.join(nets.rename({'player_id': 'player_2', 'net_chips': 'n2'}),
                  on=['hand_id', 'player_2'], how='left')
          .group_by('pair_id').agg(pl.col('n2').sum()))
    rr = (keys.select('pair_id', 'player_1', 'player_2').unique()
          .join(n1, on='pair_id').join(n2, on='pair_id'))
    a_is_1 = {r['pair_id']: ((r['n1'], r['player_1']) <= (r['n2'], r['player_2']))
              for r in rr.iter_rows(named=True)}
    print('pairs with A=player_1:', sum(a_is_1.values()), '/', len(a_is_1), flush=True)

    sym = Sym()
    hand_list = keys['hand_id'].unique().sort().to_list()
    bounds = np.linspace(0, len(hand_list), NCHUNK + 1).astype(int)
    row_keys: list[tuple] = []       # (pair_id, hand_id) in emission order
    seq_all = np.zeros((keys.height, MAXLEN), dtype=np.int32)
    seq_len = np.zeros(keys.height, dtype=np.int32)
    seq_vocab: dict = {}
    nrow = 0
    per_row_n: list[int] = []
    for c in range(NCHUNK):
        part = pl.DataFrame({'hand_id': hand_list[bounds[c]:bounds[c + 1]]})
        meta, srows, arows, strength = chunk_meta(part)
        sub = keys.join(part, on='hand_id')
        buf: list[int] = []
        for r in sub.iter_rows(named=True):
            h = r['hand_id']
            if h not in arows:
                kk, ss = [], []
            else:
                kk, ss = tok_hand(r['player_1'], r['player_2'], a_is_1[r['pair_id']], meta[h],
                                  srows[h], arows[h], strength[h], cuts, sym)
            buf.extend(kk)
            per_row_n.append(len(kk))
            row_keys.append((r['pair_id'], h))
            ids = []
            for t in ss[:MAXLEN]:
                v = seq_vocab.get(t)
                if v is None:
                    v = len(seq_vocab) + 1     # 0 = padding
                    seq_vocab[t] = v
                ids.append(v)
            seq_all[nrow, :len(ids)] = ids
            seq_len[nrow] = len(ids)
            nrow += 1
        np.save(TMP + f'\\keys_{c:02d}.npy', np.array(buf, dtype=np.int64))
        print(f'chunk {c + 1}/{NCHUNK} rows={nrow} keys={len(buf)} '
              f'{time.time() - t0:.0f}s', flush=True)
    assert nrow == keys.height, (nrow, keys.height)
    indptr = np.concatenate([[0], np.cumsum(np.array(per_row_n, dtype=np.int64))])
    np.save(TMP + r'\indptr.npy', indptr)
    json.dump({'symbols': sym.dump(), 'cuts': [list(c) for c in cuts],
               'seq_vocab': seq_vocab, 'n_rows': nrow, 'maxlen': MAXLEN, 'min_df': MIN_DF},
              open(OUT + r'\tokeniser255.json', 'w', encoding='utf-8'))
    pl.DataFrame({'pair_id': [x[0] for x in row_keys],
                  'hand_id': [x[1] for x in row_keys]}).write_parquet(OUT + r'\row_order255.parquet')
    np.savez_compressed(OUT + r'\seq255.npz', seq=seq_all, length=seq_len)
    print('symbols per view', {v: len(sym.tab[i]) for i, v in enumerate(VIEWS)},
          'seq vocab', len(seq_vocab), 'total keys', int(indptr[-1]),
          'mean keys/row', round(float(indptr[-1] / nrow), 1), f'{time.time() - t0:.0f}s',
          flush=True)
    print('DONE', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
