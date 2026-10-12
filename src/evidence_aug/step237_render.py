"""STEP237 -- compact text renderer of a hand from a labelled pair's point of view.

Development phase only. No player ids in the output: members are A / B (A = the member who
loses more chips over the pair's shared hands; ties broken by player_id order) and outsiders are
O1..On in seat order. Every quantity is in big blinds. Strength percentiles come from phevaluator
ranks of the two hole cards on the board prefix visible at that street (same convention as
src/advanced_features.py); preflop uses the same bounded rank/suit/connectivity heuristic.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '1')

import polars as pl
from phevaluator import Card, evaluate_cards

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CARDS = {r + s: Card.to_id(r + s) for r in '23456789TJQKA' for s in 'cdhs'}
STREETS = ['preflop', 'flop', 'turn', 'river']
NBOARD = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}


def preflop_strength(c1: int, c2: int) -> float:
    r1, r2 = c1 // 4 + 2, c2 // 4 + 2
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return .55 + hi / 32
    return min(.95, max(.05, (hi + lo) / 32 - max(hi - lo - 1, 0) / 40 + .07 * (c1 % 4 == c2 % 4)))


def street_strengths(c1: str, c2: str, board: list[str]) -> dict[str, float]:
    a, b = CARDS[c1], CARDS[c2]
    out = {'preflop': preflop_strength(a, b)}
    ids = [CARDS[c] for c in board]
    for s in ('flop', 'turn', 'river'):
        n = NBOARD[s]
        out[s] = (7463 - evaluate_cards(a, b, *ids[:n])) / 7462 if len(ids) >= n else None
    return out


def order_and_labels(seat_nos: list[int], button_seat: int) -> tuple[list[int], dict[int, str]]:
    """Return seats in preflop action order and a position label per seat."""
    seats = sorted(seat_nos)
    n = len(seats)
    if button_seat in seats:
        bi = seats.index(button_seat)
    else:  # button seat sat out: take the nearest lower seat
        bi = max([i for i, s in enumerate(seats) if s <= button_seat], default=n - 1)
    ring = [seats[(bi + 1 + k) % n] for k in range(n)]  # SB, BB, ..., BTN
    if n == 2:
        names = ['SB/BTN', 'BB']
        ring = [seats[bi], seats[(bi + 1) % n]]
    else:
        mids = ['UTG', 'UTG1', 'MP', 'MP1', 'HJ', 'CO'][-(n - 3):] if n > 3 else []
        names = ['SB', 'BB'] + mids + ['BTN']
    lab = {s: nm for s, nm in zip(ring, names)}
    if n == 2:
        action_order = [ring[0], ring[1]]
    else:
        action_order = ring[2:] + ring[:2]  # UTG ... BTN, SB, BB
    return action_order, lab


class HandBook:
    """Loads hands/seats/actions for a fixed set of hand_ids (development phase only)."""

    def __init__(self, hand_ids):
        ids = pl.DataFrame({'hand_id': list(dict.fromkeys(hand_ids))})
        self.hands = (pl.scan_parquet(DATA + r'\hands.parquet').join(ids.lazy(), on='hand_id')
                      .select('hand_id', 'table_id', 'phase', 'button_seat', 'small_blind',
                              'big_blind', 'board_cards', 'final_pot', 'players_dealt',
                              'players_at_showdown').collect())
        assert self.hands.height == ids.height, 'missing hands'
        assert (self.hands['phase'] == 'development').all(), 'non-development hand in sample'
        self.seats = (pl.scan_parquet(DATA + r'\seats.parquet').join(ids.lazy(), on='hand_id')
                      .collect())
        self.acts = (pl.scan_parquet(DATA + r'\actions.parquet').join(ids.lazy(), on='hand_id')
                     .collect().sort('hand_id', 'action_no'))
        self.h = {r['hand_id']: r for r in self.hands.iter_rows(named=True)}
        self.s, self.a = {}, {}
        for r in self.seats.iter_rows(named=True):
            self.s.setdefault(r['hand_id'], []).append(r)
        for r in self.acts.iter_rows(named=True):
            self.a.setdefault(r['hand_id'], []).append(r)

    def render(self, hand_id: str, member_a: str, member_b: str, header: str = '') -> str:
        h = self.h[hand_id]
        bb = h['big_blind']
        seats = sorted(self.s[hand_id], key=lambda r: r['seat_no'])
        board = (h['board_cards'] or '').split()
        order, lab = order_and_labels([r['seat_no'] for r in seats], h['button_seat'])
        alias, k = {}, 0
        for r in seats:
            if r['player_id'] == member_a:
                alias[r['player_id']] = 'A'
            elif r['player_id'] == member_b:
                alias[r['player_id']] = 'B'
            else:
                k += 1
                alias[r['player_id']] = f'O{k}'
        seat_of = {r['player_id']: r['seat_no'] for r in seats}
        by_seat = {r['seat_no']: r for r in seats}
        strength = {r['player_id']: street_strengths(r['hole_card_1'], r['hole_card_2'], board)
                    for r in seats}
        L = []
        L.append(header)
        L.append(f'  bb={bb} dealt={h["players_dealt"]} pot_final={h["final_pot"] / bb:.1f}bb '
                 f'showdown_players={h["players_at_showdown"]}')
        who = []
        for s in order:
            r = by_seat[s]
            p = r['player_id']
            who.append(f'{alias[p]}[{lab[s]}] {r["hole_card_1"]}{r["hole_card_2"]} '
                       f'stk{r["starting_stack"] / bb:.0f}')
        L.append('  seats (preflop order): ' + ' | '.join(who))
        if board:
            bs = f'flop {" ".join(board[:3])}'
            if len(board) >= 4:
                bs += f' | turn {board[3]}'
            if len(board) >= 5:
                bs += f' | river {board[4]}'
            L.append('  board: ' + bs)
        # strength table for streets that were dealt
        reached = [s for s in STREETS if len(board) >= NBOARD[s]]
        for st in reached:
            vals = []
            for s in order:
                p = by_seat[s]['player_id']
                v = strength[p][st]
                if v is None:
                    continue
                vals.append((alias[p], v))
            rank_of = {a: i + 1 for i, (a, _) in enumerate(sorted(vals, key=lambda t: -t[1]))}
            L.append(f'  str[{st[:2]}]: ' + ' '.join(
                f'{a}={v:.3f}(#{rank_of[a]})' for a, v in vals))
        # actions
        cur, prev_agg, live = None, None, None
        for r in self.a[hand_id]:
            if r['street'] != cur:
                cur = r['street']
                prev_agg = None
                L.append(f'  -- {cur.upper()} --')
            p = r['player_id']
            al = alias[p]
            amt, to_call, pot = r['amount'] / bb, r['to_call'] / bb, r['pot_before'] / bb
            bits = [f'    {al}[{lab[seat_of[p]]}]', r['action']]
            if r['action'] in ('bet', 'raise', 'all_in'):
                bits.append(f'{amt:.1f}bb to {r["amount_to"] / bb:.1f}')
                if pot > 0:
                    bits.append(f'({amt / pot:.2f}xpot)')
            elif r['action'] == 'call':
                bits.append(f'{amt:.1f}bb')
                if to_call > 0:
                    bits.append(f'(odds {to_call / (pot + to_call):.2f})')
            elif r['action'] == 'fold' and to_call > 0:
                bits.append(f'(to_call {to_call:.1f}bb, odds {to_call / (pot + to_call):.2f})')
            bits.append(f'| pot{pot:.1f} live{r["players_active"]}')
            if prev_agg is not None and prev_agg != p:
                bits.append(f'| facing {alias[prev_agg]}')
            v = strength[p][r['street']]
            if v is not None:
                bits.append(f'| str {v:.3f}')
            L.append(' '.join(bits))
            if r['action'] in ('bet', 'raise', 'all_in') and r['amount'] > r['to_call']:
                prev_agg = p
        nets = ' '.join(f'{alias[r["player_id"]]}{r["net_chips"] / bb:+.1f}'
                        + ('*SD' if r['went_to_showdown'] else '') for r in seats)
        L.append('  net_bb: ' + nets)
        return '\n'.join(L)


def roles(pairs: pl.DataFrame, nets: pl.DataFrame) -> dict[str, tuple[str, str]]:
    """A = the member with the lower summed net_chips over the pair's shared hands."""
    out = {}
    for r in pairs.iter_rows(named=True):
        n1 = nets.get(r['pair_id'], {}).get(r['player_1'], 0.)
        n2 = nets.get(r['pair_id'], {}).get(r['player_2'], 0.)
        if (n1, r['player_1']) <= (n2, r['player_2']):
            out[r['pair_id']] = (r['player_1'], r['player_2'])
        else:
            out[r['pair_id']] = (r['player_2'], r['player_1'])
    return out
