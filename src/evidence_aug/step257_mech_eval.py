"""STEP257: the STEP254 662-column mechanical block for ONE phase, parameterised.

src/evidence_aug/step254_features.py is imported unchanged, so every layout constant, the emission
index arrays, board_texture() and process_hand() - i.e. the whole computation - are literally the
STEP254 code.  What this file adds is only the I/O shell of step254_features.main()'s chunk loop with
the phase and the key set parameterised, exactly as STEP254's REPORT.md section 8 prescribes:

  * the phase assert is 'evaluation' (or 'development' for the parity gate) instead of hard wired;
  * the (pair, hand) keys and the pair -> (player_1, player_2) map come from that phase's own pair
    list, never from the other phase;
  * step254_pf169.npz is reused unchanged (it is a property of the deck, not of a phase);
  * the degeneracy prune is NOT re-run: the 662 kept columns and their order are frozen from the
    development build (step254_features.parquet's column order).

A "pair-hand" is a hand in which both members of the pair were dealt.  The chronology family (t4) is
supplied by step257_prep_eval.py and joined on, exactly as step254_features.main() joins
step254_chrono.parquet.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True

import step254_eval7 as E            # noqa: E402
import step254_features as F         # noqa: E402  (layout + process_hand + board_texture, unchanged)

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
S254 = CV + r'\experiments\step254_mechanical_tensor_20260917'


def kept_columns() -> tuple[list[str], list[str], list[str]]:
    """the frozen 662: (all, in-hand part, chronology part), in the development file's order."""
    ref = pl.read_parquet(S254 + r'\step254_features.parquet', n_rows=1)
    allc = [c for c in ref.columns if c not in ('pair_id', 'hand_id')]
    inhand = set(F.COLS)
    return allc, [c for c in allc if c in inhand], [c for c in allc if c not in inhand]


def load_pf169():
    pf = np.load(S254 + r'\step254_pf169.npz')
    return pf['pct169'], pf['eq169x169']


def build_chunk(keys: pl.DataFrame, phase: str, PCT169, EQ169) -> pl.DataFrame:
    """keys: pair_id, hand_id, player_1, player_2, p1_is_A for the hands of ONE chunk.

    Returns one row per (pair_id, hand_id) with the F.COLS in-hand mechanical columns.
    """
    part = keys.select('hand_id').unique()
    m = (pl.scan_parquet(DATA + r'\hands.parquet').join(part.lazy(), on='hand_id')
         .select('hand_id', 'phase', 'big_blind', 'board_cards', 'final_pot', 'players_dealt',
                 'players_at_showdown', 'button_seat').collect())
    assert m.height == part.height, 'hands missing'
    assert (m['phase'] == phase).all(), 'a hand of this chunk is not %s-phase' % phase
    m = m.drop('phase')
    seats = (pl.scan_parquet(DATA + r'\seats.parquet').join(part.lazy(), on='hand_id')
             .select('hand_id', 'player_id', 'seat_no', 'starting_stack', 'hole_card_1',
                     'hole_card_2', 'total_contribution', 'net_chips', 'folded',
                     'went_to_showdown', 'won_share').collect())
    acts = (pl.scan_parquet(DATA + r'\actions.parquet').join(part.lazy(), on='hand_id')
            .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount',
                    'amount_to', 'pot_before', 'stack_before', 'to_call').collect()
            .sort('hand_id', 'action_no'))
    # ---- boards and strengths, vectorised
    hidx = {h: i for i, h in enumerate(m['hand_id'].to_list())}
    nh = m.height
    board = np.full((nh, 5), -1, np.int64)
    nboard = np.zeros(nh, np.int64)
    for i, bc in enumerate(m['board_cards'].to_list()):
        cs = (bc or '').split()
        nboard[i] = len(cs)
        for j, cc in enumerate(cs):
            board[i, j] = E.CARD_ID[cc]
    si = np.array([hidx[h] for h in seats['hand_id'].to_list()], np.int64)
    c1 = np.array([E.CARD_ID[x] for x in seats['hole_card_1'].to_list()], np.int64)
    c2 = np.array([E.CARD_ID[x] for x in seats['hole_card_2'].to_list()], np.int64)
    ns = len(si)
    sarr = np.full((ns, 4), np.nan, np.float32)
    clsarr = np.full((ns, 4), -1., np.float32)
    drw = np.zeros((ns, 4), np.float32)
    cls169 = np.where(((c1 & 3) == (c2 & 3)) | ((c1 >> 2) == (c2 >> 2)),
                      np.maximum(c1 >> 2, c2 >> 2) * 13 + np.minimum(c1 >> 2, c2 >> 2),
                      np.minimum(c1 >> 2, c2 >> 2) * 13 + np.maximum(c1 >> 2, c2 >> 2))
    sarr[:, 0] = PCT169[cls169]
    nbs = nboard[si]
    for k, nb in ((1, 3), (2, 4), (3, 5)):
        sel = np.flatnonzero(nbs >= nb)
        if not len(sel):
            continue
        cards = np.concatenate([c1[sel, None], c2[sel, None], board[si[sel], :nb]], axis=1)
        rk = E.evaluate(cards)
        sarr[sel, k] = (7463.0 - rk) / 7462.0
        clsarr[sel, k] = F.made_class(rk)
        if nb in (3, 4):
            mask = np.bitwise_or.reduce(np.int64(1) << (cards >> 2), axis=1)
            sc = np.stack([((cards & 3) == su).sum(axis=1) for su in range(4)], axis=1)
            base = 0 if nb == 3 else 2
            drw[sel, base] = (sc.max(axis=1) == 4).astype(np.float32)
            drw[sel, base + 1] = (F.HAS4[mask] & ~F.HASS[mask]).astype(np.float32)
    drw2 = drw
    # ---- python side tables
    srows, stv, clsv, drwv, holev = {}, {}, {}, {}, {}
    for j, r in enumerate(seats.iter_rows(named=True)):
        h = r['hand_id']
        p = r['player_id']
        srows.setdefault(h, {})[p] = r
        stv.setdefault(h, {})[p] = sarr[j]
        clsv.setdefault(h, {})[p] = clsarr[j]
        drwv.setdefault(h, {})[p] = drw2[j]
        holev.setdefault(h, {})[p] = (int(c1[j]), int(c2[j]), int(cls169[j]))
    arows: dict = {}
    for r in acts.iter_rows(named=True):
        arows.setdefault(r['hand_id'], []).append(r)
    mrows = {r['hand_id']: r for r in m.iter_rows(named=True)}
    sub = keys
    n = sub.height
    Vout = np.zeros((n, F.NCOL), np.float32)
    rowsA = np.zeros((n, 2), np.int64)
    rowsB = np.zeros((n, 2), np.int64)
    rowh = np.zeros(n, np.int64)
    pids, hids = [], []
    eq_need = {1: [], 2: []}
    for i, r in enumerate(sub.iter_rows(named=True)):
        h = r['hand_id']
        mA = r['player_1'] if r['p1_is_A'] == 1 else r['player_2']
        mB = r['player_2'] if r['p1_is_A'] == 1 else r['player_1']
        pids.append(r['pair_id'])
        hids.append(h)
        mr = mrows[h]
        bb = mr['big_blind']
        seat = srows[h]
        st = stv[h]
        fin_best = max((st[q][3] if st[q][3] == st[q][3] else st[q][2] if st[q][2] == st[q][2]
                        else st[q][1] if st[q][1] == st[q][1] else st[q][0])
                       for q in seat if not seat[q]['folded'])
        stf = {q: np.where(np.isnan(st[q]), -1., st[q]) for q in seat}
        for q in seat:
            a4 = stf[q]
            for kk in range(1, 4):
                if a4[kk] < 0:
                    a4[kk] = a4[kk - 1]
        v = np.zeros((2, F.NPM), np.float32)
        v[:, F.B_FSP:F.B_FSP + 6] = 9.
        v[:, F.B_FSA:F.B_FSA + 5] = 9.
        v[:, F.B_ARNK:F.B_ARNK + 3] = 9.
        v[:, F.B_SPRS:F.B_SPRS + 4] = -1.
        hl = np.zeros(F.NHL, np.float32)
        button = mr['button_seat']
        o_pre = {s: (s - button - 3) % 6 for s in range(6)}
        o_post = {s: (s - button - 1) % 6 for s in range(6)}
        F.process_hand(mA, mB, arows.get(h, []), seat, stf, fin_best,
                       mr['final_pot'] / bb, bb, v, hl, button, (o_pre, o_post))
        for k, p in enumerate((mA, mB)):
            a, b, cl = holev[h][p]
            r1, r2 = a >> 2, b >> 2
            v[k, F.B_HOLE + 0] = max(r1, r2) + 2
            v[k, F.B_HOLE + 1] = min(r1, r2) + 2
            v[k, F.B_HOLE + 2] = 1. if r1 == r2 else 0.
            v[k, F.B_HOLE + 3] = 1. if (a & 3) == (b & 3) else 0.
            v[k, F.B_HOLE + 4] = abs(r1 - r2)
            v[k, F.B_HOLE + 5] = PCT169[cl]
            v[k, F.B_STR:F.B_STR + 3] = np.where(np.isnan(st[p][1:]), -1., st[p][1:])
            v[k, F.B_CLS:F.B_CLS + 3] = clsv[h][p][1:]
            v[k, F.B_DRAW:F.B_DRAW + 4] = drwv[h][p]
        for k, p in enumerate((mA, mB)):
            for s in range(1, 4):
                if np.isnan(st[p][s]):
                    v[k, F.B_RNK + s - 1] = -1.
                    continue
                nb2 = sum(1 for q in seat if q != p and not seat[q]['folded']
                          and st[q][s] > st[p][s])
                v[k, F.B_RNK + s - 1] = nb2 + 1
                v[k, F.B_BEST + s - 1] = 1. if nb2 == 0 else 0.
        nbd = int(nboard[hidx[h]])
        for s in range(1, 4):
            if nbd >= s + 2:
                tex = F.board_texture(board[hidx[h]], s + 2)
                hl[F.H_BRD + (s - 1) * 5:F.H_BRD + s * 5] = tex
            else:
                hl[F.H_BRD + (s - 1) * 5:F.H_BRD + s * 5] = -1.
        for s in range(4):
            a_s, b_s = st[mA][s], st[mB][s]
            if a_s != a_s or b_s != b_s:
                hl[F.H_AHEAD + s] = -9.
                hl[F.H_SGAP + s] = -9.
            else:
                hl[F.H_AHEAD + s] = 1. if a_s > b_s else (0. if a_s == b_s else -1.)
                hl[F.H_SGAP + s] = a_s - b_s
        hl[F.H_EQ:F.H_EQ + 4] = -1.
        hl[F.H_EQM:F.H_EQM + 4] = -1.
        ca1, ca2, cla = holev[h][mA]
        cb1, cb2, clb = holev[h][mB]
        hl[F.H_EQ] = EQ169[cla, clb]
        if nbd == 5:
            a5, b5 = st[mA][3], st[mB][3]
            hl[F.H_EQ + 3] = 1. if a5 > b5 else (.5 if a5 == b5 else 0.)
        rowsA[i] = (ca1, ca2)
        rowsB[i] = (cb1, cb2)
        rowh[i] = hidx[h]
        if hl[F.H_BOTHLIVE + 1] > 0 and nbd >= 3:
            eq_need[1].append(i)
        if hl[F.H_BOTHLIVE + 2] > 0 and nbd >= 4:
            eq_need[2].append(i)
        hl[F.H_SDN] = float(mr['players_at_showdown'])
        row = Vout[i]
        row[F.C_A] = v[0, F.P_A]
        row[F.C_B] = v[1, F.P_B]
        row[F.C_SM] = v[0, F.P_SM] + v[1, F.P_SM]
        row[F.C_MX] = np.maximum(v[0, F.P_MX], v[1, F.P_MX])
        row[F.C_MN] = np.minimum(v[0, F.P_MN], v[1, F.P_MN])
        row[F.N_EMIT:] = hl
    # ---- enumerated equity, vectorised
    for street, nb in ((1, 3), (2, 4)):
        idx = np.array(eq_need[street], np.int64)
        for a0 in range(0, len(idx), 400):
            sel = idx[a0:a0 + 400]
            R = len(sel)
            bd = board[rowh[sel], :nb]
            used = np.concatenate([rowsA[sel], rowsB[sel], bd], axis=1)
            mask = np.ones((R, 52), bool)
            mask[np.arange(R)[:, None], used] = False
            avail = np.nonzero(mask)[1].reshape(R, 52 - 4 - nb)
            if nb == 3:
                run = avail[:, F.PAIRS45]
            else:
                run = avail[:, :, None]
            nr = run.shape[1]
            base = np.concatenate([bd, np.zeros((R, 0), np.int64)], axis=1)
            bb_ = np.repeat(base[:, None, :], nr, axis=1)
            ha = np.repeat(rowsA[sel][:, None, :], nr, axis=1)
            hb = np.repeat(rowsB[sel][:, None, :], nr, axis=1)
            ca = np.concatenate([ha, bb_, run], axis=2).reshape(R * nr, -1)
            cb = np.concatenate([hb, bb_, run], axis=2).reshape(R * nr, -1)
            va = E.evaluate(ca).reshape(R, nr)
            vb = E.evaluate(cb).reshape(R, nr)
            eq = (np.where(va < vb, 1., np.where(va == vb, .5, 0.))).mean(axis=1)
            Vout[sel, F.N_EMIT + F.H_EQ + street] = eq
    for s in range(4):
        e = Vout[:, F.N_EMIT + F.H_EQ + s]
        Vout[:, F.N_EMIT + F.H_EQM + s] = np.where(e < 0, -1., np.maximum(e, 1. - e))
    return pl.DataFrame({'pair_id': pids, 'hand_id': hids}).hstack(
        pl.DataFrame(Vout, schema=F.COLS))
