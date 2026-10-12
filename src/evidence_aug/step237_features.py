"""STEP237 step 3: 25 per-hand pair features aimed at the INVISIBLE injected events.

Development phase only (asserted). One row per (pair_id, hand_id) for every hand of every
labelled pair. Roles: most features are symmetric (max / sum over the two members); the three
transfer features use the WITHIN-HAND loser (the member with the more negative net_chips).
Hole cards of outsiders are used by the five features marked OUTSIDER_CARDS below; they are
disclosed in seats.parquet for both phases.

Hypotheses behind the block (see REPORT.md):
  H1  a soft fold is a fold facing the partner while holding the best live hand / a premium
      preflop holding / at a cheap price, after voluntarily investing on that street;
  H2  coordinated isolation is a member's aggressive action taken while the partner is still
      to act, which the partner then folds to, clearing the field;
  H3  a directed transfer shows up as chips flowing from one member to the other inside a pot
      that the pair contests alone, or as the winner being strong and passive at showdown.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault('POLARS_MAX_THREADS', '1')

import math

import numpy as np
import polars as pl
from phevaluator import Card, evaluate_cards

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
OOF = ROOT + r'\Claude Version\experiments\step232_event_detector_20260917\hand_oof.parquet'
OUT = ROOT + r'\Claude Version\experiments\step237_invisible_events_20260917'
CARDS = {r + s: Card.to_id(r + s) for r in '23456789TJQKA' for s in 'cdhs'}
RANKS = {r: i + 2 for i, r in enumerate('23456789TJQKA')}
SIDX = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
NB = (0, 3, 4, 5)
AGG = ('bet', 'raise', 'all_in')

FEATURES = [
    'fp_best_live', 'fp_gap', 'fp_premium_pf', 'fp_price_min', 'fp_wouldwin', 'fp_best_pot',
    'fp_partner_wins', 'fp_betfold', 'fp_limpfold', 'fp_count',
    'iso_out_fold_share', 'iso_clean', 'iso_weak_agg_behind', 'iso_agg_partner_folds',
    'oof_clearway',
    'tr_internal_bb', 'tr_loser_share', 'hu_pair_streets', 'pay_partner_bb', 'passive_winner',
    'checkdown_best', 'coldcall_partner_bb', 'call_partner_dominated_bb', 'both_agg_live',
    'pair_pot_share',
]
OUTSIDER_CARD_FEATURES = ['fp_best_live', 'fp_wouldwin', 'fp_best_pot', 'oof_clearway',
                          'passive_winner', 'checkdown_best']


def preflop_strength(c1: int, c2: int) -> float:
    r1, r2 = c1 // 4 + 2, c2 // 4 + 2
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return .55 + hi / 32
    return min(.95, max(.05, (hi + lo) / 32 - max(hi - lo - 1, 0) / 40 + .07 * (c1 % 4 == c2 % 4)))


def premium_score(c1: str, c2: str) -> float:
    r1, r2 = RANKS[c1[0]], RANKS[c2[0]]
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return 1.0 if hi >= 10 else (.7 if hi >= 7 else .4)
    if hi == 14 and lo == 13:
        return 1.0
    if hi == 14 and lo >= 11:
        return .7
    if hi == 13 and lo == 12:
        return .7
    return .4 if lo >= 10 else 0.


def hand_tables(seats: pl.DataFrame, meta: dict):
    """strength[hand][player] = (s_pre, s_flop, s_turn, s_river) with None past the board;
    final[hand][player] = strength on the complete board."""
    strength, final = {}, {}
    for h, p, c1, c2 in seats.select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2').iter_rows():
        board = meta[h]['board']
        a, b = CARDS[c1], CARDS[c2]
        vals = [preflop_strength(a, b)]
        for n in NB[1:]:
            vals.append((7463 - evaluate_cards(a, b, *board[:n])) / 7462 if len(board) >= n else None)
        strength.setdefault(h, {})[p] = vals
        final.setdefault(h, {})[p] = next(v for v in reversed(vals) if v is not None)
    return strength, final


def pairhand_features(pid, hid, m1, m2, meta, seatrows, acts, strength, final):
    bb = meta['bb']
    st, fin = strength, final
    seat = seatrows  # player -> dict
    others = [p for p in seat if p not in (m1, m2)]
    members = (m1, m2)
    # end-of-hand facts
    nonfold_best = max((fin[p] for p, r in seat.items() if not r['folded']), default=0.)
    sd_players = [p for p, r in seat.items() if r['went_to_showdown']]
    sd_best = max((fin[p] for p in sd_players), default=0.)
    nets = {p: seat[p]['net_chips'] / bb for p in seat}
    contrib = {p: seat[p]['total_contribution'] / bb for p in seat}
    pot_bb = meta['final_pot'] / bb
    n_out = max(meta['players_dealt'] - 2, 1)

    f = {k: 0. for k in FEATURES}
    f['fp_price_min'] = 1.0

    live = set(seat)
    prev_agg = None
    street = 'preflop'
    invested = {p: 0. for p in seat}          # chips put in on the current street (no blinds)
    limped = {p: False for p in seat}
    post_agg = {p: 0 for p in seat}           # aggressive actions after the flop
    hu_streets = set()
    out_folds_to_pair = 0
    partner_folds_to = {m1: False, m2: False}  # partner folded facing this member
    agg_by = {m1: False, m2: False}
    last_pf_agg = None
    # index of later actions per street/player for "partner still to act"
    later = {}
    for i, a in enumerate(acts):
        later.setdefault((a['street'], a['player_id']), []).append(i)

    for i, a in enumerate(acts):
        p, act = a['player_id'], a['action']
        if a['street'] != street:
            street = a['street']
            prev_agg = None
            invested = {q: 0. for q in seat}
        s = SIDX[street]
        if s > 0 and live == {m1, m2}:
            hu_streets.add(s)
        aggressive = act in AGG and a['amount'] > a['to_call']
        amt = a['amount'] / bb
        pot = a['pot_before'] / bb
        to_call = a['to_call'] / bb
        odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.
        mine = p in members
        partner = m2 if p == m1 else (m1 if p == m2 else None)
        mystr = st[p][s]

        if act == 'fold':
            if prev_agg in members and not mine:
                out_folds_to_pair += 1
            if mine and prev_agg == partner:
                partner_folds_to[partner] = True
                f['fp_count'] += 1
                best_other = max((st[q][s] for q in live if q != p), default=0.)
                best_live = 1. if mystr > best_other else 0.
                f['fp_best_live'] = max(f['fp_best_live'], best_live)
                f['fp_gap'] = max(f['fp_gap'], max(0., mystr - st[partner][s]))
                if s == 0:
                    f['fp_premium_pf'] = max(f['fp_premium_pf'],
                                             premium_score(seat[p]['hole_card_1'], seat[p]['hole_card_2']))
                    if limped[p]:
                        f['fp_limpfold'] = 1.
                f['fp_price_min'] = min(f['fp_price_min'], odds)
                if fin[p] > nonfold_best:
                    f['fp_wouldwin'] = 1.
                f['fp_best_pot'] = max(f['fp_best_pot'], best_live * math.log1p(max(pot, 0.)))
                if nets[partner] > 0:
                    f['fp_partner_wins'] = 1.
                if invested[p] > 0:
                    f['fp_betfold'] = 1.
            if mine and prev_agg is not None and prev_agg not in members and partner in live:
                best_other = max((st[q][s] for q in live if q != p), default=0.)
                if mystr > best_other:
                    f['oof_clearway'] = max(f['oof_clearway'], 1. - odds)

        if act == 'call':
            invested[p] += amt
            if mine and prev_agg == partner:
                if s == 0 and a['amount_to'] == a['amount']:
                    f['coldcall_partner_bb'] = max(f['coldcall_partner_bb'], amt)
                elif s > 0 and a['amount_to'] == a['amount']:
                    f['coldcall_partner_bb'] = max(f['coldcall_partner_bb'], amt)
                if fin[p] < fin[partner]:
                    f['call_partner_dominated_bb'] += amt
                if s > 0 and nets[partner] > 0:
                    f['pay_partner_bb'] += amt
            if mine and s == 0 and a['amount_to'] == meta['bb'] and a['to_call'] > 0:
                limped[p] = True

        if aggressive:
            invested[p] += amt
            if s > 0 and mine:
                post_agg[p] += 1
            if mine:
                agg_by[p] = True
                if s == 0:
                    last_pf_agg = p
                if partner in live:
                    acts_after = any(j > i for j in later.get((street, partner), []))
                    if acts_after:
                        f['iso_weak_agg_behind'] = max(f['iso_weak_agg_behind'], 1. - mystr)
                    # partner folds later on this street facing this action
                    for j in later.get((street, partner), []):
                        if j > i:
                            if acts[j]['action'] == 'fold':
                                f['iso_agg_partner_folds'] += math.log1p(amt)
                            break
        elif act in AGG:  # all-in call
            invested[p] += amt
            if mine and prev_agg == partner and s > 0 and nets[partner] > 0:
                f['pay_partner_bb'] += amt

        if act == 'check' and mine and s == SIDX[acts[-1]['street']] and seat[p]['went_to_showdown']:
            best_other = max((st[q][s] for q in live if q != p), default=0.)
            if mystr > best_other and partner in live:
                f['checkdown_best'] = 1.

        if aggressive:
            prev_agg = p
        if act == 'fold':
            live.discard(p)

    # isolation aggregates
    f['iso_out_fold_share'] = out_folds_to_pair / n_out
    flop_out = sum(1 for p in others if not any(
        a['player_id'] == p and a['street'] == 'preflop' and a['action'] == 'fold' for a in acts))
    if last_pf_agg is not None and partner_folds_to.get(last_pf_agg) and flop_out <= 1:
        f['iso_clean'] = 1.
    f['both_agg_live'] = 1. if agg_by[m1] and agg_by[m2] else 0.
    f['hu_pair_streets'] = float(len(hu_streets))
    f['pair_pot_share'] = (contrib[m1] + contrib[m2]) / pot_bb if pot_bb > 0 else 0.

    # transfer block (within-hand loser / winner)
    n1, n2 = nets[m1], nets[m2]
    if (n1 > 0) != (n2 > 0) and min(n1, n2) < 0 and max(n1, n2) > 0:
        win, lose = (m1, m2) if n1 > n2 else (m2, m1)
        if contrib[lose] > 1.0:
            f['tr_internal_bb'] = min(-nets[lose], nets[win])
            if seat[win]['won_share'] >= .5:
                f['tr_loser_share'] = contrib[lose] / pot_bb if pot_bb > 0 else 0.
    for p in members:
        q = m2 if p == m1 else m1
        if (seat[p]['went_to_showdown'] and post_agg[p] == 0 and fin[p] >= sd_best
                and contrib[q] >= 2.0):
            f['passive_winner'] = 1.
    return f


def main() -> None:
    keys = pl.read_parquet(OOF, columns=['pair_id', 'hand_id'])
    lab = pl.read_csv(DATA + r'\development_labels.csv').select('pair_id', 'player_1', 'player_2')
    keys = keys.join(lab, on='pair_id')
    hid = keys.select('hand_id').unique()
    meta_df = (pl.scan_parquet(DATA + r'\hands.parquet').join(hid.lazy(), on='hand_id')
               .select('hand_id', 'phase', 'big_blind', 'board_cards', 'final_pot',
                       'players_dealt').collect())
    assert (meta_df['phase'] == 'development').all(), 'non-development hand in the feature build'
    assert meta_df.height == hid.height
    hand_list = meta_df['hand_id'].to_list()
    nchunk = 24
    bounds = np.linspace(0, len(hand_list), nchunk + 1).astype(int)
    out_rows = []
    for c in range(nchunk):
        part = pl.DataFrame({'hand_id': hand_list[bounds[c]:bounds[c + 1]]})
        m = meta_df.join(part, on='hand_id')
        meta = {r['hand_id']: {'bb': r['big_blind'], 'board': (r['board_cards'] or '').split(),
                               'final_pot': r['final_pot'], 'players_dealt': r['players_dealt']}
                for r in m.iter_rows(named=True)}
        seats = (pl.scan_parquet(DATA + r'\seats.parquet').join(part.lazy(), on='hand_id')
                 .select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2', 'total_contribution',
                         'net_chips', 'folded', 'went_to_showdown', 'won_share').collect())
        acts = (pl.scan_parquet(DATA + r'\actions.parquet').join(part.lazy(), on='hand_id')
                .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount',
                        'amount_to', 'pot_before', 'to_call').collect().sort('hand_id', 'action_no'))
        strength, final = hand_tables(seats, meta)
        srows, arows = {}, {}
        for r in seats.iter_rows(named=True):
            srows.setdefault(r['hand_id'], {})[r['player_id']] = r
        for r in acts.iter_rows(named=True):
            arows.setdefault(r['hand_id'], []).append(r)
        sub = keys.join(part, on='hand_id')
        for r in sub.iter_rows(named=True):
            h = r['hand_id']
            if h not in arows:
                continue
            f = pairhand_features(r['pair_id'], h, r['player_1'], r['player_2'], meta[h],
                                  srows[h], arows[h], strength[h], final[h])
            out_rows.append((r['pair_id'], h, *[f[k] for k in FEATURES]))
        print(f'chunk {c + 1}/{nchunk}: {len(out_rows)} rows', flush=True)
    df = pl.DataFrame(out_rows, schema=['pair_id', 'hand_id'] + FEATURES, orient='row')
    df = df.with_columns(pl.col(FEATURES).cast(pl.Float32))
    df.write_parquet(OUT + r'\step237_features.parquet')
    print('wrote', df.shape)
    print(df.select(FEATURES).describe())


if __name__ == '__main__':
    main()
