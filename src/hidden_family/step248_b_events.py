"""STEP248 Part B: the partner-specific EVENT EXCESS instrument.

Idea.  An honest player's behaviour does not depend on WHICH opponent is in the hand.  A
colluder does some specific thing far more often with one partner than with his 28 other
table-mates, in a handful of hands.  Means over ~77 shared hands dilute ~10 manipulated hands
(which is why OBS 14.14's ca_ means promoted 82 pairs for a NULL on the board); COUNTS of rare
directed events do not.

For every ORDERED pair (a, b) dealt into the same hand the engine counts, per event e, a
numerator C[a,b,e] and the matching OPPORTUNITY D[a,b,d(e)].  The expectation is leave-
partner-out inside the same phase and the same table:

    p_other(a, e) = (sum_{c != b} C[a,c,e] + .5) / (sum_{c != b} D[a,c,d(e)] + 1)

and the cell's evidence is the one-sided binomial upper tail P(Binom(D[a,b,d(e)], p_other) >= C).
Every event is defined for ANY ordered pair of players in a hand, so the same event exists for a
member with each of his 29 table-mates -- that is what makes the leave-partner-out baseline a
per-PLAYER control and removes the host's decoys (tilt, weak play, streaks), which are odd with
EVERYONE (OBS 15.3).

NULL.  Within each hand and independently for each actor a, a uniformly random bijection
sigma_a of a's five opponents onto themselves relabels every numerator and every opportunity
instance consistently.  This preserves the hand, the action log, a's identity and a's own total
event counts, and destroys partner specificity.  R replicates per table calibrate the pair-level
summaries including multiplicity and within-hand correlation.

Four PLACEBO events are carried that are partner-invariant by construction (the blind negative
control demanded by OBS 16.8).

One phase at a time; the phase is fixed by step248_b_prep and asserted there.
"""
import os

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from phevaluator import evaluate_cards

SCRATCH = Path(os.environ.get('STEP248_SCRATCH',
                              'C:/Users/USER/AppData/Local/Temp/claude/'
                              'C--Data-Projects-Money-Making-work-poker/'
                              'f2bb4f17-323f-4806-98b3-984023c70212/scratchpad/step248'))
N_PL = 30
BLEN = [0, 3, 4, 5]
FOLD, CHECK, CALL, BET, RAISE, ALLIN = 0, 1, 2, 3, 4, 5

# --------------------------------------------------------------------------------------
# frozen vocabulary
# --------------------------------------------------------------------------------------
DENS = ['d_shared', 'd_both_vpip', 'd_face_agg', 'd_agg_blive', 'd_bet_blive', 'd_act_blive',
        'd_both_flop', 'd_both_river', 'd_both_sd', 'd_hu', 'd_post_blive', 'd_pf_blive',
        'd_third']
DENI = {n: i for i, n in enumerate(DENS)}

EVENTS = [
    # ---- response to the partner's aggression (denominator: a decides facing b's aggression)
    ('fold_b', 'd_face_agg'), ('fold_b_W', 'd_face_agg'), ('fold_b_M', 'd_face_agg'),
    ('fold_b_S', 'd_face_agg'), ('fold_b_cheap', 'd_face_agg'), ('fold_b_mid', 'd_face_agg'),
    ('fold_b_exp', 'd_face_agg'), ('fold_b_pf', 'd_face_agg'), ('fold_b_fl', 'd_face_agg'),
    ('fold_b_tu', 'd_face_agg'), ('fold_b_ri', 'd_face_agg'), ('fold_b_hu', 'd_face_agg'),
    ('fold_b_S_hu', 'd_face_agg'), ('fold_b_cheap_S', 'd_face_agg'),
    ('fold_b_lastlive', 'd_face_agg'),
    ('call_b', 'd_face_agg'), ('call_b_W', 'd_face_agg'), ('call_b_M', 'd_face_agg'),
    ('call_b_S', 'd_face_agg'), ('call_b_hu', 'd_face_agg'), ('call_b_exp', 'd_face_agg'),
    ('raise_b', 'd_face_agg'), ('raise_b_W', 'd_face_agg'), ('raise_b_M', 'd_face_agg'),
    ('raise_b_S', 'd_face_agg'), ('raise_b_hu', 'd_face_agg'), ('raise_b_pf', 'd_face_agg'),
    ('raise_b_post', 'd_face_agg'), ('allin_vs_b', 'd_face_agg'), ('nonfold_b_W', 'd_face_agg'),
    # ---- aggression while the partner is live (denominator: a's aggressive actions, b live)
    ('agg_b_notacted', 'd_agg_blive'), ('agg_b_notacted_W', 'd_agg_blive'),
    ('agg_b_notacted_M', 'd_agg_blive'), ('agg_b_notacted_S', 'd_agg_blive'),
    ('agg_b_acted', 'd_agg_blive'), ('agg_b_W', 'd_agg_blive'), ('agg_b_M', 'd_agg_blive'),
    ('agg_b_S', 'd_agg_blive'), ('agg_b_hu', 'd_agg_blive'), ('agg_b_pf', 'd_agg_blive'),
    ('agg_b_post', 'd_agg_blive'), ('threebet_b_notacted', 'd_agg_blive'),
    ('checkraise_b', 'd_agg_blive'), ('donk_b', 'd_agg_blive'), ('agg_b_3rdlive', 'd_agg_blive'),
    # ---- BET SIZING while the partner is live (no feature in production reads this)
    ('sz_min_b', 'd_agg_blive'), ('sz_tiny_b', 'd_agg_blive'), ('sz_small_b', 'd_agg_blive'),
    ('sz_half_b', 'd_agg_blive'), ('sz_big_b', 'd_agg_blive'), ('sz_pot_b', 'd_agg_blive'),
    ('sz_over_b', 'd_agg_blive'), ('sz_huge_b', 'd_agg_blive'), ('sz_tiny_S_b', 'd_agg_blive'),
    ('sz_over_W_b', 'd_agg_blive'), ('sz_min_S_b', 'd_agg_blive'), ('sz_huge_S_b', 'd_agg_blive'),
    ('sz_allin_b', 'd_agg_blive'), ('szbb_1_b', 'd_agg_blive'), ('szbb_2_b', 'd_agg_blive'),
    ('szbb_3_b', 'd_agg_blive'), ('szbb_5_b', 'd_agg_blive'), ('szbb_9_b', 'd_agg_blive'),
    ('szbb_19_b', 'd_agg_blive'), ('szbb_20p_b', 'd_agg_blive'), ('sz_odd_b', 'd_agg_blive'),
    ('sz_round10_b', 'd_agg_blive'), ('sz_match_b', 'd_agg_blive'),
    ('sz_repeat_self_b', 'd_agg_blive'),
    # ---- preflop-specific (denominator: a's preflop decisions while b live)
    ('limp_behind_b', 'd_pf_blive'), ('coldcall_b_raise', 'd_pf_blive'),
    ('coldcall_b_raise_W', 'd_pf_blive'), ('iso_raise_b_limp', 'd_pf_blive'),
    ('squeeze_after_b_call', 'd_pf_blive'), ('fold_pf_b_live', 'd_pf_blive'),
    ('limp_b_live', 'd_pf_blive'), ('openraise_b_live', 'd_pf_blive'),
    # ---- postflop passivity (denominator: a's postflop decisions while b live)
    ('check_S_b', 'd_post_blive'), ('check_M_b', 'd_post_blive'), ('check_S_hu_b', 'd_post_blive'),
    ('checkback_S_b', 'd_post_blive'), ('call_W_b', 'd_post_blive'), ('call_W_hu_b', 'd_post_blive'),
    ('bet_W_b', 'd_post_blive'), ('fold_S_b', 'd_post_blive'),
    # ---- hand level, both live at the flop
    ('both_agg_same_street', 'd_both_flop'), ('both_agg_X_folds', 'd_both_flop'),
    ('both_agg_X_calls', 'd_both_flop'), ('both_agg_X_loses', 'd_both_flop'),
    ('a_never_agg_post_b', 'd_both_flop'), ('a_never_agg_post_S_b', 'd_both_flop'),
    ('a_folds_post_b', 'd_both_flop'), ('ab_only_vpip', 'd_both_flop'),
    ('allin_a_b_live', 'd_both_flop'), ('allin_a_b_calls', 'd_both_flop'),
    ('allin_a_b_folds', 'd_both_flop'),
    # ---- heads-up with the partner
    ('hu_checked_down', 'd_hu'), ('hu_a_wins', 'd_hu'), ('hu_a_wins_big', 'd_hu'),
    ('hu_bigpot', 'd_hu'), ('hu_a_folds', 'd_hu'),
    # ---- third party present
    ('third_loses_20bb', 'd_third'), ('third_loses_50bb', 'd_third'),
    ('third_loses_ab_sd', 'd_third'), ('squeeze_pattern', 'd_both_vpip'),
    ('sandwich_X', 'd_both_vpip'), ('sandwich_X_folds', 'd_both_vpip'),
    # ---- showdown
    ('sd_a_wins', 'd_both_sd'), ('sd_b_wins', 'd_both_sd'), ('sd_b_wins_a_S', 'd_both_sd'),
    ('sd_a_shows_W', 'd_both_sd'), ('sd_a_wins_big', 'd_both_sd'),
    # ---- chip flow buckets
    ('flow_a_lose10_b_win', 'd_shared'), ('flow_a_lose30_b_win', 'd_shared'),
    ('flow_a_lose60_b_win', 'd_shared'), ('flow_a_win10_b_lose', 'd_shared'),
    ('flow_a_win30_b_lose', 'd_shared'), ('flow_a_win60_b_lose', 'd_shared'),
    ('both_win_hand', 'd_shared'),
    # ---- river
    ('riv_a_bets_b', 'd_both_river'), ('riv_a_checks_S_b', 'd_both_river'),
    ('riv_a_folds_b', 'd_both_river'),
    # ---- PLACEBOS: partner-invariant by construction (blind negative control)
    ('plc_even_hand', 'd_shared'), ('plc_b_button', 'd_shared'), ('plc_a_utg', 'd_shared'),
    ('plc_a_bb', 'd_shared'),
]
NAMES = [e[0] for e in EVENTS]
EVI = {n: i for i, n in enumerate(NAMES)}
NN = len(EVENTS)
ND = len(DENS)
EDEN = np.array([DENI[d] for _, d in EVENTS], np.int32)
PLACEBO = np.array([i for i, n in enumerate(NAMES) if n.startswith('plc_')], np.int32)
NCODE = NN + ND


def preflop_bucket(c1, c2):
    """Chen-formula buckets: 0 weak (<5), 1 medium (5-7.5), 2 strong (>=8)."""
    RV = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10,
          'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    r1, r2 = RV[c1[0]], RV[c2[0]]
    hi, lo = max(r1, r2), min(r1, r2)
    pts = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}.get(hi, hi / 2.0)
    if r1 == r2:
        pts = max(pts * 2, 5.0)
    if c1[1] == c2[1]:
        pts += 2
    gap = hi - lo - 1
    if r1 != r2:
        pts -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
        if gap <= 1 and hi < 12:
            pts += 1
    return 2 if pts >= 8 else (1 if pts >= 5 else 0)


def made_bucket(rank):
    """phevaluator rank -> 0 high card, 1 one pair, 2 two pair or better."""
    return 0 if rank >= 6186 else (1 if rank >= 3326 else 2)


def replay(hloc, button, bb, board, final_pot, seat, ploc, hc1, hc2, net, folded, sd,
           contrib, acts, out):
    """Emit packed (hloc, a_local, b_mask, code) ints for one hand into list `out`."""
    n = 6
    ap = out.append
    base_h = hloc << 17
    ALL = 63

    def em(a, mask, code):
        if mask:
            ap(base_h | (a << 14) | (mask << 8) | code)

    # events whose denominator is counted once per HAND but that could fire on several
    # actions (sandwich / squeeze / river).  em1 emits each (a, b, code) at most once, so the
    # numerator can never exceed its opportunity.
    once = {}

    def em1(a, mask, code):
        k = (a << 8) | code
        seen = once.get(k, 0)
        m = mask & ~seen
        if m:
            once[k] = seen | m
            ap(base_h | (a << 14) | (m << 8) | code)

    pfb = [preflop_bucket(hc1[l], hc2[l]) for l in range(n)]
    cache = {}

    def stg(l, s):
        if s == 0:
            return pfb[l]
        if len(board) < 3:
            return pfb[l]
        k = l * 4 + s
        v = cache.get(k)
        if v is None:
            v = made_bucket(evaluate_cards(hc1[l], hc2[l], *board[:BLEN[s]]))
            cache[k] = v
        return v

    posto = [(seat[l] - button - 1) % 6 for l in range(n)]     # postflop order, 0 acts first
    bbseat = (button + 2) % 6
    utgseat = (button + 3) % 6

    live = [True] * n
    acted = [False] * n
    vpip = [False] * n
    live_at = [[True] * n, [False] * n, [False] * n, [False] * n]
    aggmask = [0] * n            # bitmask of postflop streets where l bet/raised
    aggpf = [False] * n
    lastamt = [0] * n            # last bet/raise `amount` by l in this hand
    lastsz = [0] * n
    folded_post = [False] * n
    allin_agg = [False] * n
    street = -1
    n_live = n
    last_agg_vol = -1
    pf_raise_seen = False
    pf_limpers = 0
    limpmask = 0
    callmask = 0
    pfaggr = -1
    seq = []                     # (local, is_agg) for the current street
    dealt_mask = ALL

    # ---------------- denominators that need no replay ---------------------------------
    for l in range(n):
        em(l, ALL ^ (1 << l), NN + DENI['d_shared'])
    if hloc % 2 == 0:
        for l in range(n):
            em(l, ALL ^ (1 << l), EVI['plc_even_hand'])
    for l in range(n):
        if seat[l] == bbseat:
            em(l, ALL ^ (1 << l), EVI['plc_a_bb'])
        if seat[l] == utgseat:
            em(l, ALL ^ (1 << l), EVI['plc_a_utg'])
    bl = [l for l in range(n) if seat[l] == button]
    if bl:
        em0 = 1 << bl[0]
        for l in range(n):
            if l != bl[0]:
                em(l, em0, EVI['plc_b_button'])

    # ---------------- replay -----------------------------------------------------------
    for (a, st, kd, amt, amt_to, pot_b, to_call) in acts:
        if st != street:
            street = st
            acted = [False] * n
            last_agg_vol = -1
            seq = []
            if st >= 1:
                for l in range(n):
                    live_at[st][l] = live[l]
        if not live[a]:
            continue
        lmask = 0
        namask = 0
        for x in range(n):
            if x != a and live[x]:
                lmask |= 1 << x
                if not acted[x]:
                    namask |= 1 << x
        hu = (n_live == 2)
        s = stg(a, st)
        agg = kd in (BET, RAISE) or (kd == ALLIN and amt > to_call)
        pot_b = pot_b if pot_b > 0 else 1

        em(a, lmask, NN + DENI['d_act_blive'])
        if st == 0:
            em(a, lmask, NN + DENI['d_pf_blive'])
        else:
            em(a, lmask, NN + DENI['d_post_blive'])
        fmask = (1 << last_agg_vol) if (to_call > 0 and last_agg_vol >= 0
                                        and last_agg_vol != a and live[last_agg_vol]) else 0
        if fmask:
            em(a, fmask, NN + DENI['d_face_agg'])
            price = to_call / pot_b
            if kd == FOLD:
                em(a, fmask, EVI['fold_b'])
                em(a, fmask, EVI[('fold_b_W', 'fold_b_M', 'fold_b_S')[s]])
                em(a, fmask, EVI['fold_b_cheap'] if price < .35 else
                   (EVI['fold_b_mid'] if price <= 1.0 else EVI['fold_b_exp']))
                em(a, fmask, EVI[('fold_b_pf', 'fold_b_fl', 'fold_b_tu', 'fold_b_ri')[st]])
                if hu:
                    em(a, fmask, EVI['fold_b_hu'])
                    em(a, fmask, EVI['fold_b_lastlive'])
                    if s == 2:
                        em(a, fmask, EVI['fold_b_S_hu'])
                if price < .35 and s == 2:
                    em(a, fmask, EVI['fold_b_cheap_S'])
            else:
                if s == 0:
                    em(a, fmask, EVI['nonfold_b_W'])
                if kd == CALL or (kd == ALLIN and amt <= to_call):
                    em(a, fmask, EVI['call_b'])
                    em(a, fmask, EVI[('call_b_W', 'call_b_M', 'call_b_S')[s]])
                    if hu:
                        em(a, fmask, EVI['call_b_hu'])
                    if price > 1.0:
                        em(a, fmask, EVI['call_b_exp'])
                elif agg:
                    em(a, fmask, EVI['raise_b'])
                    em(a, fmask, EVI[('raise_b_W', 'raise_b_M', 'raise_b_S')[s]])
                    if hu:
                        em(a, fmask, EVI['raise_b_hu'])
                    em(a, fmask, EVI['raise_b_pf'] if st == 0 else EVI['raise_b_post'])
                    if kd == ALLIN:
                        em(a, fmask, EVI['allin_vs_b'])

        if agg:
            em(a, lmask, NN + DENI['d_agg_blive'])
            if to_call == 0:
                em(a, lmask, NN + DENI['d_bet_blive'])
            em(a, namask, EVI['agg_b_notacted'])
            em(a, namask, EVI[('agg_b_notacted_W', 'agg_b_notacted_M',
                               'agg_b_notacted_S')[s]])
            em(a, lmask ^ namask, EVI['agg_b_acted'])
            em(a, lmask, EVI[('agg_b_W', 'agg_b_M', 'agg_b_S')[s]])
            if hu:
                em(a, lmask, EVI['agg_b_hu'])
            em(a, lmask, EVI['agg_b_pf'] if st == 0 else EVI['agg_b_post'])
            if st == 0 and to_call > 0:
                em(a, namask, EVI['threebet_b_notacted'])
            if n_live >= 3:
                em(a, lmask, EVI['agg_b_3rdlive'])
            if st >= 1 and any(x[0] == a and not x[1] for x in seq):
                em(a, lmask, EVI['checkraise_b'])
            if st == 1 and to_call == 0 and pfaggr >= 0 and pfaggr != a and \
                    posto[a] < posto[pfaggr]:
                em(a, lmask, EVI['donk_b'])
            # ---- bet sizing ------------------------------------------------------------
            ratio = (amt / pot_b) if to_call == 0 else (amt_to / pot_b)
            minsz = (amt <= bb) if to_call == 0 else (amt <= to_call + lastsz[a] + bb)
            if minsz:
                em(a, lmask, EVI['sz_min_b'])
                if s == 2:
                    em(a, lmask, EVI['sz_min_S_b'])
            if ratio < .25:
                em(a, lmask, EVI['sz_tiny_b'])
                if s == 2:
                    em(a, lmask, EVI['sz_tiny_S_b'])
            elif ratio < .42:
                em(a, lmask, EVI['sz_small_b'])
            elif ratio < .58:
                em(a, lmask, EVI['sz_half_b'])
            elif ratio < .85:
                em(a, lmask, EVI['sz_big_b'])
            elif ratio < 1.15:
                em(a, lmask, EVI['sz_pot_b'])
            elif ratio < 2.0:
                em(a, lmask, EVI['sz_over_b'])
                if s == 0:
                    em(a, lmask, EVI['sz_over_W_b'])
            else:
                em(a, lmask, EVI['sz_huge_b'])
                if s == 0:
                    em(a, lmask, EVI['sz_over_W_b'])
                if s == 2:
                    em(a, lmask, EVI['sz_huge_S_b'])
            if kd == ALLIN:
                em(a, lmask, EVI['sz_allin_b'])
            tb = amt_to / bb
            em(a, lmask, EVI['szbb_1_b'] if tb <= 1.5 else
               (EVI['szbb_2_b'] if tb <= 2.5 else
                (EVI['szbb_3_b'] if tb <= 3.5 else
                 (EVI['szbb_5_b'] if tb <= 5.5 else
                  (EVI['szbb_9_b'] if tb <= 9.5 else
                   (EVI['szbb_19_b'] if tb <= 19.5 else EVI['szbb_20p_b']))))))
            if amt % bb:
                em(a, lmask, EVI['sz_odd_b'])
            if amt_to % (10 * bb) == 0:
                em(a, lmask, EVI['sz_round10_b'])
            mm = 0
            for x in range(n):
                if x != a and lastamt[x] and lastamt[x] == amt:
                    mm |= 1 << x
            em(a, mm & lmask, EVI['sz_match_b'])
            if lastamt[a] == amt:
                em(a, lmask, EVI['sz_repeat_self_b'])
            lastamt[a] = amt
            lastsz[a] = amt - to_call
            if kd == ALLIN:
                allin_agg[a] = True

        # ---- preflop specifics --------------------------------------------------------
        if st == 0:
            if kd == FOLD:
                em(a, lmask, EVI['fold_pf_b_live'])
            elif agg:
                if not pf_raise_seen:
                    em(a, lmask, EVI['openraise_b_live'])
                if limpmask & lmask:
                    em(a, limpmask & lmask, EVI['iso_raise_b_limp'])
                if to_call > 0 and (callmask & lmask):
                    em(a, callmask & lmask, EVI['squeeze_after_b_call'])
            elif kd == CALL:
                if not pf_raise_seen:
                    em(a, lmask, EVI['limp_b_live'])
                    if limpmask & lmask:
                        em(a, limpmask & lmask, EVI['limp_behind_b'])
                elif not vpip[a]:
                    rm = (1 << pfaggr) if pfaggr >= 0 and pfaggr != a else 0
                    em(a, rm & lmask, EVI['coldcall_b_raise'])
                    if s == 0:
                        em(a, rm & lmask, EVI['coldcall_b_raise_W'])
        else:
            # ---- postflop passivity ----------------------------------------------------
            if kd == CHECK:
                if s == 2:
                    em(a, lmask, EVI['check_S_b'])
                    if hu:
                        em(a, lmask, EVI['check_S_hu_b'])
                    if namask == 0:
                        em(a, lmask, EVI['checkback_S_b'])
                elif s == 1:
                    em(a, lmask, EVI['check_M_b'])
            elif kd == CALL and s == 0:
                em(a, lmask, EVI['call_W_b'])
                if hu:
                    em(a, lmask, EVI['call_W_hu_b'])
            elif agg and s == 0:
                em(a, lmask, EVI['bet_W_b'])
            elif kd == FOLD and s == 2:
                em(a, lmask, EVI['fold_S_b'])
            if kd == FOLD:
                em(a, lmask, EVI['a_folds_post_b'])
                folded_post[a] = True
            if st == 3:
                if agg:
                    em1(a, lmask, EVI['riv_a_bets_b'])
                elif kd == CHECK and s == 2:
                    em1(a, lmask, EVI['riv_a_checks_S_b'])
                elif kd == FOLD:
                    em1(a, lmask, EVI['riv_a_folds_b'])

        # ---- state update -------------------------------------------------------------
        seq.append((a, agg))
        acted[a] = True
        if kd == FOLD:
            live[a] = False
            n_live -= 1
        else:
            if amt > 0:
                vpip[a] = True
            if st == 0:
                if agg:
                    pf_raise_seen = True
                    pfaggr = a
                    aggpf[a] = True
                    limpmask = 0
                    callmask = 0
                elif kd == CALL:
                    if not pf_raise_seen:
                        limpmask |= 1 << a
                    else:
                        callmask |= 1 << a
            elif agg:
                aggmask[a] |= 1 << st
        if agg:
            last_agg_vol = a
            for x in range(n):
                if x != a and live[x]:
                    acted[x] = False
        # sandwich / squeeze detection inside the street
        if agg and len(seq) >= 3:
            pri = -1
            for i in range(len(seq) - 2, -1, -1):
                if seq[i][1] and seq[i][0] != a:
                    pri = i
                    break
            if pri >= 0:
                mid = {seq[j][0] for j in range(pri + 1, len(seq) - 1)} - {a, seq[pri][0]}
                if mid:
                    u = seq[pri][0]
                    em1(a, 1 << u, EVI['sandwich_X'])
                    em1(u, 1 << a, EVI['sandwich_X'])
                    if to_call > 0:
                        em1(a, 1 << u, EVI['squeeze_pattern'])
                        em1(u, 1 << a, EVI['squeeze_pattern'])
                    if any(not live[m] for m in mid):
                        em1(a, 1 << u, EVI['sandwich_X_folds'])
                        em1(u, 1 << a, EVI['sandwich_X_folds'])

    # ---------------- hand-level events -------------------------------------------------
    flmask = 0
    for l in range(n):
        if live_at[1][l]:
            flmask |= 1 << l
    rimask = 0
    for l in range(n):
        if live_at[3][l]:
            rimask |= 1 << l
    sdmask = 0
    vpmask = 0
    for l in range(n):
        if sd[l]:
            sdmask |= 1 << l
        if vpip[l]:
            vpmask |= 1 << l
    nfl = bin(flmask).count('1')
    third_loses = [0, 0]
    for l in range(n):
        if net[l] <= -20 * bb:
            third_loses[0] |= 1 << l
        if net[l] <= -50 * bb:
            third_loses[1] |= 1 << l

    for a in range(n):
        ma = 1 << a
        oth = ALL ^ ma
        if flmask & ma:
            fo = flmask ^ ma
            em(a, fo, NN + DENI['d_both_flop'])
            if nfl == 2:
                em(a, fo, NN + DENI['d_hu'])
            if nfl >= 3:
                em(a, fo, NN + DENI['d_third'])
                t20 = third_loses[0] & ~(ma | fo)
                if t20 & flmask:
                    em(a, fo, EVI['third_loses_20bb'])
                    if third_loses[1] & flmask & ~(ma | fo):
                        em(a, fo, EVI['third_loses_50bb'])
                    if sd[a]:
                        em(a, fo & sdmask, EVI['third_loses_ab_sd'])
            if aggmask[a]:
                mm = 0
                for b in range(n):
                    if b != a and (flmask >> b) & 1 and (aggmask[b] & aggmask[a]):
                        mm |= 1 << b
                if mm:
                    em(a, mm, EVI['both_agg_same_street'])
                    fx = any((not live[x]) and x != a and not ((mm >> x) & 1) and
                             ((flmask >> x) & 1) for x in range(n))
                    if fx:
                        em(a, mm, EVI['both_agg_X_folds'])
                    cx = any(live[x] and x != a and not ((mm >> x) & 1) and
                             ((flmask >> x) & 1) for x in range(n))
                    if cx:
                        em(a, mm, EVI['both_agg_X_calls'])
                    if third_loses[0] & flmask & ~(ma | mm):
                        em(a, mm, EVI['both_agg_X_loses'])
            else:
                em(a, fo, EVI['a_never_agg_post_b'])
                laststreet = 3 if live_at[3][a] else (2 if live_at[2][a] else 1)
                if stg(a, laststreet) == 2:
                    em(a, fo, EVI['a_never_agg_post_S_b'])
            if vpmask == (ma | (vpmask & fo)) and bin(vpmask).count('1') == 2:
                em(a, vpmask ^ ma, EVI['ab_only_vpip'])
            if allin_agg[a]:
                em(a, fo, EVI['allin_a_b_live'])
                for b in range(n):
                    if b != a and (flmask >> b) & 1:
                        em(a, 1 << b, EVI['allin_a_b_calls'] if live[b]
                           else EVI['allin_a_b_folds'])
            if nfl == 2:
                fb = fo
                if aggmask[a] == 0 and all(aggmask[x] == 0 for x in range(n) if (fo >> x) & 1):
                    em(a, fb, EVI['hu_checked_down'])
                if net[a] > 0:
                    em(a, fb, EVI['hu_a_wins'])
                    if net[a] >= 20 * bb:
                        em(a, fb, EVI['hu_a_wins_big'])
                if final_pot >= 30 * bb:
                    em(a, fb, EVI['hu_bigpot'])
                if folded_post[a]:
                    em(a, fb, EVI['hu_a_folds'])
        if rimask & ma:
            em(a, rimask ^ ma, NN + DENI['d_both_river'])
        if sdmask & ma:
            so = sdmask ^ ma
            em(a, so, NN + DENI['d_both_sd'])
            if net[a] > 0:
                em(a, so, EVI['sd_a_wins'])
                if net[a] >= 20 * bb:
                    em(a, so, EVI['sd_a_wins_big'])
            else:
                mm = 0
                for b in range(n):
                    if b != a and (so >> b) & 1 and net[b] > 0:
                        mm |= 1 << b
                em(a, mm, EVI['sd_b_wins'])
                if stg(a, 3 if live_at[3][a] else 2) == 2:
                    em(a, mm, EVI['sd_b_wins_a_S'])
            if stg(a, 3 if live_at[3][a] else 2) == 0:
                em(a, so, EVI['sd_a_shows_W'])
        if vpmask & ma:
            em(a, vpmask ^ ma, NN + DENI['d_both_vpip'])
        if net[a] < 0:
            mm = 0
            for b in range(n):
                if b != a and net[b] > 0:
                    mm |= 1 << b
            if net[a] <= -10 * bb:
                em(a, mm, EVI['flow_a_lose10_b_win'])
            if net[a] <= -30 * bb:
                em(a, mm, EVI['flow_a_lose30_b_win'])
            if net[a] <= -60 * bb:
                em(a, mm, EVI['flow_a_lose60_b_win'])
        elif net[a] > 0:
            mm = 0
            for b in range(n):
                if b != a and net[b] < 0:
                    mm |= 1 << b
            wm = 0
            for b in range(n):
                if b != a and net[b] > 0:
                    wm |= 1 << b
            em(a, wm, EVI['both_win_hand'])
            if net[a] >= 10 * bb:
                em(a, mm, EVI['flow_a_win10_b_lose'])
            if net[a] >= 30 * bb:
                em(a, mm, EVI['flow_a_win30_b_lose'])
            if net[a] >= 60 * bb:
                em(a, mm, EVI['flow_a_win60_b_lose'])
