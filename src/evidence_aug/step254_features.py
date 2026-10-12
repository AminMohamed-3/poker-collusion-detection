"""STEP254: the MECHANICAL pair-relative decomposition of the raw columns into per-hand variables.

Development phase only (asserted). One row per (pair_id, hand_id) for every hand of every labelled
pair (223,749 rows). No player ids in the output.

Five families, all enumerated mechanically - nothing here was chosen because a hypothesis said so:
  t1  ACTION TENSOR   street x action x PARTNER STATE x OUTSIDER STATE x own strength tercile,
                      plus per-decision sizing (pot fraction, raise multiple, pot odds, SPR,
                      min-raise, overbet, all-in) and first-occurrence streets.
  t2  CARDS           hole-card decomposition, board texture, made-hand class, draw flags, exact
                      heads-up comparison and enumerated equity against the PARTNER'S real cards,
                      rank among all live players per street and at each fold / call / raise.
  t3  POSITION        position vs the button, order of the two members, live outsiders between
                      them in action order, blinds, live counts.
  t4  STACKS/FLOW     stacks in bb, effective stack, cover, SPR per street  (+ the chronological
                      block built by step254_prep.py: rebuys, session age, session profit,
                      cumulative chip flow between the members over EARLIER shared hands).
  t5  OUTCOME         who won, won_share, net in bb, direct transfer, fold street and to whom,
                      showdown flags, pot, contributions.

Roles: A = the member who loses more chips over the pair's shared hands (STEP237 convention).
Emission modes per per-member quantity:
  S -> _sm (sum over the two members) and _mx (max)
  M -> _mn and _mx
  D -> _A, _B, _sm, _mx        (the partner-state-indexed core of the action tensor)
  E -> _A, _B, _mn, _mx
Sentinels: -1 = street not reached / not applicable, 9 = the event never happened.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import itertools
import time

import numpy as np
import polars as pl

import step254_eval7 as E

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
OUT = CV + r'\experiments\step254_mechanical_tensor_20260917'
OOF = CV + r'\experiments\step232_event_detector_20260917\hand_oof.parquet'
TMP = (r'C:\Users\USER\AppData\Local\Temp\claude\C--Data-Projects-Money-Making-work-poker'
       r'\f2bb4f17-323f-4806-98b3-984023c70212\scratchpad\step254_chunks')

SNAME = ('pre', 'flop', 'turn', 'river')
ANAME = ('fold', 'check', 'call', 'bet', 'raise')
PNAME = ('nodeal', 'folded', 'yettoact', 'passive', 'aggressor', 'allin')
ONAME = ('noout', 'outnoagg', 'outagg')
TNAME = ('lo', 'mid', 'hi')
SIDX = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
MODE_COLS = {'S': ('_sm', '_mx'), 'M': ('_mn', '_mx'), 'D': ('_A', '_B', '_sm', '_mx'),
             'E': ('_A', '_B', '_mn', '_mx')}

PM: list[tuple[str, str]] = []      # per-member (name, mode)
DOC: dict[str, str] = {}
HL: list[str] = []                  # hand-level names


def reg(name: str, mode: str, doc: str) -> int:
    i = len(PM)
    PM.append((name, mode))
    DOC[name] = doc
    return i


def regh(name: str, doc: str) -> int:
    i = len(HL)
    HL.append(name)
    DOC[name] = doc
    return i


# ---------------------------------------------------------------- layout: family 1 action tensor
B_A = len(PM)
for s in range(4):
    for a in range(5):
        reg(f't1_a_{SNAME[s]}_{ANAME[a]}', 'S', f'(1) decisions: street {SNAME[s]} x action {ANAME[a]}')
B_AP = len(PM)
for a in range(5):
    for p in range(6):
        reg(f't1_ap_{ANAME[a]}_{PNAME[p]}', 'D', f'(1) decisions: action {ANAME[a]} while the partner is {PNAME[p]}')
B_SP = len(PM)
for s in range(4):
    for q in range(2):
        reg(f't1_sp_{SNAME[s]}_{q}', 'S', f'(1) decisions on {SNAME[s]} with partner-live-and-yet-to-act = {q}')
B_AO = len(PM)
for a in range(5):
    for o in range(3):
        reg(f't1_ao_{ANAME[a]}_{ONAME[o]}', 'S', f'(1) decisions: action {ANAME[a]} x outsider state {ONAME[o]}')
B_PT = len(PM)
for p in range(6):
    for t in range(3):
        reg(f't1_pt_{PNAME[p]}_{TNAME[t]}', 'S', f'(1) decisions: partner {PNAME[p]} x own strength tercile {TNAME[t]}')
B_AT = len(PM)
for a in range(5):
    for t in range(3):
        reg(f't1_at_{ANAME[a]}_{TNAME[t]}', 'S', f'(1) decisions: action {ANAME[a]} x own strength tercile {TNAME[t]}')
B_PO = len(PM)
for p in range(6):
    for o in range(3):
        reg(f't1_po_{PNAME[p]}_{ONAME[o]}', 'S', f'(1) decisions: partner {PNAME[p]} x outsider state {ONAME[o]}')
B_WGS = len(PM)
for p in range(6):
    reg(f't1_wgs_{PNAME[p]}', 'D', f'(1) sum of (1 - own strength) over AGGRESSIVE actions with the partner {PNAME[p]}')
B_WGM = len(PM)
for p in range(6):
    reg(f't1_wgm_{PNAME[p]}', 'D', f'(1) max of (1 - own strength) over AGGRESSIVE actions with the partner {PNAME[p]}')
B_WGSS = len(PM)
for s in range(4):
    reg(f't1_wgss_{SNAME[s]}', 'D', f'(1) sum of (1 - own strength) over aggressive actions on {SNAME[s]}')
B_BB = len(PM)
for s in range(4):
    for a in (2, 3, 4):
        reg(f't1_bb_{SNAME[s]}_{ANAME[a]}', 'S', f'(1) chips in bb put in by {ANAME[a]} on {SNAME[s]}')
B_BBS = len(PM)
for p in range(6):
    reg(f't1_bbs_{PNAME[p]}', 'D', f'(1) chips in bb put in voluntarily while the partner is {PNAME[p]}')
B_BBM = len(PM)
for p in range(6):
    reg(f't1_bbm_{PNAME[p]}', 'D', f'(1) largest single bb amount put in while the partner is {PNAME[p]}')
B_PF = len(PM)
PFB = ('le33', 'le60', 'le90', 'le130', 'gt130')
for k in range(5):
    for q in range(2):
        reg(f't1_pf_{PFB[k]}_{q}', 'S', f'(1) bet/raise sized {PFB[k]}% of pot, partner-yet-to-act = {q}')
B_RM = len(PM)
RMB = ('le2.2x', 'le3x', 'le4x', 'gt4x')
for k in range(4):
    for q in range(2):
        reg(f't1_rm_{RMB[k]}_{q}', 'S', f'(1) raise to {RMB[k]} the call, partner-yet-to-act = {q}')
B_OD = len(PM)
ODB = ('le15', 'le25', 'le35', 'gt35')
for k in range(4):
    for q in range(2):
        reg(f't1_od_{ODB[k]}_{q}', 'S', f'(1) call/fold decision at pot odds {ODB[k]}%, facing the partner = {q}')
B_SPR = len(PM)
SPRB = ('le1', 'le3', 'le8', 'gt8')
for k in range(4):
    for q in range(2):
        reg(f't1_spr_{SPRB[k]}_{q}', 'S', f'(1) decision at SPR {SPRB[k]} (stack_before/pot_before), postflop = {q}')
B_FL = len(PM)
FLB = ('minraise', 'overbet', 'allin')
for k in range(3):
    for q in range(2):
        reg(f't1_fl_{FLB[k]}_{q}', 'S', f'(1) {FLB[k]} decisions, facing the partner = {q}')
B_FSP = len(PM)
for p in range(6):
    reg(f't1_fsP_{PNAME[p]}', 'E', f'(1) first street index of a decision with the partner {PNAME[p]} (9 = never)')
B_FSA = len(PM)
for a in range(5):
    reg(f't1_fsA_{ANAME[a]}', 'M', f'(1) first street index of a {ANAME[a]} (9 = never)')
N_T1 = len(PM)

# ---------------------------------------------------------------- layout: family 2 cards (member)
B_HOLE = len(PM)
for nm, d in (('hi', 'high rank 2-14'), ('lo', 'low rank 2-14'), ('pair', 'pocket pair'),
              ('suited', 'suited'), ('gap', 'rank gap'), ('pf169', '169-class equity percentile')):
    reg(f't2_hole_{nm}', 'E', f'(2) hole cards: {d}')
B_STR = len(PM)
for s in range(1, 4):
    reg(f't2_str_{SNAME[s]}', 'E', f'(2) phevaluator strength percentile on the {SNAME[s]}')
B_CLS = len(PM)
for s in range(1, 4):
    reg(f't2_cls_{SNAME[s]}', 'M', f'(2) made-hand class 0=high card .. 8=straight flush on the {SNAME[s]}')
B_RNK = len(PM)
for s in range(1, 4):
    reg(f't2_rank_{SNAME[s]}', 'M', f'(2) rank of own strength among all LIVE players on the {SNAME[s]} (1 = best)')
B_BEST = len(PM)
for s in range(1, 4):
    reg(f't2_best_{SNAME[s]}', 'S', f'(2) holds the best hand among live players on the {SNAME[s]}')
B_DRAW = len(PM)
for s in (1, 2):
    for nm in ('fdraw', 'oesd'):
        reg(f't2_{nm}_{SNAME[s]}', 'S', f'(2) {"four-flush" if nm == "fdraw" else "open-ended straight"} draw on the {SNAME[s]}')
B_ARNK = len(PM)
for a in ('fold', 'call', 'raise'):
    reg(f't2_rank_at_{a}', 'M', f'(2) best (lowest) rank among live players held at a {a} (9 = never)')
B_ABEST = len(PM)
for a in ('fold', 'call', 'raise'):
    reg(f't2_best_at_{a}', 'S', f'(2) number of {a} decisions taken while holding the best live hand')
N_T2M = len(PM)

# ---------------------------------------------------------------- layout: family 4 (in hand)
B_SPRS = len(PM)
for s in range(4):
    reg(f't4_spr_{SNAME[s]}', 'M', f'(4) SPR (stack_before/pot_before) at the first own action on {SNAME[s]} (-1 none)')
N_T4M = len(PM)

# ---------------------------------------------------------------- layout: family 5 (per member)
B_NET = reg('t5_net_bb', 'E', '(5) net chips in bb')
B_WON = reg('t5_won_share', 'E', '(5) share of the pot won')
B_CON = reg('t5_contrib_bb', 'E', '(5) total contribution in bb')
B_FSTR = reg('t5_fold_street', 'E', '(5) street index of the fold (9 = did not fold)')
B_FTP = reg('t5_fold_to_partner', 'S', '(5) folded facing the partner as last aggressor')
B_FTO = reg('t5_fold_to_outsider', 'S', '(5) folded facing an outsider as last aggressor')
B_FTN = reg('t5_fold_to_nobody', 'S', '(5) folded with no aggressor on that street')
B_SD = reg('t5_showdown', 'S', '(5) went to showdown')
B_VP = reg('t5_vpip', 'S', '(5) voluntarily put chips in')
B_WIN = reg('t5_won', 'S', '(5) won a share of the pot')
B_BESTF = reg('t5_best_final', 'S', '(5) best final 7-card hand among the players who did not fold')
NPM = len(PM)

# ---------------------------------------------------------------- layout: hand level
H_EQ = len(HL)
for s in range(4):
    regh(f't2_eq_{SNAME[s]}_A', f"(2) A's exact equity against the PARTNER'S real cards on the {SNAME[s]} (-1 n/a)")
H_EQM = len(HL)
for s in range(4):
    regh(f't2_eq_{SNAME[s]}_mx', f'(2) max(eq, 1-eq) on the {SNAME[s]} (-1 n/a)')
H_AHEAD = len(HL)
for s in range(4):
    regh(f't2_ahead_{SNAME[s]}', f'(2) +1 A ahead of B / 0 tied / -1 behind on the {SNAME[s]} (-9 n/a)')
H_SGAP = len(HL)
for s in range(4):
    regh(f't2_strgap_{SNAME[s]}', f'(2) A strength minus B strength on the {SNAME[s]} (-9 n/a)')
H_BRD = len(HL)
for s in range(1, 4):
    for nm in ('paired', 'mono', 'twotone', 'maxrun', 'toprank'):
        regh(f't2_board_{nm}_{SNAME[s]}', f'(2) board texture on the {SNAME[s]}: {nm}')
H_NSTR = regh('t2_n_streets', '(2) number of streets played')
H_POSA = len(HL)
regh('t3_pos_A', '(3) A position vs the button 0=BTN,1=SB,2=BB,3=UTG,4=MP,5=CO')
regh('t3_pos_B', '(3) B position vs the button')
regh('t3_pos_mn', '(3) min of the two positions')
regh('t3_pos_mx', '(3) max of the two positions')
H_POS1H = len(HL)
for w in 'AB':
    for p in range(6):
        regh(f't3_pos{w}_{p}', f'(3) one-hot: {w} sits at position {p} vs the button')
H_ORD = len(HL)
regh('t3_A_first_pre', '(3) A acts before B preflop')
regh('t3_A_first_post', '(3) A acts before B postflop')
regh('t3_seatgap', '(3) circular seat distance between the members')
regh('t3_order_dist_pre', '(3) signed distance A->B in preflop action order')
H_BETW = len(HL)
for s in range(4):
    regh(f't3_between_{SNAME[s]}', f'(3) live outsiders between the two members in action order on the {SNAME[s]} (-1 n/a)')
H_BL = len(HL)
regh('t3_both_blinds', '(3) both members are in the blinds')
regh('t3_A_blind', '(3) A is in the blinds')
regh('t3_B_blind', '(3) B is in the blinds')
regh('t3_A_button', '(3) A is on the button')
regh('t3_B_button', '(3) B is on the button')
H_BEF = len(HL)
for s in range(4):
    regh(f't3_A_before_B_{SNAME[s]}', f'(3) A acts before B on the {SNAME[s]} (-1 n/a)')
H_LIVE = len(HL)
for s in range(4):
    regh(f't3_live_{SNAME[s]}', f'(3) players live at the start of the {SNAME[s]} (-1 n/a)')
H_BOTHLIVE = len(HL)
for s in range(4):
    regh(f't3_bothlive_{SNAME[s]}', f'(3) both members live at the start of the {SNAME[s]}')
H_RATIO = regh('t4_stack_ratio_AB', '(4) A starting stack / B starting stack')
H_COVER = regh('t4_A_covers_B', '(4) +1 A covers B, -1 B covers A, 0 equal')
H_EFF = regh('t4_eff_stack_bb', '(4) effective stack between the members in bb')
H_POT = regh('t5_pot_bb', '(5) final pot in bb')
H_SDN = regh('t5_players_at_showdown', '(5) players at showdown')
H_PSH = regh('t5_pair_pot_share', '(5) (both members contributions) / final pot')
H_TRU = regh('t5_transfer_bb', '(5) direct transfer between the members in bb (unsigned)')
H_TRS = regh('t5_transfer_bb_A2B', '(5) signed transfer, positive = chips move from A to B')
H_OUTW = regh('t5_outsider_won', '(5) an outsider took the pot')
H_NWIN = regh('t5_n_members_won', '(5) how many members won a share')
H_WPART = regh('t5_winner_is_partner_of_folder', '(5) a member folded and the partner finished net positive')
H_BOTHSD = regh('t5_both_showdown', '(5) both members reached showdown')
NHL = len(HL)

# ---------------------------------------------------------------- emitted column names
COLS: list[str] = []
EMIT: list[tuple[int, str]] = []    # (per-member index, suffix)
for i, (nm, mode) in enumerate(PM):
    for suf in MODE_COLS[mode]:
        COLS.append(nm + suf)
        EMIT.append((i, suf))
N_EMIT = len(COLS)
COLS += HL
NCOL = len(COLS)
# vectorised emit index arrays: column positions and the per-member slots they read
C_A = np.array([j for j, (_, s) in enumerate(EMIT) if s == '_A'], np.int64)
P_A = np.array([i for i, s in EMIT if s == '_A'], np.int64)
C_B = np.array([j for j, (_, s) in enumerate(EMIT) if s == '_B'], np.int64)
P_B = np.array([i for i, s in EMIT if s == '_B'], np.int64)
C_SM = np.array([j for j, (_, s) in enumerate(EMIT) if s == '_sm'], np.int64)
P_SM = np.array([i for i, s in EMIT if s == '_sm'], np.int64)
C_MX = np.array([j for j, (_, s) in enumerate(EMIT) if s == '_mx'], np.int64)
P_MX = np.array([i for i, s in EMIT if s == '_mx'], np.int64)
C_MN = np.array([j for j, (_, s) in enumerate(EMIT) if s == '_mn'], np.int64)
P_MN = np.array([i for i, s in EMIT if s == '_mn'], np.int64)

PAIRS45 = np.array(list(itertools.combinations(range(45), 2)), dtype=np.int64)
POP = np.array([bin(m).count('1') for m in range(8192)], dtype=np.int8)
HAS4 = np.zeros(8192, bool)
HASS = np.zeros(8192, bool)
for m in range(8192):
    HAS4[m] = any(all(m >> (i + k) & 1 for k in range(4)) for i in range(10))
    HASS[m] = any(all(m >> (i + k) & 1 for k in range(5)) for i in range(9)) or \
        ((m & 1) and (m >> 1 & 1) and (m >> 2 & 1) and (m >> 3 & 1) and (m >> 12 & 1))
CLS_CUT = np.array([10, 166, 322, 1599, 1609, 2467, 3325, 6185, 7462])


def made_class(rank: np.ndarray) -> np.ndarray:
    return (8 - np.searchsorted(CLS_CUT, rank, side='left')).astype(np.float32)


def bucket(x: float, cuts) -> int:
    for i, c in enumerate(cuts):
        if x <= c:
            return i
    return len(cuts)


def tercile(s: float) -> int:
    return 0 if s < 1 / 3 else (1 if s < 2 / 3 else 2)


def process_hand(mA, mB, acts, seat, stv, fin_best, pot_bb, bb, v, hl, button, nseat_order):
    """fill v (2 x NPM) and the per-hand slots of hl for one (pair, hand)."""
    members = (mA, mB)
    live = set(seat)
    allin: set = set()
    acted: set = set()
    prev_agg = None
    street = 0
    invested = {p: 0. for p in seat}
    live_at_start = [set(seat), None, None, None]
    fold_street = {mA: 9., mB: 9.}
    fold_kind = {mA: -1, mB: -1}
    first_act_street = {mA: set(), mB: set()}
    n_streets = 1
    for a in acts:
        s = SIDX[a['street']]
        if s != street:
            for ss in range(street + 1, s + 1):
                live_at_start[ss] = set(live)
            street = s
            prev_agg = None
            acted = set()
            invested = {p: 0. for p in seat}
            n_streets = s + 1
        p = a['player_id']
        act_raw = a['action']
        to_call = a['to_call']
        amt = a['amount']
        aggressive = act_raw in ('bet', 'raise', 'all_in') and amt > to_call
        if act_raw == 'fold':
            ai = 0
        elif act_raw == 'check':
            ai = 1
        elif not aggressive:
            ai = 2
        else:
            ai = 4 if to_call > 0 else 3
        mine = p in members
        if mine:
            k = 0 if p == mA else 1
            partner = mB if k == 0 else mA
            # partner state
            if partner not in seat:
                ps = 0
            elif partner not in live:
                ps = 1
            elif prev_agg == partner:
                ps = 4
            elif partner in allin:
                ps = 5
            elif partner in acted:
                ps = 3
            else:
                ps = 2
            outs_live = [q for q in live if q not in members]
            if not outs_live:
                os_ = 0
            elif prev_agg is not None and prev_agg not in members:
                os_ = 2
            else:
                os_ = 1
            mystr = stv[p][s]
            ter = tercile(mystr if mystr == mystr else 0.)
            pot = a['pot_before'] / bb
            amtb = amt / bb
            q2 = 1 if ps == 2 else 0
            fp = 1 if ps == 4 else 0
            v[k, B_A + s * 5 + ai] += 1
            v[k, B_AP + ai * 6 + ps] += 1
            v[k, B_SP + s * 2 + q2] += 1
            v[k, B_AO + ai * 3 + os_] += 1
            v[k, B_PT + ps * 3 + ter] += 1
            v[k, B_AT + ai * 3 + ter] += 1
            v[k, B_PO + ps * 3 + os_] += 1
            if v[k, B_FSP + ps] > s:
                v[k, B_FSP + ps] = s
            if v[k, B_FSA + ai] > s:
                v[k, B_FSA + ai] = s
            if s not in first_act_street[p]:
                first_act_street[p].add(s)
                spr = a['stack_before'] / max(a['pot_before'], 1)
                v[k, B_SPRS + s] = min(v[k, B_SPRS + s], spr) if v[k, B_SPRS + s] >= 0 else spr
            if ai in (2, 3, 4):
                v[k, B_BB + s * 3 + (ai - 2)] += amtb
                v[k, B_BBS + ps] += amtb
                v[k, B_BBM + ps] = max(v[k, B_BBM + ps], amtb)
            if aggressive:
                w = 1. - (mystr if mystr == mystr else 0.)
                v[k, B_WGS + ps] += w
                v[k, B_WGM + ps] = max(v[k, B_WGM + ps], w)
                v[k, B_WGSS + s] += w
                pfrac = amt / max(a['pot_before'], 1)
                v[k, B_PF + bucket(pfrac, (.33, .60, .90, 1.30)) * 2 + q2] += 1
                if ai == 4 and to_call > 0:
                    rmul = a['amount_to'] / to_call
                    v[k, B_RM + bucket(rmul, (2.2, 3., 4.)) * 2 + q2] += 1
                    if rmul <= 2.2:
                        v[k, B_FL + 0 * 2 + fp] += 1
                if amt > a['pot_before']:
                    v[k, B_FL + 1 * 2 + fp] += 1
            if act_raw == 'all_in':
                v[k, B_FL + 2 * 2 + fp] += 1
            if ai in (0, 2):
                odds = to_call / (a['pot_before'] + to_call) if (a['pot_before'] + to_call) > 0 else 0.
                v[k, B_OD + bucket(odds, (.15, .25, .35)) * 2 + fp] += 1
            v[k, B_SPR + bucket(a['stack_before'] / max(a['pot_before'], 1), (1., 3., 8.)) * 2
              + (1 if s > 0 else 0)] += 1
            # rank among live players
            nbetter = 0
            for q in live:
                if q != p and stv[q][s] > mystr:
                    nbetter += 1
            rk = float(nbetter + 1)
            ai2 = {0: 0, 2: 1, 3: 2, 4: 2}.get(ai)
            if ai2 is not None:
                v[k, B_ARNK + ai2] = min(v[k, B_ARNK + ai2], rk)
                if rk == 1.:
                    v[k, B_ABEST + ai2] += 1
            if ai == 0:
                fold_street[p] = float(s)
                fold_kind[p] = 0 if prev_agg == partner else (1 if prev_agg is not None else 2)
        if ai in (2, 3, 4):
            invested[p] += amt
        if aggressive:
            prev_agg = p
        if act_raw == 'all_in':
            allin.add(p)
        acted.add(p)
        if ai == 0:
            live.discard(p)
    for ss in range(street + 1, 4):
        live_at_start[ss] = None
    # ---- per member end-of-hand
    for k, p in enumerate(members):
        r = seat[p]
        v[k, B_NET] = r['net_chips'] / bb
        v[k, B_WON] = r['won_share']
        v[k, B_CON] = r['total_contribution'] / bb
        v[k, B_FSTR] = fold_street[p]
        if fold_kind[p] == 0:
            v[k, B_FTP] = 1.
        elif fold_kind[p] == 1:
            v[k, B_FTO] = 1.
        elif fold_kind[p] == 2:
            v[k, B_FTN] = 1.
        v[k, B_SD] = 1. if r['went_to_showdown'] else 0.
        v[k, B_VP] = 1. if r['total_contribution'] > bb else 0.
        v[k, B_WIN] = 1. if r['won_share'] > 0 else 0.
        v[k, B_BESTF] = 1. if (not r['folded'] and stv[p][3] >= fin_best - 1e-9) else 0.
    # ---- hand level
    for s in range(4):
        ls = live_at_start[s]
        hl[H_LIVE + s] = float(len(ls)) if ls is not None else -1.
        hl[H_BOTHLIVE + s] = 1. if (ls is not None and mA in ls and mB in ls) else 0.
        if ls is None:
            hl[H_BETW + s] = -1.
            hl[H_BEF + s] = -1.
            continue
        order = nseat_order[0 if s == 0 else 1]
        oa, ob = order[seat[mA]['seat_no']], order[seat[mB]['seat_no']]
        hl[H_BEF + s] = 1. if oa < ob else 0.
        lo, hi = (oa, ob) if oa < ob else (ob, oa)
        hl[H_BETW + s] = float(sum(1 for q in ls if q not in members
                                   and lo < order[seat[q]['seat_no']] < hi))
    hl[H_NSTR] = float(n_streets)
    posA = (seat[mA]['seat_no'] - button) % 6
    posB = (seat[mB]['seat_no'] - button) % 6
    hl[H_POSA] = float(posA)
    hl[H_POSA + 1] = float(posB)
    hl[H_POSA + 2] = float(min(posA, posB))
    hl[H_POSA + 3] = float(max(posA, posB))
    hl[H_POS1H + posA] = 1.
    hl[H_POS1H + 6 + posB] = 1.
    o0 = nseat_order[0]
    o1 = nseat_order[1]
    hl[H_ORD] = 1. if o0[seat[mA]['seat_no']] < o0[seat[mB]['seat_no']] else 0.
    hl[H_ORD + 1] = 1. if o1[seat[mA]['seat_no']] < o1[seat[mB]['seat_no']] else 0.
    d = (seat[mA]['seat_no'] - seat[mB]['seat_no']) % 6
    hl[H_ORD + 2] = float(min(d, 6 - d))
    hl[H_ORD + 3] = float(o0[seat[mA]['seat_no']] - o0[seat[mB]['seat_no']])
    hl[H_BL] = 1. if posA in (1, 2) and posB in (1, 2) else 0.
    hl[H_BL + 1] = 1. if posA in (1, 2) else 0.
    hl[H_BL + 2] = 1. if posB in (1, 2) else 0.
    hl[H_BL + 3] = 1. if posA == 0 else 0.
    hl[H_BL + 4] = 1. if posB == 0 else 0.
    sa = seat[mA]['starting_stack'] / bb
    sb = seat[mB]['starting_stack'] / bb
    hl[H_RATIO] = sa / sb if sb > 0 else 0.
    hl[H_COVER] = 1. if sa > sb else (-1. if sb > sa else 0.)
    hl[H_EFF] = min(sa, sb)
    hl[H_POT] = pot_bb
    hl[H_PSH] = ((seat[mA]['total_contribution'] + seat[mB]['total_contribution']) / bb / pot_bb
                 if pot_bb > 0 else 0.)
    nA, nB = seat[mA]['net_chips'] / bb, seat[mB]['net_chips'] / bb
    tr = 0.
    if nA < 0 < nB:
        tr = min(-nA, nB)
    elif nB < 0 < nA:
        tr = -min(-nB, nA)
    hl[H_TRU] = abs(tr)
    hl[H_TRS] = tr
    nwin = (1 if seat[mA]['won_share'] > 0 else 0) + (1 if seat[mB]['won_share'] > 0 else 0)
    hl[H_NWIN] = float(nwin)
    hl[H_OUTW] = 1. if (seat[mA]['won_share'] + seat[mB]['won_share']) < .999 else 0.
    hl[H_WPART] = 1. if ((fold_street[mA] < 9 and nB > 0) or (fold_street[mB] < 9 and nA > 0)) else 0.
    hl[H_BOTHSD] = 1. if seat[mA]['went_to_showdown'] and seat[mB]['went_to_showdown'] else 0.


def board_texture(board: np.ndarray, nb: int) -> tuple:
    r = board[:nb] >> 2
    s = board[:nb] & 3
    cnt = np.bincount(r, minlength=13)
    mask = int(sum(1 << int(x) for x in r))
    sc = np.bincount(s, minlength=4)
    run = 0
    best = 0
    for i in range(13):
        run = run + 1 if mask >> i & 1 else 0
        best = max(best, run)
    return (float((cnt > 1).any()), float(sc.max() == nb), float(sc.max() == nb - 1),
            float(best), float(r.max() + 2))


def main() -> None:
    os.makedirs(TMP, exist_ok=True)
    t_start = time.time()
    pf = np.load(OUT + r'\step254_pf169.npz')
    PCT169, EQ169 = pf['pct169'], pf['eq169x169']
    chrono = pl.read_parquet(OUT + r'\step254_chrono.parquet')
    keys = pl.read_parquet(OOF, columns=['pair_id', 'hand_id'])
    lab = pl.read_csv(DATA + r'\development_labels.csv').select('pair_id', 'player_1', 'player_2')
    keys = keys.join(lab, on='pair_id').join(chrono.select('pair_id', 'hand_id', 'p1_is_A'),
                                             on=['pair_id', 'hand_id'])
    assert keys.height == 223749, keys.height
    hid = keys.select('hand_id').unique()
    meta_df = (pl.scan_parquet(DATA + r'\hands.parquet').join(hid.lazy(), on='hand_id')
               .select('hand_id', 'phase', 'big_blind', 'board_cards', 'final_pot', 'players_dealt',
                       'players_at_showdown', 'button_seat').collect())
    assert (meta_df['phase'] == 'development').all(), 'non-development hand in the STEP254 build'
    assert meta_df.height == hid.height
    hand_list = meta_df['hand_id'].to_list()
    nchunk = 40
    bounds = np.linspace(0, len(hand_list), nchunk + 1).astype(int)
    maxchunk = int(os.environ.get('STEP254_MAXCHUNK', nchunk))
    files = []
    for c in range(min(nchunk, maxchunk)):
        t0 = time.time()
        part = pl.DataFrame({'hand_id': hand_list[bounds[c]:bounds[c + 1]]})
        m = meta_df.join(part, on='hand_id')
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
        # drw columns, in the registration order: flop fdraw, flop oesd, turn fdraw, turn oesd
        for k, nb in ((1, 3), (2, 4), (3, 5)):
            sel = np.flatnonzero(nbs >= nb)
            if not len(sel):
                continue
            cards = np.concatenate([c1[sel, None], c2[sel, None], board[si[sel], :nb]], axis=1)
            rk = E.evaluate(cards)
            sarr[sel, k] = (7463.0 - rk) / 7462.0
            clsarr[sel, k] = made_class(rk)
            if nb in (3, 4):
                mask = np.bitwise_or.reduce(np.int64(1) << (cards >> 2), axis=1)
                sc = np.stack([((cards & 3) == su).sum(axis=1) for su in range(4)], axis=1)
                base = 0 if nb == 3 else 2
                drw[sel, base] = (sc.max(axis=1) == 4).astype(np.float32)
                drw[sel, base + 1] = (HAS4[mask] & ~HASS[mask]).astype(np.float32)
        drw2 = drw
        # ---- python side tables
        srows, stv, clsv, drwv, holev = {}, {}, {}, {}, {}
        s_hand = seats['hand_id'].to_list()
        s_play = seats['player_id'].to_list()
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
        sub = keys.join(part, on='hand_id')
        n = sub.height
        Vout = np.zeros((n, NCOL), np.float32)
        rowsA = np.zeros((n, 2), np.int64)   # hole card ids of A for equity
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
            for q in seat:                       # forward-fill: strength on an unreached street
                a4 = stf[q]
                for kk in range(1, 4):
                    if a4[kk] < 0:
                        a4[kk] = a4[kk - 1]
            v = np.zeros((2, NPM), np.float32)
            v[:, B_FSP:B_FSP + 6] = 9.
            v[:, B_FSA:B_FSA + 5] = 9.
            v[:, B_ARNK:B_ARNK + 3] = 9.
            v[:, B_SPRS:B_SPRS + 4] = -1.
            hl = np.zeros(NHL, np.float32)
            button = mr['button_seat']
            o_pre = {s: (s - button - 3) % 6 for s in range(6)}
            o_post = {s: (s - button - 1) % 6 for s in range(6)}
            process_hand(mA, mB, arows.get(h, []), seat, stf, fin_best,
                         mr['final_pot'] / bb, bb, v, hl, button, (o_pre, o_post))
            # cards (per member)
            for k, p in enumerate((mA, mB)):
                a, b, cl = holev[h][p]
                r1, r2 = a >> 2, b >> 2
                v[k, B_HOLE + 0] = max(r1, r2) + 2
                v[k, B_HOLE + 1] = min(r1, r2) + 2
                v[k, B_HOLE + 2] = 1. if r1 == r2 else 0.
                v[k, B_HOLE + 3] = 1. if (a & 3) == (b & 3) else 0.
                v[k, B_HOLE + 4] = abs(r1 - r2)
                v[k, B_HOLE + 5] = PCT169[cl]
                v[k, B_STR:B_STR + 3] = np.where(np.isnan(st[p][1:]), -1., st[p][1:])
                v[k, B_CLS:B_CLS + 3] = clsv[h][p][1:]
                v[k, B_DRAW:B_DRAW + 4] = drwv[h][p]
            for k, p in enumerate((mA, mB)):
                for s in range(1, 4):
                    if np.isnan(st[p][s]):
                        v[k, B_RNK + s - 1] = -1.
                        continue
                    nb2 = sum(1 for q in seat if q != p and not seat[q]['folded']
                              and st[q][s] > st[p][s])
                    v[k, B_RNK + s - 1] = nb2 + 1
                    v[k, B_BEST + s - 1] = 1. if nb2 == 0 else 0.
            nbd = int(nboard[hidx[h]])
            for s in range(1, 4):
                if nbd >= s + 2:
                    tex = board_texture(board[hidx[h]], s + 2)
                    hl[H_BRD + (s - 1) * 5:H_BRD + s * 5] = tex
                else:
                    hl[H_BRD + (s - 1) * 5:H_BRD + s * 5] = -1.
            for s in range(4):
                a_s, b_s = st[mA][s], st[mB][s]
                if a_s != a_s or b_s != b_s:
                    hl[H_AHEAD + s] = -9.
                    hl[H_SGAP + s] = -9.
                else:
                    hl[H_AHEAD + s] = 1. if a_s > b_s else (0. if a_s == b_s else -1.)
                    hl[H_SGAP + s] = a_s - b_s
            hl[H_EQ:H_EQ + 4] = -1.
            hl[H_EQM:H_EQM + 4] = -1.
            ca1, ca2, cla = holev[h][mA]
            cb1, cb2, clb = holev[h][mB]
            hl[H_EQ] = EQ169[cla, clb]
            if nbd == 5:
                a5, b5 = st[mA][3], st[mB][3]
                hl[H_EQ + 3] = 1. if a5 > b5 else (.5 if a5 == b5 else 0.)
            rowsA[i] = (ca1, ca2)
            rowsB[i] = (cb1, cb2)
            rowh[i] = hidx[h]
            if hl[H_BOTHLIVE + 1] > 0 and nbd >= 3:
                eq_need[1].append(i)
            if hl[H_BOTHLIVE + 2] > 0 and nbd >= 4:
                eq_need[2].append(i)
            hl[H_SDN] = float(mr['players_at_showdown'])
            # emit (vectorised)
            row = Vout[i]
            row[C_A] = v[0, P_A]
            row[C_B] = v[1, P_B]
            row[C_SM] = v[0, P_SM] + v[1, P_SM]
            row[C_MX] = np.maximum(v[0, P_MX], v[1, P_MX])
            row[C_MN] = np.minimum(v[0, P_MN], v[1, P_MN])
            row[N_EMIT:] = hl
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
                    run = avail[:, PAIRS45]                      # (R, 990, 2)
                else:
                    run = avail[:, :, None]                      # (R, 44, 1)
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
                Vout[sel, N_EMIT + H_EQ + street] = eq
        for s in range(4):
            e = Vout[:, N_EMIT + H_EQ + s]
            Vout[:, N_EMIT + H_EQM + s] = np.where(e < 0, -1., np.maximum(e, 1. - e))
        df = pl.DataFrame({'pair_id': pids, 'hand_id': hids}).hstack(
            pl.DataFrame(Vout, schema=COLS))
        fn = TMP + f'\\chunk_{c:03d}.parquet'
        df.write_parquet(fn)
        files.append(fn)
        print(f'chunk {c + 1}/{nchunk}  rows {df.height}  {time.time() - t0:.1f}s  '
              f'total {time.time() - t_start:.0f}s', flush=True)
    full = pl.concat([pl.read_parquet(f) for f in files])
    if maxchunk < nchunk:
        print('partial run, stopping before the prune/write step')
        print(full.select([c for c in full.columns if c.startswith('t1_ap_')][:6]).describe())
        return
    assert full.height == 223749, full.height
    full = full.join(chrono.drop('p1_is_A'), on=['pair_id', 'hand_id'], how='left')
    cand = [c for c in full.columns if c not in ('pair_id', 'hand_id')]
    n = full.height
    keep, dropped = [], []
    share = {}
    for c in cand:
        vc = full[c].value_counts().sort('count', descending=True)
        top = vc['count'][0] / n
        share[c] = top
        if vc.height == 1 or top > .995:
            dropped.append((c, round(float(top), 5)))
        else:
            keep.append(c)
    if len(keep) > 700:
        keep = sorted(keep, key=lambda c: share[c])[:700]
    keep = [c for c in cand if c in set(keep)]
    print(f'generated {len(cand)} columns, dropped {len(dropped)} degenerate, kept {len(keep)}')
    out = full.select('pair_id', 'hand_id', *keep)
    out = out.with_columns(pl.col(keep).cast(pl.Float32))
    assert out.null_count().to_numpy().sum() == 0, 'nulls in the mechanical block'
    out.write_parquet(OUT + r'\step254_features.parquet')
    print('wrote', out.shape, round(time.time() - t_start, 1), 's')
    with open(OUT + r'\feature_dictionary.md', 'w', encoding='utf-8') as fh:
        fh.write('# STEP254 mechanical block - feature dictionary\n\n')
        fh.write(f'{len(keep)} columns kept of {len(cand)} generated, one row per '
                 f'(pair_id, hand_id) for all {n} hands of the 1,860 labelled development pairs.\n\n')
        fh.write('Roles: A = the member who loses more chips over the pair\'s shared hands, B = the '
                 'other. Suffixes: `_A` / `_B` directional, `_mx` max over the two members, `_mn` '
                 'min, `_sm` sum. Sentinels: -1 = street not reached / not applicable, 9 = the '
                 'event never happened, -9 = strength comparison undefined.\n\n')
        fh.write('| column | family | definition |\n|---|---|---|\n')
        for c in keep:
            base = c
            for suf in ('_A', '_B', '_mx', '_mn', '_sm'):
                if base.endswith(suf) and base[:-len(suf)] in DOC:
                    base = base[:-len(suf)]
                    break
            d = DOC.get(base, DOC.get(c, ''))
            fam = d[1] if d.startswith('(') else c[1]
            fh.write(f'| `{c}` | {fam} | {d[4:] if d.startswith("(") else d} |\n')
        fh.write('\n## dropped as degenerate (constant or one value in >99.5% of rows)\n\n')
        for c, t in dropped:
            fh.write(f'- `{c}` (top value share {t})\n')
    print('wrote feature_dictionary.md')


if __name__ == '__main__':
    main()
