"""STEP264 PROMPT v2: a richer, still fully mechanical renderer of one (pair, hand).

Everything here is computed from the hand record (hands / seats / actions parquet plus the
STEP237 and STEP254 feature tables).  No free-form prose is written by hand anywhere: the worked
examples' rationales are TEMPLATES per class, filled from the same computed facts that the hand's
own fact block prints.  No LLM is called.

What v2 adds on top of render_hand.py (v1):
  * a STREET SUMMARY block: for every street that was actually played - pot at the start of the
    street, the pair's combined share of that pot, how many outsiders were still in, the
    stack-to-pot ratio of A and of B, A's equity against B's real cards, and (flop / turn /
    river) A's and B's exact equity against the LIVE FIELD, every remaining board card
    enumerated;
  * a COUNTS line: bets / raises / calls / checks / folds for A, B and each outsider;
  * six new derived-fact lines: the preflop aggressor, who collected the most chips, the
    showdown and the categories of the hands shown, whether a member folded a hand that would
    have beaten the player who collected the pot, the largest price each outsider faced from A
    or B and whether they folded to it, and the net chip flow A->B and B->A separately;
  * a definition block (what a positive is, what CONFIDENCE means, mechanical cues per family)
    and SIX worked examples in front of the hand to judge.

Conventions inherited from v1 unchanged: A / B / O1-O4 aliases, positions from
(seat_no - button_seat) % 6, preflop action order, bb units, the Task block and the exact
two-line answer format (reward.py must keep parsing it).

Definitions settled here (stated in PROMPT_V2.md too):
  aggressor       the last player on that street whose action was bet, raise, or an all_in with
                  amount > to_call.  Posting a blind is NOT aggression, so before the first
                  voluntary raise the preflop aggressor is nobody.
  street reached  a street with at least one action row.  Board cards dealt after the betting
                  stopped (all-in run-out) never create a "street reached".
  pot winner      the seat that collected the most chips, collected = net_chips +
                  total_contribution (this includes an uncalled bet returned; ties are listed).
  net chip flow   proportional attribution: with loss_i = max(0, -net_i), gain_j = max(0, net_j)
                  and T = sum of the losses (= sum of the gains), flow(i->j) = loss_i*gain_j/T.
                  At most one of flow(A->B), flow(B->A) can be non-zero.
  equity vs field exact enumeration of the remaining board cards with ALL twelve hole cards
                  removed from the deck, against the players still live at the start of that
                  street; ties split.  Preflop it is NOT computed (C(40,5) = 658,008 boards per
                  hand); the preflop line prints the project's starting-hand strength percentile
                  instead and says so.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import sys
from itertools import combinations

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import render_hand as R                                                        # noqa: E402
from render_hand import INSTRUCTION, K, NBOARD, SIDX, STREETS, nf, pc, sf      # noqa: E402,F401
from evidence_aug.step237_render import CARDS                                  # noqa: E402
from phevaluator import evaluate_cards                                         # noqa: E402

CV = R.CV
STREET_SHORT = {'preflop': 'preflop', 'flop': 'flop   ', 'turn': 'turn   ', 'river': 'river  '}

# phevaluator rank classes (1 = royal flush .. 7462 = worst high card); boundaries verified
# on nine constructed hands before use.
RANK_CLASS = ((10, 'straight flush'), (166, 'quads'), (322, 'full house'), (1599, 'flush'),
              (1609, 'straight'), (2467, 'trips'), (3325, 'two pair'), (6185, 'one pair'),
              (7462, 'high card'))


def rank_cat(rank: int) -> str:
    for hi, name in RANK_CLASS:
        if rank <= hi:
            return name
    return 'high card'


def est_tokens(text: str) -> float:
    """project-wide cheap token estimate: characters / 3.6"""
    return len(text) / 3.6


# ---------------------------------------------------------------------------- the data context


class HandContextV2(R.HandContext):
    """v1's context plus the river A-vs-B equity column and per-hand mechanical derivations."""

    F254 = R.HandContext.F254 + ['t2_eq_river_A']

    def __init__(self, keys: pl.DataFrame):
        super().__init__(keys)
        self._state: dict[str, dict] = {}
        self._eqfield: dict[str, dict] = {}

    # ------------------------------------------------------------------ per-hand mechanics
    def state(self, hand_id: str) -> dict:
        """Everything that depends only on the hand, not on which pair is being judged."""
        if hand_id in self._state:
            return self._state[hand_id]
        h = self.h[hand_id]
        bb = float(h['big_blind'])
        rows = {r['player_id']: r for r in self.s[hand_id]}
        btn = h['button_seat']
        acts = self.a.get(hand_id, [])

        # blinds are never action rows; positions are always BTN/SB/BB/UTG/MP/CO 6-handed
        blind = {}
        for p, r in rows.items():
            k = (r['seat_no'] - btn) % 6
            if k == 1:
                blind[p] = h['small_blind']
            elif k == 2:
                blind[p] = h['big_blind']

        played = [s for s in STREETS if any(a['street'] == s for a in acts)]
        # per-street contribution of each player
        contrib = {s: {} for s in STREETS}
        for s in STREETS:
            for p in rows:
                mine = [a for a in acts if a['street'] == s and a['player_id'] == p]
                if mine:
                    contrib[s][p] = float(max(a['amount_to'] for a in mine))
                else:
                    contrib[s][p] = float(blind.get(p, 0)) if s == 'preflop' else 0.0
        for p, r in rows.items():
            tot = sum(contrib[s][p] for s in STREETS)
            assert abs(tot - float(r['total_contribution'])) < 1e-6, \
                f'contribution mismatch {hand_id} {p}: {tot} vs {r["total_contribution"]}'

        # who is out before each street, stacks and pots at the start of each street
        fold_street = {}
        for a in acts:
            if a['action'] == 'fold' and a['player_id'] not in fold_street:
                fold_street[a['player_id']] = SIDX[a['street']]
        # contributions standing at the START of the street: earlier streets, plus the blind on
        # preflop (blinds are not action rows but they are already in pot_before = 1.5 bb).
        contrib_before = {s: {p: sum(contrib[t][p] for t in STREETS if SIDX[t] < SIDX[s])
                              + (float(blind.get(p, 0)) if s == 'preflop' else 0.0)
                              for p in rows} for s in played}
        live_start, stack_start, pot_start = {}, {}, {}
        for s in played:
            i = SIDX[s]
            live_start[s] = [p for p in rows if fold_street.get(p, 9) >= i]
            stack_start[s] = {p: (float(rows[p]['starting_stack']) - contrib_before[s][p]) / bb
                              for p in rows}
            pot_start[s] = float(next(a for a in acts if a['street'] == s)['pot_before']) / bb
        pair_contrib_before = contrib_before

        # action counts and the preflop aggressor
        counts = {p: [0, 0, 0, 0, 0] for p in rows}          # bet, raise, call, check, fold
        pre_agg = None
        for a in acts:
            p, act = a['player_id'], a['action']
            if act == 'bet':
                counts[p][0] += 1
            elif act == 'raise':
                counts[p][1] += 1
            elif act == 'call':
                counts[p][2] += 1
            elif act == 'check':
                counts[p][3] += 1
            elif act == 'fold':
                counts[p][4] += 1
            else:                                            # all_in
                counts[p][1 if a['amount'] > a['to_call'] else 2] += 1
            if a['street'] == 'preflop' and _is_agg(a):
                pre_agg = p

        # who collected the pot
        coll = {p: (float(rows[p]['net_chips']) + float(rows[p]['total_contribution'])) / bb
                for p in rows}
        best_coll = max(coll.values())
        winners = sorted([p for p in rows if coll[p] >= best_coll - 1e-9],
                         key=lambda p: rows[p]['seat_no'])

        # net chip flow, proportional attribution of the losses to the gains
        loss = {p: max(0.0, -float(rows[p]['net_chips'])) / bb for p in rows}
        gain = {p: max(0.0, float(rows[p]['net_chips'])) / bb for p in rows}
        tot_loss = sum(loss.values())

        # best five-card hand of every player on the dealt board
        board = (h['board_cards'] or '').split()
        nb = len(board)
        bids = [CARDS[c] for c in board]
        best5 = {}
        if nb >= 3:
            for p, r in rows.items():
                best5[p] = evaluate_cards(CARDS[r['hole_card_1']], CARDS[r['hole_card_2']], *bids)

        st = {'bb': bb, 'rows': rows, 'acts': acts, 'blind': blind, 'played': played,
              'contrib': contrib, 'live_start': live_start, 'stack_start': stack_start,
              'pot_start': pot_start, 'contrib_before': pair_contrib_before, 'counts': counts,
              'pre_agg': pre_agg, 'coll': coll, 'winners': winners, 'loss': loss, 'gain': gain,
              'tot_loss': tot_loss, 'best5': best5, 'board': board, 'nb': nb,
              'fold_street': fold_street}
        self._state[hand_id] = st
        return st

    def eq_field(self, hand_id: str) -> dict:
        """{street: {player: equity vs the live field}} for flop / turn / river, exact."""
        if hand_id in self._eqfield:
            return self._eqfield[hand_id]
        st = self.state(hand_id)
        rows, board, nb = st['rows'], st['board'], st['nb']
        hole = {p: (CARDS[r['hole_card_1']], CARDS[r['hole_card_2']]) for p, r in rows.items()}
        dead_base = set()
        for p in hole:
            dead_base.update(hole[p])
        out: dict[str, dict] = {}
        for s in ('flop', 'turn', 'river'):
            if s not in st['played']:
                continue
            need = NBOARD[SIDX[s]]
            if nb < need:
                continue
            live = [p for p in st['live_start'][s] if p in hole]
            base = [CARDS[c] for c in board[:need]]
            # only the cards visible at that moment are removed, plus every hole card
            deck = [c for c in range(52) if c not in dead_base and c not in base]
            miss = 5 - need
            eq = {p: 0.0 for p in live}
            if miss == 0:
                combos = [()]
            else:
                combos = list(combinations(deck, miss))
            for comb in combos:
                full = base + list(comb)
                vals = [(evaluate_cards(hole[p][0], hole[p][1], *full), p) for p in live]
                best = min(v for v, _ in vals)
                win = [p for v, p in vals if v == best]
                share = 1.0 / len(win)
                for p in win:
                    eq[p] += share
            n = len(combos)
            out[s] = {p: eq[p] / n for p in live}
        self._eqfield[hand_id] = out
        return out


def _is_agg(a: dict) -> bool:
    return a['action'] in ('bet', 'raise') or (a['action'] == 'all_in'
                                               and a['amount'] > a['to_call'])


# ---------------------------------------------------------------------------- the fact block


def _facts_v2(ctx: HandContextV2, pair_id: str, hand_id: str, ma: str, mb: str,
              order: list, alias: dict, rows: dict, nb: int, bb: float) -> list[tuple[str, str]]:
    """(key, line) pairs.  The first ten reproduce v1 exactly; six new ones follow."""
    f = ctx.f237[(pair_id, hand_id)]
    g = ctx.f254[(pair_id, hand_id)]
    st = ctx.state(hand_id)
    p1, _p2 = ctx.members[pair_id]
    flip = ((ctx.p1_is_A[(pair_id, hand_id)] == 1) != (ma == p1))

    out: list[tuple[str, str]] = []
    # ---------------------------------------------------------------- v1, verbatim logic
    if g['t2_eq_turn_A'] >= 0:
        eq, where = g['t2_eq_turn_A'], 'from the turn, every remaining card enumerated'
    elif g['t2_eq_flop_A'] >= 0:
        eq, where = g['t2_eq_flop_A'], 'from the flop, every remaining card enumerated'
    else:
        eq, where = g['t2_eq_pre_A'], 'all in before the flop (starting-hand estimate)'
    if flip:
        eq = 1.0 - eq
    out.append(('eq_ab', f'- A\'s cards beat B\'s cards {pc(eq)} of the time {where}.'))
    out.append(('ranks', R._rank_line(ctx, hand_id, order, alias, nb)))
    n_fp = int(f['fp_count'])
    if n_fp == 0:
        out.append(('fold_to_partner',
                    '- Neither member folded facing a bet or raise from the other.'))
    else:
        extra = []
        if f['fp_best_live'] > 0:
            extra.append('at one such fold the folder held the best hand still in')
        if f['fp_betfold'] > 0:
            extra.append('the folder had already put chips in that street')
        if f['fp_partner_wins'] > 0:
            extra.append('the other member finished up')
        out.append(('fold_to_partner',
                    f'- A member folded to the other\'s bet or raise {n_fp}x'
                    + ('; ' + '; '.join(extra) if extra else '') + '.'))
    nout = int(round(f['iso_out_fold_share'] * 4))
    out.append(('out_folds',
                {0: '- No non-member folded facing a bet or raise from A or B.',
                 4: '- All four non-members folded facing a bet or raise from A or B.'}.get(
                     nout, f'- {nout} of the four non-members folded facing a bet or raise '
                           f'from A or B.')))
    hu = int(f['hu_pair_streets'])
    out.append(('hu_streets',
                f'- A and B were the only players left for {hu} postflop '
                f'street{"s" if hu > 1 else ""}.' if hu else
                '- A and B were never the only two players left after the flop.'))
    tr = float(f['tr_internal_bb']) or float(g['t5_transfer_bb'])
    out.append(('transfer',
                f'- {nf(tr)} bb moved directly from one member to the other.' if tr > 0 else
                '- No chips moved directly between A and B.'))
    out.append(('pot_share',
                f'- A and B together supplied {pc(float(f["pair_pot_share"]))} of the final pot.'))

    cond: list[tuple[str, str]] = []
    if n_fp > 0 and f['fp_price_min'] < 1.0:
        cond.append(('fp_price',
                     f'- The cheapest such fold was at {pc(float(f["fp_price_min"]))} pot odds.'))
    if f['fp_wouldwin'] > 0 and nb >= 3:
        cond.append(('fp_wouldwin',
                     '- That folder\'s cards would have beaten every player who stayed in.'))
    if f['checkdown_best'] > 0:
        cond.append(('checkdown_best',
                     '- On the last street a member checked holding the best hand still in, with '
                     'the other member still in the pot.'))
    if f['iso_clean'] > 0:
        cond.append(('iso_clean',
                     '- A member made the last preflop raise, the other folded to a member\'s '
                     'bet, and at most one non-member saw the flop.'))
    if f['passive_winner'] > 0:
        cond.append(('passive_winner',
                     '- A member won at showdown with the best hand there without betting or '
                     'raising after the flop, the other having put in at least 2 bb.'))
    if f['pay_partner_bb'] > 0:
        cond.append(('pay_partner',
                     f'- A member put {nf(float(f["pay_partner_bb"]))} bb in after the flop '
                     f'calling the other member, who finished up.'))
    if f['coldcall_partner_bb'] > 0:
        cond.append(('coldcall',
                     f'- A member cold-called {nf(float(f["coldcall_partner_bb"]))} bb facing the '
                     f'other member.'))
    if f['call_partner_dominated_bb'] > 0:
        cond.append(('dominated_call',
                     f'- A member called {nf(float(f["call_partner_dominated_bb"]))} bb facing '
                     f'the other member while holding the weaker of the two hands.'))
    if f['iso_weak_agg_behind'] > .6:
        cond.append(('weak_agg_behind',
                     '- A member bet or raised with a weak holding while the other still had to '
                     'act behind.'))
    if f['oof_clearway'] > 0:
        cond.append(('oof_clearway',
                     '- A member folded to a non-member\'s bet holding the best hand still in, '
                     'with the other member still in the pot.'))
    if f['tr_loser_share'] > 0:
        cond.append(('loser_share',
                     f'- The member who finished down supplied '
                     f'{pc(float(f["tr_loser_share"]))} of the final pot and the other collected '
                     f'at least half of it.'))
    cond.append(('both_agg', '- Both members bet or raised at some point.'
                 if f['both_agg_live'] > 0 else '- At most one of the two members bet or raised.'))
    vpip = [any(x['player_id'] == m and x['action'] not in ('fold', 'check')
                for x in ctx.a.get(hand_id, [])) for m in (ma, mb)]
    cond.append(('vpip', {2: '- Both members put chips in voluntarily.',
                          1: '- Only one of the two members put chips in voluntarily.',
                          0: '- Neither member put chips in voluntarily beyond the blinds.'
                          }[sum(vpip)]))
    nnf = 6 - len([p for p in order if rows[p]['folded']])
    cond.append(('nfold', '- Only one player did not fold.' if nnf == 1 else
                 f'- {nnf} of the six players did not fold.'))
    out += cond[:3]

    # ---------------------------------------------------------------- v2 additions
    out.append(('pre_agg',
                f'- Last preflop aggressor: {alias[st["pre_agg"]]}.' if st['pre_agg'] else
                '- Nobody raised before the flop.'))
    w = st['winners']
    out.append(('winner',
                '- Most chips collected: '
                + ', '.join(f'{alias[p]} {nf(st["coll"][p])} bb' for p in w) + '.'))
    sd = [p for p in order if rows[p]['went_to_showdown']]
    if len(sd) >= 2 and st['best5']:
        sd = sorted(sd, key=lambda p: st['best5'][p])
        parts, prev = [f'{alias[sd[0]]} {rank_cat(st["best5"][sd[0]])}'], st['best5'][sd[0]]
        for p in sd[1:]:
            parts.append(('= ' if st['best5'][p] == prev else '> ')
                         + f'{alias[p]} {rank_cat(st["best5"][p])}')
            prev = st['best5'][p]
        out.append(('showdown', '- Showdown: ' + ' '.join(parts) + '.'))
    else:
        out.append(('showdown', '- No showdown: the pot was not contested to the end.'))
    out.append(('fold_ahead', _fold_ahead_line(st, ma, mb, alias, nb)))
    out.append(('out_price', _out_price_line(st, ma, mb, alias, order)))
    fab = (st['loss'][ma] * st['gain'][mb] / st['tot_loss']) if st['tot_loss'] > 0 else 0.0
    fba = (st['loss'][mb] * st['gain'][ma] / st['tot_loss']) if st['tot_loss'] > 0 else 0.0
    out.append(('flow', f'- Net chip flow between the members: A->B {nf(fab)} bb, '
                        f'B->A {nf(fba)} bb.'))
    return out


def _fold_ahead_line(st: dict, ma: str, mb: str, alias: dict, nb: int) -> str:
    if nb < 3:
        return '- No board was dealt, so no folded hand can be compared with the winner\'s.'
    folders = [m for m in (ma, mb) if st['rows'][m]['folded']]
    if not folders:
        return '- Neither member folded, so no folded member hand can be compared.'
    w = st['winners'][0]
    ahead = [m for m in folders if m != w and st['best5'][m] < st['best5'][w]]
    if not ahead:
        return ('- The member who folded did not hold a hand better than the winner\'s.'
                if len(folders) == 1 else
                '- Neither folding member held a hand better than the winner\'s.')
    return ('- ' + ' and '.join(alias[m] for m in ahead)
            + f' folded a hand that would have beaten {alias[w]}, who collected the pot.')


def _out_price_line(st: dict, ma: str, mb: str, alias: dict, order: list) -> str:
    """Largest to_call an outsider faced while A or B was the standing aggressor."""
    members = {ma, mb}
    best: dict[str, tuple[float, bool]] = {}
    cur = None
    street = None
    for a in st['acts']:
        if a['street'] != street:
            street, cur = a['street'], None
        p = a['player_id']
        if p not in members and cur in members and a['to_call'] > 0:
            price = float(a['to_call']) / st['bb']
            if p not in best or price > best[p][0]:
                best[p] = (price, a['action'] == 'fold')
        if _is_agg(a):
            cur = p
    outs = [p for p in order if p not in members]
    parts = []
    for p in outs:
        if p in best:
            parts.append(f'{alias[p]} {nf(best[p][0])} {"folded" if best[p][1] else "paid"}')
        else:
            parts.append(f'{alias[p]} none')
    return '- Largest price each outsider faced from A or B (bb): ' + ', '.join(parts) + '.'


# ---------------------------------------------------------------------------- blocks


def _street_block(ctx: HandContextV2, pair_id: str, hand_id: str, ma: str, mb: str,
                  compact: bool = False) -> list[str]:
    st = ctx.state(hand_id)
    g = ctx.f254[(pair_id, hand_id)]
    p1, _ = ctx.members[pair_id]
    flip = ((ctx.p1_is_A[(pair_id, hand_id)] == 1) != (ma == p1))
    eqab = {}
    for s, col in (('preflop', 't2_eq_pre_A'), ('flop', 't2_eq_flop_A'),
                   ('turn', 't2_eq_turn_A'), ('river', 't2_eq_river_A')):
        v = float(g[col])
        eqab[s] = (1.0 - v) if (v >= 0 and flip) else v
    ef = ctx.eq_field(hand_id)
    stl = ctx.strengths(hand_id)
    members = {ma, mb}
    L = ['Street summary (values at the START of each street that was played)']
    for s in st['played']:
        pot = st['pot_start'][s]
        cb = st['contrib_before'][s]
        share = (cb[ma] + cb[mb]) / (pot * st['bb']) if pot > 0 else 0.0
        nout = len([p for p in st['live_start'][s] if p not in members])
        seg = [f'  {STREET_SHORT[s]} pot {nf(pot)}  A+B share {pc(share)}  outsiders in {nout}']
        if not compact:
            sp = []
            for m, nm in ((ma, 'A'), (mb, 'B')):
                if m in st['live_start'][s]:
                    sp.append(f'{nm} {nf(st["stack_start"][s][m] / pot) if pot > 0 else "-"}')
                else:
                    sp.append(f'{nm} out')
            seg.append('SPR ' + ' '.join(sp))
        if eqab[s] >= 0:
            seg.append(f'A beats B {pc(eqab[s])}')
        if s in ef:
            fa = ef[s].get(ma)
            fb = ef[s].get(mb)
            bits = []
            if fa is not None:
                bits.append(f'A {pc(fa)}')
            if fb is not None:
                bits.append(f'B {pc(fb)}')
            if bits:
                seg.append('vs field ' + ' '.join(bits))
        elif s == 'preflop':
            seg.append(f'start-hand pct A {pc(stl[ma][0])} B {pc(stl[mb][0])}')
        L.append('  '.join(seg))
    return L


def _counts_block(ctx: HandContextV2, hand_id: str, order: list, alias: dict) -> list[str]:
    st = ctx.state(hand_id)
    cells = ['/'.join(str(x) for x in st['counts'][p]) for p in order]
    return ['Counts per player, bets/raises/calls/checks/folds',
            '  ' + '   '.join(f'{alias[p]} {c}' for p, c in zip(order, cells))]


def body(ctx: HandContextV2, pair_id: str, hand_id: str, a_is_player_1: bool,
         fact_keys: list[str] | None = None, with_counts: bool = True,
         compact: bool = False) -> str:
    """The hand itself: Table, Action, Result, Street summary, Counts, Derived facts.

    fact_keys: when given, the FIRST SIX of those keys that this hand actually has are emitted,
    in that order (used by the worked examples, which are trimmed to six lines).
    compact:   drops the Collected line and the SPR segment (worked examples, to save tokens)."""
    p1, p2 = ctx.members[pair_id]
    ma, mb = (p1, p2) if a_is_player_1 else (p2, p1)
    h = ctx.h[hand_id]
    bb = float(h['big_blind'])
    sb = h['small_blind'] / bb
    rows, order, alias, pos = R._aliases(ctx, hand_id, ma, mb)
    acts = ctx.a.get(hand_id, [])
    board = (h['board_cards'] or '').split()
    nb = len(board)
    last_street = acts[-1]['street'] if acts else 'preflop'

    L = ['Table', f'  Blinds {nf(sb)} / 1 bb, so the pot starts at {nf(sb + 1)} bb.',
         '  Seats in preflop action order (position, cards, stack):']
    for p in order:
        r = rows[p]
        L.append(f'    {alias[p]:<2} {pos[p]:<3}  {r["hole_card_1"]} {r["hole_card_2"]}  '
                 f'{nf(r["starting_stack"] / bb)} bb')
    if nb:
        b = f'  Board: flop {" ".join(board[:3])}'
        if nb >= 4:
            b += f' | turn {board[3]}'
        if nb >= 5:
            b += f' | river {board[4]}'
        L.append(b)
    L += ['', 'Action']
    cur = None
    for r in acts:
        if r['street'] != cur:
            cur = r['street']
            L.append(f'  -- {R.STREET_TITLE[cur]} --')
        L.append(R._action_line(r, alias, pos, bb))
    if nb > NBOARD[SIDX[last_street]]:
        L.append(f'  -- betting ended on the {last_street}; a player was all in and the rest of '
                 f'the board was dealt --')
    L += ['', 'Result']
    pot = h['final_pot'] / bb
    sdn = h['players_at_showdown']
    sd = [alias[p] for p in order if rows[p]['went_to_showdown']]
    L.append(f'  Final pot {nf(pot)} bb. ' +
             (f'Showdown between {", ".join(sd)}.' if sdn >= 2 else 'No showdown.'))
    if not compact:
        coll = [(alias[p], (rows[p]['net_chips'] + rows[p]['total_contribution']) / bb)
                for p in order if rows[p]['net_chips'] + rows[p]['total_contribution'] > 0]
        L.append('  Collected: ' + ', '.join(f'{a} {nf(v)}' for a, v in coll) +
                 ' (includes any uncalled bet returned).')
    L.append('  Net (bb): ' + ', '.join(f'{alias[p]} {sf(rows[p]["net_chips"] / bb)}'
                                        for p in order) + '.')
    L.append('')
    L += _street_block(ctx, pair_id, hand_id, ma, mb, compact)
    if with_counts:
        L.append('')
        L += _counts_block(ctx, hand_id, order, alias)
    L.append('')
    facts = _facts_v2(ctx, pair_id, hand_id, ma, mb, order, alias, rows, nb, bb)
    if fact_keys is None:
        L.append('Derived facts (computed from the hand record)')
        L += [t for _k, t in facts]
    else:
        d = dict(facts)
        keep = [k for k in fact_keys if k in d][:6]
        L.append('Derived facts (the six most relevant lines)')
        L += [d[k] for k in keep]
    return '\n'.join(L)


def facts_dict(ctx: HandContextV2, pair_id: str, hand_id: str, a_is_player_1: bool) -> dict:
    p1, p2 = ctx.members[pair_id]
    ma, mb = (p1, p2) if a_is_player_1 else (p2, p1)
    rows, order, alias, _pos = R._aliases(ctx, hand_id, ma, mb)
    nb = len((ctx.h[hand_id]['board_cards'] or '').split())
    bb = float(ctx.h[hand_id]['big_blind'])
    return dict(_facts_v2(ctx, pair_id, hand_id, ma, mb, order, alias, rows, nb, bb))


# ---------------------------------------------------------------------------- the fixed head

PREAMBLE = """You are reviewing one six-handed no-limit hold'em hand.  Two players are marked A \
and B, the other four O1-O4 in preflop action order.  Amounts are in big blinds (bb); "odds" is \
the call price as a share of the pot after calling; SPR is stack / pot at the start of a street.

WHAT COUNTS AS A POSITIVE
A positive is a hand strong enough that an investigator would CITE it as evidence of coordination
between A and B - the label is "listed evidence hand of a true pair".  Ordinary hands of a
colluding pair are NOT positives; only the cited ones are.

CONFIDENCE
CONFIDENCE is your probability, an integer in [0,100], that THIS hand is such a cited piece of
evidence.  It is not how sure you feel about your verdict.  Hard rule: a NONE verdict must carry
CONFIDENCE below 50, every other verdict 50 or above.

MECHANICAL CUES
DIRECTED_TRANSFER
  - a member folds a hand better than the winner's to the other's bet, at cheap pot odds;
  - chips move member -> member: the net flow A->B or B->A is large next to the pot;
  - the member who finishes down supplied most of the pot and the other collected it.
SOFT_PLAY
  - heads-up against each other the members check or call instead of betting or raising;
  - a member checks the last street holding the best hand still in, the other still in the pot;
  - a member wins at showdown having never bet or raised after the flop.
COORDINATED_ISOLATION
  - both members bet or raise and the outsiders fold: the outsider fold share is high;
  - one member makes the last preflop raise, the other backs it, and at most one outsider sees
    the flop;
  - a member bets or raises weak while the other still acts behind, leaving one outsider facing
    them both.
Cues are evidence, not proof.  Most hands between any two players are ordinary, so NONE is the
usual answer."""

EX_HEAD = ('=== SIX WORKED EXAMPLES (same rendering; their derived facts are trimmed to the six '
           'most relevant lines) ===')
EX_FOOT = '=== END OF WORKED EXAMPLES ==='
TARGET_HEAD = '=== THE HAND TO JUDGE ==='


def assemble(examples_block: str, hand_body: str) -> str:
    """PREAMBLE, the six worked examples, the hand to judge, then v1's Task block verbatim."""
    return ('\n\n'.join([PREAMBLE, examples_block, TARGET_HEAD + '\n' + hand_body])
            + '\n' + INSTRUCTION)
