"""STEP264: deterministic text renderer of one (pair, hand) as an LLM prompt.

Development phase only (asserted).  The prompt contains NO pair_id / hand_id / player_id /
table_id string, no timestamp and no hand-order index: ID formats and chronological ordering are
prohibited artefacts in this competition.  The two pair members are printed as A and B, the other
four seats as O1..O4 in preflop action order.  WHICH member is printed as A is decided by the
caller (``a_is_player_1``) and is randomised per prompt in build_dataset.py, because an earlier
LLM study failed when the model followed the position of the named player instead of the evidence.

Poker conventions reused from the project rather than re-invented:
  * hand-strength percentiles and the preflop heuristic come from
    ``evidence_aug.step237_render`` (``CARDS``, ``preflop_strength``), identical to the
    STEP237 / STEP254 feature builders;
  * positions come from ``(seat_no - button_seat) % 6`` with 0 = BTN, exactly the STEP254
    convention (``src/evidence_aug/step254_features.py``, slot ``t3_pos_A``);
  * preflop action order key ``(seat_no - button_seat - 3) % 6`` and postflop
    ``(seat_no - button_seat - 1) % 6`` are the ``nseat_order`` maps of step254_features.main().

Raw-table semantics settled by inspection on 2026-09-18 (see README.txt):
  amount      = incremental chips added by this action
  amount_to   = the actor's TOTAL contribution on this street after the action, blind included
  pot_before  = running pot before the action, blinds included
  to_call     = chips needed to call, capped at stack_before
  players_active = players not yet folded at the moment of the action, actor included
  won_share   = share of final_pot RETURNED to the seat, uncalled bets included, so it is not
                "the pot won"; the renderer prints net_chips and
                collected = net_chips + total_contribution, both exact integers
  board_cards = may hold 5 cards although betting stopped earlier; every such hand has
                players_at_showdown >= 2, i.e. it is an all-in run-out, which the renderer flags.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import sys

import polars as pl

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
S237 = CV + r'\experiments\step237_invisible_events_20260917\step237_features.parquet'
S254D = CV + r'\experiments\step254_mechanical_tensor_20260917'

if CV + r'\src' not in sys.path:
    sys.path.insert(0, CV + r'\src')
from evidence_aug.step237_render import CARDS, preflop_strength  # noqa: E402
from phevaluator import evaluate_cards  # noqa: E402

POS_NAMES = ('BTN', 'SB', 'BB', 'UTG', 'MP', 'CO')
STREETS = ('preflop', 'flop', 'turn', 'river')
SIDX = {s: i for i, s in enumerate(STREETS)}
NBOARD = (0, 3, 4, 5)
STREET_TITLE = {'preflop': 'PREFLOP', 'flop': 'FLOP', 'turn': 'TURN', 'river': 'RIVER'}

# the fixed tail of every prompt
INSTRUCTION = """
Task
Did A and B play this hand in coordination with each other?
  DIRECTED_TRANSFER      one of them deliberately loses chips to the other.
  SOFT_PLAY              they avoid betting or raising against each other.
  COORDINATED_ISOLATION  they act together against a third player.
  NONE                   an ordinary hand.
Most hands between any two players are ordinary, so NONE is the usual answer.
Reason briefly, then end your reply with exactly these two lines and nothing after them:
VERDICT: <NONE|DIRECTED_TRANSFER|SOFT_PLAY|COORDINATED_ISOLATION>
CONFIDENCE: <integer 0-100, your probability that A and B coordinated on this hand>"""

# ---------------------------------------------------------------------------- number formatting


def nf(x: float) -> str:
    """one decimal, trailing '.0' dropped: 1.0 -> '1', 2.5 -> '2.5', -11.25 -> '-11.2'"""
    s = f'{x:.1f}'
    return s[:-2] if s.endswith('.0') else s


def sf(x: float) -> str:
    """signed, one decimal, '.0' dropped, plain '0' for zero: +9, -2.5, 0"""
    if abs(x) < .05:
        return '0'
    s = f'{x:+.1f}'
    return s[:-2] if s.endswith('.0') else s


def pc(x: float) -> str:
    return f'{round(100 * x):.0f}%'


def est_tokens(text: str) -> float:
    """project-wide cheap token estimate: characters / 3.6"""
    return len(text) / 3.6


# ---------------------------------------------------------------------------- the data context


class HandContext:
    """Loads everything the renderer needs for a fixed set of (pair_id, hand_id) keys."""

    F237 = ['fp_best_live', 'fp_price_min', 'fp_wouldwin', 'fp_partner_wins', 'fp_betfold',
            'fp_count', 'iso_out_fold_share', 'iso_clean', 'iso_weak_agg_behind', 'oof_clearway',
            'tr_internal_bb', 'tr_loser_share', 'hu_pair_streets', 'pay_partner_bb',
            'passive_winner', 'checkdown_best', 'coldcall_partner_bb',
            'call_partner_dominated_bb', 'both_agg_live', 'pair_pot_share']
    F254 = ['t2_eq_pre_A', 't2_eq_flop_A', 't2_eq_turn_A', 't5_transfer_bb']

    def __init__(self, keys: pl.DataFrame):
        """keys: a frame with columns pair_id, hand_id (duplicates allowed)."""
        k = keys.select('pair_id', 'hand_id').unique()
        ids = k.select('hand_id').unique()
        lab = pl.read_csv(DATA + r'\development_labels.csv').select('pair_id', 'player_1',
                                                                    'player_2')
        self.members = {r['pair_id']: (r['player_1'], r['player_2'])
                        for r in lab.join(k.select('pair_id').unique(), on='pair_id')
                        .iter_rows(named=True)}
        hands = (pl.scan_parquet(DATA + r'\hands.parquet').join(ids.lazy(), on='hand_id')
                 .select('hand_id', 'phase', 'button_seat', 'small_blind', 'big_blind',
                         'board_cards', 'final_pot', 'players_dealt', 'players_at_showdown')
                 .collect())
        assert hands.height == ids.height, 'missing hands'
        assert (hands['phase'] == 'development').all(), 'non-development hand in the render set'
        assert (hands['players_dealt'] == 6).all(), 'not a 6-max hand'
        self.h = {r['hand_id']: r for r in hands.drop('phase').iter_rows(named=True)}
        seats = (pl.scan_parquet(DATA + r'\seats.parquet').join(ids.lazy(), on='hand_id')
                 .select('hand_id', 'player_id', 'seat_no', 'starting_stack', 'hole_card_1',
                         'hole_card_2', 'total_contribution', 'net_chips', 'folded',
                         'went_to_showdown', 'won_share').collect().sort('hand_id', 'seat_no'))
        self.s: dict[str, list[dict]] = {}
        for r in seats.iter_rows(named=True):
            self.s.setdefault(r['hand_id'], []).append(r)
        acts = (pl.scan_parquet(DATA + r'\actions.parquet').join(ids.lazy(), on='hand_id')
                .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount',
                        'amount_to', 'pot_before', 'to_call', 'players_active')
                .collect().sort('hand_id', 'action_no'))
        self.a: dict[str, list[dict]] = {}
        for r in acts.iter_rows(named=True):
            self.a.setdefault(r['hand_id'], []).append(r)
        f237 = pl.read_parquet(S237, columns=['pair_id', 'hand_id'] + self.F237).join(k, on=K)
        self.f237 = {(r['pair_id'], r['hand_id']): r for r in f237.iter_rows(named=True)}
        f254 = pl.read_parquet(S254D + r'\step254_features.parquet',
                               columns=['pair_id', 'hand_id'] + self.F254).join(k, on=K)
        self.f254 = {(r['pair_id'], r['hand_id']): r for r in f254.iter_rows(named=True)}
        ch = pl.read_parquet(S254D + r'\step254_chrono.parquet',
                             columns=['pair_id', 'hand_id', 'p1_is_A']).join(k, on=K)
        self.p1_is_A = {(r['pair_id'], r['hand_id']): int(r['p1_is_A'])
                        for r in ch.iter_rows(named=True)}
        assert len(self.f237) == k.height and len(self.f254) == k.height, 'feature join lost rows'
        self._strength: dict[str, dict] = {}

    # ------------------------------------------------------------------ per-hand helpers
    def strengths(self, hand_id: str) -> dict[str, list]:
        """player -> [preflop, flop, turn, river] percentile, None past the dealt board."""
        if hand_id in self._strength:
            return self._strength[hand_id]
        board = (self.h[hand_id]['board_cards'] or '').split()
        ids = [CARDS[c] for c in board]
        out = {}
        for r in self.s[hand_id]:
            a, b = CARDS[r['hole_card_1']], CARDS[r['hole_card_2']]
            vals = [preflop_strength(a, b)]
            for n in NBOARD[1:]:
                vals.append((7463 - evaluate_cards(a, b, *ids[:n])) / 7462
                            if len(ids) >= n else None)
            out[r['player_id']] = vals
        self._strength[hand_id] = out
        return out


K = ['pair_id', 'hand_id']


# ---------------------------------------------------------------------------- the renderer


def _aliases(ctx: HandContext, hand_id: str, ma: str, mb: str):
    h = ctx.h[hand_id]
    btn = h['button_seat']
    rows = {r['player_id']: r for r in ctx.s[hand_id]}
    pre_key = {p: (r['seat_no'] - btn - 3) % 6 for p, r in rows.items()}
    order = sorted(rows, key=lambda p: pre_key[p])
    alias, k = {}, 0
    for p in order:
        if p == ma:
            alias[p] = 'A'
        elif p == mb:
            alias[p] = 'B'
        else:
            k += 1
            alias[p] = f'O{k}'
    pos = {p: POS_NAMES[(rows[p]['seat_no'] - btn) % 6] for p in rows}
    return rows, order, alias, pos


def _action_line(r: dict, alias: dict, pos: dict, bb: float) -> str:
    p = r['player_id']
    who = f'{alias[p]}({pos[p]})'
    amt, to_call, pot = r['amount'] / bb, r['to_call'] / bb, r['pot_before'] / bb
    to = r['amount_to'] / bb
    act = r['action']
    tail = f'  pot {nf(pot)}  {r["players_active"]} in'
    if act == 'fold':
        head = f'{who} folds'
        if to_call > 0:
            head += f' ({nf(to_call)} to call, {pc(to_call / (pot + to_call))} odds)'
    elif act == 'check':
        head = f'{who} checks'
    elif act == 'call':
        head = f'{who} calls {nf(amt)}'
        if to_call > 0 and pot + to_call > 0:
            head += f' ({pc(to_call / (pot + to_call))} odds)'
    elif act == 'bet':
        head = f'{who} bets {nf(amt)}'
        if pot > 0:
            head += f' ({pc(amt / pot)} pot)'
    elif act == 'raise':
        head = f'{who} raises to {nf(to)} (+{nf(amt)}'
        head += f', {pc(amt / pot)} pot)' if pot > 0 else ')'
    else:                                    # all_in
        if r['amount'] > r['to_call']:
            head = f'{who} all in to {nf(to)} (+{nf(amt)}'
            head += f', {pc(amt / pot)} pot)' if pot > 0 else ')'
        else:
            head = f'{who} calls all in {nf(amt)}'
            if to_call > 0 and pot + to_call > 0:
                head += f' ({pc(to_call / (pot + to_call))} odds)'
    return '  ' + head + tail


def _rank_line(ctx: HandContext, hand_id: str, order: list, alias: dict, nb: int) -> str:
    """Ordering of all six final five-card hands on the dealt board (LLMs cannot do this)."""
    if nb < 3:
        return '- The hand ended before the flop, so no board was dealt.'
    st = ctx.strengths(hand_id)
    sidx = 1 if nb == 3 else (2 if nb == 4 else 3)
    vals = [(alias[p], st[p][sidx]) for p in order]
    vals.sort(key=lambda t: -t[1])
    parts, prev = [vals[0][0]], vals[0][1]
    for a, v in vals[1:]:
        parts.append(('=' if abs(v - prev) < 1e-12 else '>') + ' ' + a)
        prev = v
    chain = parts[0] + ' ' + ' '.join(parts[1:])
    return f'- Best five-card hands on the dealt board: {chain}.'


def _facts(ctx: HandContext, pair_id: str, hand_id: str, ma: str, mb: str,
           order: list, alias: dict, rows: dict, nb: int, bb: float) -> list[str]:
    """At most 10 plain-language lines; 7 are emitted for every hand so that the block length
    does not itself signal the label, then up to 3 conditional lines in priority order."""
    f = ctx.f237[(pair_id, hand_id)]
    g = ctx.f254[(pair_id, hand_id)]
    p1, _p2 = ctx.members[pair_id]
    s254_A_is_p1 = ctx.p1_is_A[(pair_id, hand_id)] == 1
    prompt_A_is_p1 = (ma == p1)
    flip = (s254_A_is_p1 != prompt_A_is_p1)           # our A is STEP254's B -> equity flips

    always: list[str] = []
    # (1) exact enumerated equity of A's two cards against B's two cards
    if g['t2_eq_turn_A'] >= 0:
        eq, where = g['t2_eq_turn_A'], 'from the turn, every remaining card enumerated'
    elif g['t2_eq_flop_A'] >= 0:
        eq, where = g['t2_eq_flop_A'], 'from the flop, every remaining card enumerated'
    else:
        eq, where = g['t2_eq_pre_A'], 'all in before the flop (starting-hand estimate)'
    if flip:
        eq = 1.0 - eq
    always.append(f'- A\'s cards beat B\'s cards {pc(eq)} of the time {where}.')
    # (2) full ranking of the six final hands on the dealt board
    always.append(_rank_line(ctx, hand_id, order, alias, nb))
    # (3) folds by one member facing the other
    n_fp = int(f['fp_count'])
    if n_fp == 0:
        always.append('- Neither member folded facing a bet or raise from the other.')
    else:
        extra = []
        if f['fp_best_live'] > 0:
            extra.append('at one such fold the folder held the best hand still in')
        if f['fp_betfold'] > 0:
            extra.append('the folder had already put chips in that street')
        if f['fp_partner_wins'] > 0:
            extra.append('the other member finished up')
        always.append(f'- A member folded to the other\'s bet or raise {n_fp}x'
                      + ('; ' + '; '.join(extra) if extra else '') + '.')
    # (4) how many non-members folded facing pair aggression
    nout = int(round(f['iso_out_fold_share'] * 4))
    always.append({0: '- No non-member folded facing a bet or raise from A or B.',
                   4: '- All four non-members folded facing a bet or raise from A or B.'}.get(
                       nout, f'- {nout} of the four non-members folded facing a bet or raise '
                             f'from A or B.'))
    # (5) heads-up postflop streets
    hu = int(f['hu_pair_streets'])
    always.append(f'- A and B were the only players left for {hu} postflop '
                  f'street{"s" if hu > 1 else ""}.' if hu else
                  '- A and B were never the only two players left after the flop.')
    # (6) chips moved directly between the two members
    tr = float(f['tr_internal_bb']) or float(g['t5_transfer_bb'])
    always.append(f'- {nf(tr)} bb moved directly from one member to the other.' if tr > 0 else
                  '- No chips moved directly between A and B.')
    # (7) how much of the pot the pair supplied
    always.append(f'- A and B together supplied {pc(float(f["pair_pot_share"]))} of the final pot.')

    # ---- conditional lines, priority order, at most 3
    cond: list[str] = []
    if n_fp > 0 and f['fp_price_min'] < 1.0:
        cond.append(f'- The cheapest such fold was at {pc(float(f["fp_price_min"]))} pot odds.')
    if f['fp_wouldwin'] > 0 and nb >= 3:
        cond.append('- That folder\'s cards would have beaten every player who stayed in.')
    if f['checkdown_best'] > 0:
        cond.append('- On the last street a member checked holding the best hand still in, with '
                    'the other member still in the pot.')
    if f['iso_clean'] > 0:
        cond.append('- A member made the last preflop raise, the other folded to a member\'s bet, '
                    'and at most one non-member saw the flop.')
    if f['passive_winner'] > 0:
        cond.append('- A member won at showdown with the best hand there without betting or '
                    'raising after the flop, the other having put in at least 2 bb.')
    if f['pay_partner_bb'] > 0:
        cond.append(f'- A member put {nf(float(f["pay_partner_bb"]))} bb in after the flop calling '
                    f'the other member, who finished up.')
    if f['coldcall_partner_bb'] > 0:
        cond.append(f'- A member cold-called {nf(float(f["coldcall_partner_bb"]))} bb facing the '
                    f'other member.')
    if f['call_partner_dominated_bb'] > 0:
        cond.append(f'- A member called {nf(float(f["call_partner_dominated_bb"]))} bb facing the '
                    f'other member while holding the weaker of the two hands.')
    if f['iso_weak_agg_behind'] > .6:
        cond.append('- A member bet or raised with a weak holding while the other still had to '
                    'act behind.')
    if f['oof_clearway'] > 0:
        cond.append('- A member folded to a non-member\'s bet holding the best hand still in, '
                    'with the other member still in the pot.')
    if f['tr_loser_share'] > 0:
        cond.append(f'- The member who finished down supplied {pc(float(f["tr_loser_share"]))} of '
                    f'the final pot and the other collected at least half of it.')
    # always-available fillers so the block is 10 lines on essentially every hand
    cond.append('- Both members bet or raised at some point.' if f['both_agg_live'] > 0
                else '- At most one of the two members bet or raised.')
    vpip = [any(x['player_id'] == m and x['action'] not in ('fold', 'check')
                for x in ctx.a.get(hand_id, [])) for m in (ma, mb)]
    cond.append({2: '- Both members put chips in voluntarily.',
                 1: '- Only one of the two members put chips in voluntarily.',
                 0: '- Neither member put chips in voluntarily beyond the blinds.'}[sum(vpip)])
    nnf = 6 - len([p for p in order if rows[p]['folded']])
    cond.append('- Only one player did not fold.' if nnf == 1 else
                f'- {nnf} of the six players did not fold.')
    return always + cond[:3]


def render(ctx: HandContext, pair_id: str, hand_id: str, a_is_player_1: bool) -> str:
    """Deterministic: the same (pair_id, hand_id, a_is_player_1) always yields the same text."""
    p1, p2 = ctx.members[pair_id]
    ma, mb = (p1, p2) if a_is_player_1 else (p2, p1)
    h = ctx.h[hand_id]
    bb = float(h['big_blind'])
    sb = h['small_blind'] / bb
    rows, order, alias, pos = _aliases(ctx, hand_id, ma, mb)
    acts = ctx.a.get(hand_id, [])
    board = (h['board_cards'] or '').split()
    nb = len(board)
    last_street = acts[-1]['street'] if acts else 'preflop'

    L = ['One six-handed no-limit hold\'em hand. Two players are marked A and B, the other four '
         'O1-O4.',
         'All amounts are in big blinds (bb); "odds" is the call price as a share of the pot '
         'after calling.', '',
         'Table',
         f'  Blinds {nf(sb)} / 1 bb, so the pot starts at {nf(sb + 1)} bb.',
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
    L.append('')
    L.append('Action')
    cur = None
    for r in acts:
        if r['street'] != cur:
            cur = r['street']
            L.append(f'  -- {STREET_TITLE[cur]} --')
        L.append(_action_line(r, alias, pos, bb))
    if nb > NBOARD[SIDX[last_street]]:
        L.append(f'  -- betting ended on the {last_street}; a player was all in and the rest of '
                 f'the board was dealt --')
    L.append('')
    L.append('Result')
    pot = h['final_pot'] / bb
    sdn = h['players_at_showdown']
    sd = [alias[p] for p in order if rows[p]['went_to_showdown']]
    L.append(f'  Final pot {nf(pot)} bb. ' +
             (f'Showdown between {", ".join(sd)}.' if sdn >= 2 else 'No showdown.'))
    coll = [(alias[p], (rows[p]['net_chips'] + rows[p]['total_contribution']) / bb)
            for p in order if rows[p]['net_chips'] + rows[p]['total_contribution'] > 0]
    L.append('  Collected: ' + ', '.join(f'{a} {nf(v)}' for a, v in coll) +
             ' (includes any uncalled bet returned).')
    L.append('  Net (bb): ' + ', '.join(f'{alias[p]} {sf(rows[p]["net_chips"] / bb)}'
                                        for p in order) + '.')
    L.append('')
    L.append('Derived facts (computed from the hand record)')
    L += _facts(ctx, pair_id, hand_id, ma, mb, order, alias, rows, nb, bb)
    L.append(INSTRUCTION)
    return '\n'.join(L)


def make_renderer(keys: pl.DataFrame):
    """Return (ctx, fn) where fn(pair_id, hand_id, a_is_player_1) -> prompt text."""
    ctx = HandContext(keys)
    return ctx, lambda pair_id, hand_id, a_is_player_1: render(ctx, pair_id, hand_id,
                                                               a_is_player_1)


# ---------------------------------------------------------------------------- eyeball check


def _eyeball(n: int = 5) -> None:
    """Print n prompts next to their raw action rows for manual verification."""
    import numpy as np
    sys.path.insert(0, CV + r'\src')
    from evidence_aug import step254_eval as E
    h, _ = E.load()
    h = h.select('pair_id', 'hand_id', 'label', 'behavior_family', 'listed')
    rng = np.random.default_rng(1)
    pick = pl.concat([h.filter(pl.col('listed') == 1).sample(3, seed=1),
                      h.filter(pl.col('label') == 0).sample(2, seed=1)])
    ctx, fn = make_renderer(pick)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_fmt_str_lengths(20)
    for i, r in enumerate(pick.iter_rows(named=True)):
        txt = fn(r['pair_id'], r['hand_id'], i % 2 == 0)
        print('=' * 100)
        print(f'#{i}  label={r["label"]} family={r["behavior_family"]} listed={r["listed"]}  '
              f'a_is_player_1={i % 2 == 0}  chars={len(txt)}  est_tokens={est_tokens(txt):.0f}')
        print('-' * 100)
        print(txt)
        print('-' * 100)
        print('RAW seats:')
        print(pl.DataFrame(ctx.s[r['hand_id']]).select(
            'player_id', 'seat_no', 'starting_stack', 'hole_card_1', 'hole_card_2',
            'total_contribution', 'net_chips', 'folded', 'went_to_showdown', 'won_share'))
        print('RAW hand:', {k: v for k, v in ctx.h[r['hand_id']].items() if k != 'hand_id'})
        print('RAW actions:')
        print(pl.DataFrame(ctx.a[r['hand_id']]).select(
            'action_no', 'street', 'player_id', 'action', 'amount', 'amount_to', 'pot_before',
            'to_call', 'players_active'))
        print('members (player_1, player_2):', ctx.members[r['pair_id']],
              'p1_is_A(step254):', ctx.p1_is_A[(r['pair_id'], r['hand_id'])])
        det = fn(r['pair_id'], r['hand_id'], i % 2 == 0)
        assert det == txt, 'renderer is not deterministic'


if __name__ == '__main__':
    _eyeball(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
