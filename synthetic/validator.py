# -*- coding: utf-8 -*-
"""Built-in accounting validator for synthetic tables (verifier-lane partner).

Checks, in charter kill-gate-1 order:
  V1 schema-exact columns (names AND order) for hands/seats/actions/players.
  V2 chip conservation per hand: sum(net)==0, sum(contrib)==final_pot,
      won_shares sum to ~1 and imply the same integer pot split.
  V3 action legality by full re-simulation: amount vs to_call/stack_before,
      players_active consistency, pot/stack chaining, class rules
      (check/fold/call/bet/raise/all_in), amount_to == street-after.
  V4 seat/phase consistency: 6 seats 0-5, valid distinct cards (12 hole +
      board from one 52-deck), board length in {0,3,4,5}, showdown flags,
      street order, action_no sequence, blind/pot opener (pot_before[0]==SB+BB).
  V5 determinism: same (seed, table, n_hands) re-run -> identical row dicts.
  V6 ID hygiene: opaque hex shapes H+15/T+8/U+12 (no format leakage).

validate_all(hands, seats, actions, players) -> (ok, [messages]).
check_determinism(seed, table_idx, n_hands) -> (ok, message).
"""
import re
from collections import defaultdict

try:
    from engine import (HAND_COLS, SEAT_COLS, ACTION_COLS, PLAYER_COLS, SB, BB,
                        START_STACK, N_SEATS, TableEngine)
except ImportError:  # pragma: no cover - alternate sys.path layouts
    from synthetic.engine import (HAND_COLS, SEAT_COLS, ACTION_COLS, PLAYER_COLS,
                                  SB, BB, START_STACK, N_SEATS, TableEngine)

CARD_RE = re.compile(r"^[2-9TJQKA][shdc]$")
HID_RE = re.compile(r"^H[0-9A-F]{15}$")
TID_RE = re.compile(r"^T[0-9A-F]{8}$")
UID_RE = re.compile(r"^U[0-9A-F]{12}$")
STREET_ORD = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def _cols_ok(name, rows, want, msgs):
    if not rows:
        msgs.append("FAIL V1 %s: empty" % name)
        return False
    got = list(rows[0].keys())
    if got != want:
        msgs.append("FAIL V1 %s: cols %s != %s" % (name, got, want))
        return False
    return True


def validate_all(hands, seats, actions, players):
    msgs = []
    ok = True

    def fail(m):
        msgs.append("FAIL " + m)

    for name, rows, want in (("hands", hands, HAND_COLS), ("seats", seats, SEAT_COLS),
                             ("actions", actions, ACTION_COLS),
                             ("players", players, PLAYER_COLS)):
        if not _cols_ok(name, rows, want, msgs):
            ok = False
    if not ok:
        return False, msgs

    # ---- V6 id hygiene ----
    for h in hands:
        if not HID_RE.match(h["hand_id"]) or not TID_RE.match(h["table_id"]):
            fail("V6 bad hand/table id %s %s" % (h["hand_id"], h["table_id"]))
            ok = False
            break
    for p in players:
        if not UID_RE.match(p["player_id"]):
            fail("V6 bad player id %s" % p["player_id"])
            ok = False
            break

    seats_by = defaultdict(list)
    for s in seats:
        seats_by[s["hand_id"]].append(s)
    acts_by = defaultdict(list)
    for a in actions:
        acts_by[a["hand_id"]].append(a)

    tids = {h["table_id"] for h in hands}
    if len(tids) != 1:
        fail("V4 table_ids != 1: %s" % sorted(tids)[:3])
        ok = False

    for h in hands:
        hid = h["hand_id"]
        # ---- V4 seats ----
        sr = seats_by.get(hid, [])
        if len(sr) != N_SEATS or sorted(s["seat_no"] for s in sr) != list(range(N_SEATS)):
            fail("V4 %s: seats != 0..5" % hid)
            ok = False
            continue
        if h["phase"] not in ("synthetic",):
            fail("V4 %s: phase %r" % (hid, h["phase"]))
            ok = False
        if not (0 <= h["button_seat"] < N_SEATS):
            fail("V4 %s: button %r" % (hid, h["button_seat"]))
            ok = False
        if (h["small_blind"], h["big_blind"]) != (SB, BB):
            fail("V4 %s: blinds %r" % (hid, (h["small_blind"], h["big_blind"])))
            ok = False
        # cards: 12 hole + board, valid + all distinct
        cards = [s["hole_card_1"] for s in sr] + [s["hole_card_2"] for s in sr]
        board = h["board_cards"].split() if h["board_cards"] else []
        if len(board) not in (0, 3, 4, 5):
            fail("V4 %s: board len %d" % (hid, len(board)))
            ok = False
        if any(not CARD_RE.match(c) for c in cards + board):
            fail("V4 %s: bad card token" % hid)
            ok = False
        if len(set(cards + board)) != len(cards) + len(board):
            fail("V4 %s: duplicate cards" % hid)
            ok = False
        # ---- V2 conservation ----
        if sum(s["net_chips"] for s in sr) != 0:
            fail("V2 %s: sum(net) != 0" % hid)
            ok = False
        if sum(s["total_contribution"] for s in sr) != h["final_pot"]:
            fail("V2 %s: contrib != pot" % hid)
            ok = False
        if h["final_pot"] <= 0:
            fail("V2 %s: pot <= 0" % hid)
            ok = False
        wsum = sum(s["won_share"] for s in sr)
        if abs(wsum - 1.0) > 1e-9:
            fail("V2 %s: won_share sums %r" % (hid, wsum))
            ok = False
        for s in sr:
            if not (0.0 <= s["won_share"] <= 1.0):
                fail("V2 %s: won_share range" % hid)
                ok = False
                break
            if s["won_share"] > 0 and s["folded"]:
                fail("V2 %s: folded winner share" % hid)
                ok = False
                break
        # showdown flags
        sd = [s for s in sr if s["went_to_showdown"]]
        if h["players_at_showdown"] != len(sd):
            fail("V4 %s: showdown count" % hid)
            ok = False
        for s in sr:
            if s["went_to_showdown"] and s["folded"]:
                fail("V4 %s: folded+showdown" % hid)
                ok = False
                break
        if sd and len(sd) < 2:
            fail("V4 %s: lone showdown" % hid)
            ok = False
        # ---- V3 action re-simulation ----
        ar = sorted(acts_by.get(hid, []), key=lambda r: r["action_no"])
        if [r["action_no"] for r in ar] != list(range(len(ar))):
            fail("V3 %s: action_no not 0..n-1" % hid)
            ok = False
            continue
        if not ar:
            fail("V3 %s: no actions" % hid)
            ok = False
            continue
        if ar[0]["pot_before"] != SB + BB:
            fail("V3 %s: opener pot %r" % (hid, ar[0]["pot_before"]))
            ok = False
        stacks = {s["seat_no"]: START_STACK for s in sr}
        pid2seat = {s["player_id"]: s["seat_no"] for s in sr}
        spent = {s["seat_no"]: 0 for s in sr}
        # blinds are forced contributions, not logged actions: prime stacks/spent
        sb_seat0 = (h["button_seat"] + 1) % N_SEATS
        bb_seat0 = (h["button_seat"] + 2) % N_SEATS
        stacks[sb_seat0] -= SB
        stacks[bb_seat0] -= BB
        spent[sb_seat0] += SB
        spent[bb_seat0] += BB
        street_of = None
        level = BB
        committed = None
        folds = 0
        last_street_seen = -1
        for i, r in enumerate(ar):
            if r["street"] not in STREET_ORD:
                fail("V3 %s: bad street" % hid)
                ok = False
                break
            so = STREET_ORD[r["street"]]
            if so < last_street_seen:
                fail("V3 %s: street regress" % hid)
                ok = False
                break
            if so > last_street_seen:
                last_street_seen = so
                level = BB if r["street"] == "preflop" else 0
                committed = {s["seat_no"]: 0 for s in sr}
                if r["street"] == "preflop":
                    sb_seat = (h["button_seat"] + 1) % N_SEATS
                    bb_seat = (h["button_seat"] + 2) % N_SEATS
                    committed[sb_seat] = SB
                    committed[bb_seat] = BB
            seat = pid2seat.get(r["player_id"])
            if seat is None:
                fail("V3 %s: unknown pid" % hid)
                ok = False
                break
            exp_call = level - committed[seat]
            if r["to_call"] != exp_call:
                fail("V3 %s a%d: to_call %r != %r" % (hid, i, r["to_call"], exp_call))
                ok = False
                break
            if r["players_active"] != N_SEATS - folds:
                fail("V3 %s a%d: players_active %r" % (hid, i, r["players_active"]))
                ok = False
                break
            if r["stack_before"] != stacks[seat]:
                fail("V3 %s a%d: stack_before" % (hid, i))
                ok = False
                break
            if i > 0 and r["pot_before"] != ar[i - 1]["pot_before"] + ar[i - 1]["amount"]:
                fail("V3 %s a%d: pot chain" % (hid, i))
                ok = False
                break
            if not (0 <= r["amount"] <= r["stack_before"]):
                fail("V3 %s a%d: amount>stack" % (hid, i))
                ok = False
                break
            if r["amount_to"] != committed[seat] + r["amount"]:
                fail("V3 %s a%d: amount_to" % (hid, i))
                ok = False
                break
            a = r["action"]
            if a == "check":
                if not (r["amount"] == 0 and r["to_call"] == 0):
                    fail("V3 %s a%d: illegal check" % (hid, i))
                    ok = False
                    break
            elif a == "fold":
                if not (r["amount"] == 0 and r["to_call"] > 0):
                    fail("V3 %s a%d: illegal fold" % (hid, i))
                    ok = False
                    break
                folds += 1
            elif a == "call":
                if not (r["amount"] == min(r["to_call"], r["stack_before"]) and r["amount"] > 0):
                    fail("V3 %s a%d: illegal call" % (hid, i))
                    ok = False
                    break
            elif a == "bet":
                if not (r["to_call"] == 0 and r["amount"] > 0):
                    fail("V3 %s a%d: illegal bet" % (hid, i))
                    ok = False
                    break
                if r["amount_to"] <= level:
                    fail("V3 %s a%d: bet not raising level" % (hid, i))
                    ok = False
                    break
                level = r["amount_to"]
            elif a == "raise":
                if not (r["to_call"] > 0 and r["amount"] > 0 and r["amount_to"] > level):
                    fail("V3 %s a%d: illegal raise" % (hid, i))
                    ok = False
                    break
                level = r["amount_to"]
            elif a == "all_in":
                if not (r["amount"] == r["stack_before"] and r["amount"] > 0):
                    fail("V3 %s a%d: illegal all_in" % (hid, i))
                    ok = False
                    break
                if r["amount_to"] > level:
                    level = r["amount_to"]
            else:
                fail("V3 %s a%d: unknown action %r" % (hid, i, a))
                ok = False
                break
            committed[seat] += r["amount"]
            spent[seat] += r["amount"]
            stacks[seat] -= r["amount"]
        else:
            # per-seat contribution cross-check + final pot chain
            for s in sr:
                if spent[s["seat_no"]] != s["total_contribution"]:
                    fail("V2 %s: seat %d contrib" % (hid, s["seat_no"]))
                    ok = False
                    break
            if ar[-1]["pot_before"] + ar[-1]["amount"] != h["final_pot"]:
                fail("V2 %s: final pot chain" % hid)
                ok = False
            continue
        # duplicate-seat guard above uses sr; ensure loop completed
    if ok:
        msgs.append("PASS all: %d hands, %d seats, %d actions, %d players"
                    % (len(hands), len(seats), len(actions), len(players)))
    return ok, msgs


def check_determinism(seed, table_idx, n_hands):
    e1 = TableEngine(seed, table_idx, n_hands)
    e2 = TableEngine(seed, table_idx, n_hands)
    r1 = e1.run_table() + (e1.players_table(),)
    r2 = e2.run_table() + (e2.players_table(),)
    same = all(a == b for a, b in zip(r1, r2))
    lens = [len(x) for x in r1]
    if same:
        return True, "PASS V5 determinism: seed %r bit-exact %s" % (seed, lens)
    return False, "FAIL V5 determinism: seed %r %s" % (seed, lens)
