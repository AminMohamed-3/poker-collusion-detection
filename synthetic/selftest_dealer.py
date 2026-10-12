# -*- coding: utf-8 -*-
"""Protocol self-test (TEST SCAFFOLDING ONLY — stubs are NOT data).

Stubs below (RANDOM-legal + checkfold) exist only to exercise the dealer
file-protocol. They must NEVER be mistaken for agent play or training data.

Run (CPU only, from synthetic/):  python selftest_dealer.py
Checks: 1 table x 20 hands -> validator PASS, determinism PASS (bit-exact
re-run), illegal-reject path >=1, retry+fallback path >=1, per-seat privacy.
Fixed seed 2901 (project split seed), table 0. UTF-8, 2 threads.
"""
import json
import os
import shutil
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTROOT = os.path.abspath(os.path.join(HERE, "..", "data", "synthetic", "tables"))
DIR1 = os.path.join(OUTROOT, "dealer_selftest")
DIR2 = os.path.join(OUTROOT, "dealer_selftest_rerun")

from engine import BB, stream  # noqa: E402
from dealer import TurnDealer  # noqa: E402
from validator import validate_all  # noqa: E402

SEED = 2901
TABLE = 0
N_HANDS = 20
MODES = {0: "random", 1: "random", 2: "random", 3: "checkfold",
         4: "random", 5: "checkfold"}


def stub_action(mode, request, rng):
    """TEST SCAFFOLD: returns {action, amount, reason}. No strategy content."""
    legal = list(request["legal_actions"])
    to_call = request["to_call"]
    stack = request["stack_remaining"]
    committed = request["street_committed"]
    level = request["level"]
    pot = request["pot"]
    if mode == "checkfold":
        a = "check" if to_call == 0 else "fold"
        return {"action": a, "amount": 0, "reason": "TEST-SCAFFOLD checkfold %s" % a}
    a = rng.choices(sorted(legal),
                    weights=[{"fold": 3, "check": 3, "call": 3, "bet": 2,
                               "raise": 2, "all_in": 1}[x] for x in sorted(legal)])[0]
    if a in ("check", "fold"):
        amt = 0
    elif a == "call":
        amt = min(to_call, stack)
    elif a == "all_in":
        amt = stack
    elif a == "bet":
        amt = min(stack, max(2 * BB, (pot + 1) // 2 if pot >= 2 * BB else 2 * BB))
    else:  # raise
        target = max(level + 2 * BB, level + (pot + 1) // 2)
        amt = target - committed
        if amt <= 0 or amt >= stack:
            a, amt = "all_in", stack
    return {"action": a, "amount": amt, "reason": "TEST-SCAFFOLD random-legal %s" % a}


def make_player_fn(seed, table_idx):
    def player_fn(seat, request, attempt, last_error):
        rng = stream("stub", seed, table_idx, request["hand_idx"],
                     request["step"], seat, attempt)
        return stub_action(MODES[seat], request, rng)
    return player_fn


def make_rig():
    """TEST SCAFFOLD: force one reject+retry (hand 0) and one double-fail
    fallback (hand 1). Returns rig(seat, request, attempt) -> dict|None."""
    seen_h1 = set()

    def rig(seat, request, attempt):
        h, step = request["hand_idx"], request["step"]
        if h == 0 and step == 0 and attempt == 0:
            return {"action": "check", "amount": 0,
                    "reason": "TEST-SCAFFOLD forced illegal check"}
        if h == 1 and request["to_call"] > 0:
            if "seat" not in seen_h1:
                seen_h1.add("seat")
                seen_h1.add(seat)
            if seat in seen_h1 and step <= 3:
                # illegal on BOTH attempts -> forces reject, retry, fallback
                return {"action": "bet", "amount": request["stack_remaining"] + 50,
                        "reason": "TEST-SCAFFOLD forced over-stack bet try%d" % attempt}
        return None
    return rig


def fresh(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


def run_once(tdir, rig):
    dealer = TurnDealer(SEED, TABLE, N_HANDS, table_dir=tdir)
    res = dealer.run_table(make_player_fn(SEED, TABLE), rig=rig)
    return res  # hands, seats, actions, players, totals


def main():
    fails = []
    fresh(DIR1)
    fresh(DIR2)
    h1, s1, a1, p1, t1 = run_once(DIR1, make_rig())
    ok, msgs = validate_all(h1, s1, a1, p1)
    print("hands=%d seats=%d actions=%d players=%d" % (len(h1), len(s1), len(a1), len(p1)))
    print("VALIDATOR:", "PASS" if ok else "FAIL", "|", msgs[-1] if msgs else "")
    for m in msgs[:-1][:5]:
        print("  ", m)
    if not ok:
        fails.append("validator FAIL")
    h2, s2, a2, p2, t2 = run_once(DIR2, make_rig())
    det = (h1, s1, a1, p1) == (h2, s2, a2, p2)
    print("DETERMINISM:", "PASS bit-exact 20 hands" if det else "FAIL mismatch")
    if not det:
        fails.append("determinism FAIL")
    print("REJECT-PATH: rejections=%d retries=%d fallbacks=%d" % (
        t1["rejections"], t1["retries"], t1["fallbacks"]))
    if t1["rejections"] < 2:
        fails.append("reject path not exercised (%d)" % t1["rejections"])
    if t1["fallbacks"] < 1:
        fails.append("fallback path not exercised")
    kinds = {}
    for r in a1:
        kinds[r["action"]] = kinds.get(r["action"], 0) + 1
    n_sd = sum(1 for h in h1 if h["players_at_showdown"] > 0)
    print("showdown_hands=%d/%d action_mix=%s" % (n_sd, len(h1), kinds))
    if not (n_sd > 0 and "fold" in kinds):
        fails.append("missing showdown/fold mix")
    # per-seat privacy: own hole only, no sibling-hole key
    priv_ok = True
    by_hand = {}
    for r in s1:
        by_hand.setdefault(r["hand_id"], {})[r["seat_no"]] = (
            r["hole_card_1"], r["hole_card_2"])
    hid0 = h1[0]["hand_id"]
    for seat in range(6):
        rp = os.path.join(DIR1, "hand_0000_step_000_request_seat%d.json" % seat)
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as f:
                req = json.load(f)
            own = by_hand[hid0][seat]
            if tuple(req.get("hole", [])) != tuple(own):
                priv_ok = False
            for k in ("holes", "hole_by_seat", "partner_hole"):
                if k in req:
                    priv_ok = False
            break
    print("PRIVACY:", "PASS own-hole-only" if priv_ok else "FAIL leak")
    if not priv_ok:
        fails.append("privacy FAIL")
    audit_n = sum(1 for _ in open(os.path.join(DIR1, "audit.jsonl"),
                                  encoding="utf-8"))
    print("audit_lines=%d stats=%s" % (audit_n, t1))
    print("SELF-TEST:", "PASS" if not fails else "FAIL %s" % fails)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
