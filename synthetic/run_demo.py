# -*- coding: utf-8 -*-
"""Demo: 1 table x 200 hands -> data/synthetic/demo_*.parquet + validation.

Run (CPU only):  python run_demo.py   (from synthetic/)
Seeds: demo seed 7329 (project model seed), table 0.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTDIR = os.path.abspath(os.path.join(HERE, "..", "data", "synthetic"))

from engine import TableEngine, to_parquet
from validator import validate_all, check_determinism
from directors import PassthroughDirector

SEED = 7329
N_HANDS = 200


def main():
    t0 = time.time()
    eng = TableEngine(SEED, 0, N_HANDS,
                      directors={s: PassthroughDirector() for s in range(6)})
    hands, seats, actions = eng.run_table()
    players = eng.players_table()
    t1 = time.time()
    paths = to_parquet(hands, seats, actions, players, OUTDIR)
    t2 = time.time()
    ok, msgs = validate_all(hands, seats, actions, players)
    det_ok, det_msg = check_determinism(SEED, 0, N_HANDS)
    kinds = {}
    for r in actions:
        kinds[r["action"]] = kinds.get(r["action"], 0) + 1
    n_sd = sum(1 for h in hands if h["players_at_showdown"] > 0)
    print("hands=%d seats=%d actions=%d players=%d" % (len(hands), len(seats),
                                                       len(actions), len(players)))
    print("simulate_s=%.2f write_s=%.2f" % (t1 - t0, t2 - t1))
    for name, p in sorted(paths.items()):
        print("%s -> %s (%d bytes)" % (name, p, os.path.getsize(p)))
    print("VALIDATOR:", "PASS" if ok else "FAIL", "|", msgs[-1] if msgs else "")
    for m in msgs[:-1][:5]:
        print("  ", m)
    print("DETERMINISM:", det_msg)
    print("showdown_hands=%d/%d action_mix=%s" % (n_sd, len(hands), kinds))
    print("hooks: calls=%d overrides=%d fallbacks=%d"
          % (eng.hook_calls, eng.hook_overrides, eng.hook_fallbacks))
    if not (ok and det_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
