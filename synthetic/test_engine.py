# -*- coding: utf-8 -*-
"""Unit tests for the synthetic world engine (stdlib unittest, CPU only).

Run: python -m unittest test_engine -v   (from synthetic/)
Covers: conservation/legality (via validator), determinism, deck uniqueness,
director override path, illegal-override fallback, showdown presence.
"""
import unittest

try:
    from engine import TableEngine
    from validator import validate_all, check_determinism
    from directors import (FamilyDirector, PassthroughDirector,
                           AlwaysCheckFoldDirector)
except ImportError:  # pragma: no cover
    from synthetic.engine import TableEngine
    from synthetic.validator import validate_all, check_determinism
    from synthetic.directors import (FamilyDirector, PassthroughDirector,
                                     AlwaysCheckFoldDirector)

SEED = 7329


def small_run(seed=SEED, n=25, **kw):
    e = TableEngine(seed, 0, n, **kw)
    hands, seats, actions = e.run_table()
    return e, hands, seats, actions, e.players_table()


class TestWorld(unittest.TestCase):
    def test_validator_passes(self):
        e, h, s, a, p = small_run()
        ok, msgs = validate_all(h, s, a, p)
        self.assertTrue(ok, msgs[:5])

    def test_determinism(self):
        ok, msg = check_determinism(SEED, 0, 25)
        self.assertTrue(ok, msg)

    def test_deck_uniqueness_and_shapes(self):
        e, h, s, a, p = small_run(n=10)
        by = {}
        for row in s:
            by.setdefault(row["hand_id"], []).append(row)
        for hh in h:
            cards = ([r["hole_card_1"] for r in by[hh["hand_id"]]]
                     + [r["hole_card_2"] for r in by[hh["hand_id"]]]
                     + (hh["board_cards"].split() if hh["board_cards"] else []))
            self.assertEqual(len(set(cards)), len(cards))
            self.assertIn(len(hh["board_cards"].split() if hh["board_cards"] else []),
                          (0, 3, 4, 5))

    def test_showdowns_and_folds_present(self):
        e, h, s, a, p = small_run(n=60)
        self.assertGreater(sum(1 for hh in h if hh["players_at_showdown"] > 0), 0)
        self.assertGreater(sum(1 for hh in h if hh["players_at_showdown"] == 0), 0)
        kinds = {r["action"] for r in a}
        self.assertTrue({"fold", "call", "check"} <= kinds, kinds)

    def test_passthrough_hook_zero_change(self):
        e0, h0, s0, a0, p0 = small_run(seed=99, n=20)
        regs = {s: PassthroughDirector() for s in range(6)}
        e1, h1, s1, a1, p1 = small_run(seed=99, n=20, directors=regs)
        self.assertEqual((h0, s0, a0), (h1, s1, a1))
        self.assertGreater(e1.hook_calls, 0)
        self.assertEqual(e1.hook_fallbacks, 0)

    def test_checkfold_director_stays_valid(self):
        regs = {0: AlwaysCheckFoldDirector(), 3: AlwaysCheckFoldDirector()}
        e, h, s, a, p = small_run(seed=7, n=20, directors=regs)
        ok, msgs = validate_all(h, s, a, p)
        self.assertTrue(ok, msgs[:5])
        self.assertGreater(e.hook_overrides, 0)

    def test_illegal_override_falls_back(self):
        class Rogue(FamilyDirector):
            def decide(self, ctx, legal):
                return "raise"  # often illegal (e.g. facing no bet)

        regs = {s: Rogue() for s in range(6)}
        e, h, s, a, p = small_run(seed=11, n=20, directors=regs)
        ok, msgs = validate_all(h, s, a, p)
        self.assertTrue(ok, msgs[:5])
        self.assertGreater(e.hook_fallbacks, 0)


if __name__ == "__main__":
    unittest.main()
