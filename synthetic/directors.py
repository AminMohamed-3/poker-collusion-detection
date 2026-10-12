# -*- coding: utf-8 -*-
"""Director hook API for the synthetic world engine (family-director lane).

Contract (clean separation: directors NEVER touch deck, pot, stacks, or cards):
- The engine owns all game logic, legality, sizing, and accounting.
- At each decision point for a seat registered in engine.directors, the engine
  builds a DecisionContext (read-only snapshot) plus `legal` (frozenset of
  action NAMES legal here) and calls director.decide(ctx, legal).
- Return one of the legal names, or None (= abstain -> engine default policy).
- Any other return value (illegal name, exception) -> engine falls back to the
  default normal-play policy and counts a hook_fallback. Sizing is ALWAYS done
  by the engine from the chosen action name.
- before_hand(ctx_hand) is an optional hook called once per hand per director
  (e.g. to elect roles); it must be side-effect free w.r.t. game state.

Coercion table (what the engine does with a director's return):
  to_call == 0 : legal = {check, bet, all_in}
  to_call >  0 : legal = {fold, call, raise, all_in}
  capped street (4+ aggressions): {check, all_in} / {fold, call, all_in}
  'all_in' always means "commit remaining stack", any spot with chips in front.
Illegal/unknown/exception -> default policy (counted, never crashes a run).

Family directors (DT/SP/CI lane) implement THIS interface. Guidance from the
charter (Claude OBS §§6-9): emit LISTED-LEVEL events, especially cheap folds of
strong holdings facing the partner (fold-benefit ~2.8σ, early-hand bias) --
NOT surprising calls (ca_residual_2 was the prior failure mode).
"""


class DecisionContext:
    """Read-only snapshot for one decision point. No mutators by convention."""

    __slots__ = ("hand_id", "hand_idx", "street", "seat_no", "player_id",
                 "partner_ids", "hole", "board", "pot", "level", "to_call",
                 "stack_remaining", "street_committed", "players_active",
                 "history", "rng")

    def __init__(self, hand_id, hand_idx, street, seat_no, player_id,
                 partner_ids, hole, board, pot, level, to_call,
                 stack_remaining, street_committed, players_active, history,
                 rng):
        self.hand_id = hand_id
        self.hand_idx = hand_idx
        self.street = street
        self.seat_no = seat_no
        self.player_id = player_id
        self.partner_ids = partner_ids      # active opponents (ids only)
        self.hole = hole                    # own 2 cards, e.g. ('As','Kd')
        self.board = board                  # community so far (tuple, may be ())
        self.pot = pot
        self.level = level                  # current street bet level
        self.to_call = to_call
        self.stack_remaining = stack_remaining
        self.street_committed = street_committed
        self.players_active = players_active
        self.history = history              # ((street,pid,action,amount),...)
        self.rng = rng                      # per-decision deterministic stream


class HandContext:
    """Snapshot passed to before_hand (role election, read-only)."""

    __slots__ = ("hand_id", "hand_idx", "button_seat", "roster", "hole_by_seat")

    def __init__(self, hand_id, hand_idx, button_seat, roster, hole_by_seat):
        self.hand_id = hand_id
        self.hand_idx = hand_idx
        self.button_seat = button_seat
        self.roster = roster
        self.hole_by_seat = hole_by_seat    # own table view; directors see own cards only


class FamilyDirector:
    """Base class. Subclass per family; override decide (and optionally before_hand)."""
    name = "base"

    def before_hand(self, hctx):
        return None

    def decide(self, ctx, legal):
        return None  # abstain -> default normal-play policy


class PassthroughDirector(FamilyDirector):
    """Exercised in the demo: valid hook traffic, zero behavior change."""
    name = "passthrough"

    def decide(self, ctx, legal):
        return None


class AlwaysCheckFoldDirector(FamilyDirector):
    """Test/example director: checks when free, else folds (always legal)."""
    name = "checkfold"

    def decide(self, ctx, legal):
        if "check" in legal:
            return "check"
        return "fold"
