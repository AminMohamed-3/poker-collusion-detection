# -*- coding: utf-8 -*-
"""Turn-protocol dealer: the CASINO for agent-played poker (synthetic track).

Code deals, enforces, and records. Agents are the PLAYERS: every poker
decision comes from the seat agent via files. This module NEVER chooses a
poker action and NEVER invents a cheating event. The only dealer-imposed
action is the protocol fallback (check-when-free / fold-when-facing-a-bet)
after a seat burns its one retry with illegal submissions.

Card/chip core is REUSED from engine.py (deck, scoring, blinds, settlement,
schema column orders). Decisions stay OUT of code: the dealer validates and
applies seat-supplied {action, amount} against stack/to_call/players_active.

File protocol (shared place: data/synthetic/tables/<table>/):
  dealer -> hand_{n:04d}_step_{k:03d}_request_seat{s}.json  (public + OWN hole)
  seat   -> hand_{n:04d}_step_{k:03d}_action_seat{s}.json   ({action,amount,reason})
  dealer -> hand_{n:04d}_step_{k:03d}_rejected_seat{s}_try{t}.json (on illegal)
  dealer -> audit.jsonl (every request/action/reject/retry/fallback/hand_close)
  dealer -> hands.json / seats.json / actions.json (schema-exact row dicts)

Self-test mode: pass player_fn(seat, request, attempt, last_error) which the
dealer calls, writes the returned action to the action file, then reads it
back through the same intake path as live play. Live mode: player_fn=None,
the dealer blocks until the seat agent writes the action file.

CPU only, stdlib only, utf-8, fixed seeds, no wall-clock in files.
"""
import json
import os
import time

try:
    from engine import (SB, BB, START_STACK, N_SEATS, STREETS, BOARD_LEN,
                        HAND_COLS, SEAT_COLS, ACTION_COLS, PLAYER_COLS,
                        BASE_EPOCH_US, HAND_SPACING_US, MAX_ACTIONS_PER_HAND,
                        MAX_AGGRESSIONS_PER_STREET, new_deck, score7, stream,
                        hex_id, TableEngine)
except ImportError:  # pragma: no cover - alternate sys.path layouts
    from synthetic.engine import (SB, BB, START_STACK, N_SEATS, STREETS,
                                  BOARD_LEN, HAND_COLS, SEAT_COLS,
                                  ACTION_COLS, PLAYER_COLS, BASE_EPOCH_US,
                                  HAND_SPACING_US, MAX_ACTIONS_PER_HAND,
                                  MAX_AGGRESSIONS_PER_STREET, new_deck, score7,
                                  stream, hex_id, TableEngine)

PROTOCOL = "turn-file-protocol/v1"
ALLOWED = ("fold", "check", "call", "bet", "raise", "all_in")
MAX_REASON = 200


def legal_set(to_call, aggressions):
    """Legal action NAMES here (same sets as engine directors see)."""
    if aggressions >= MAX_AGGRESSIONS_PER_STREET:
        if to_call == 0:
            return frozenset({"check", "all_in"})
        return frozenset({"fold", "call", "all_in"})
    if to_call == 0:
        return frozenset({"check", "bet", "all_in"})
    return frozenset({"fold", "call", "raise", "all_in"})


def fallback_action(to_call):
    """Protocol fallback after one failed retry. A rule, not a strategy."""
    if to_call == 0:
        return "check"
    return "fold"


def clean_reason(reason):
    """One line, <=200 chars. Dealer never inspects brief content."""
    if reason is None:
        return ""
    line = str(reason).replace("\r", " ").replace("\n", " ").strip()
    return line[:MAX_REASON]


def validate_action(action, amount, to_call, stack, committed, level, legal):
    """Returns (ok, reason, label). label coerces short-stack bets to all_in.

    Mirrors validator.py V3 class rules exactly.
    """
    if action not in ALLOWED:
        return False, "unknown action %r; allowed=%s" % (action, sorted(legal)), None
    if isinstance(amount, bool) or not isinstance(amount, int):
        return False, "amount must be an integer chip count, got %r" % (amount,), None
    if not (0 <= amount <= stack):
        return False, "amount %r outside [0, stack=%d]" % (amount, stack), None
    if action not in legal:
        return False, "action %r not legal here; legal=%s (to_call=%d)" % (
            action, sorted(legal), to_call), None
    if amount == stack and amount > 0 and action in ("call", "bet", "raise", "all_in"):
        return True, "", "all_in"  # engine-consistent short-stack coercion
    if action == "check":
        if amount == 0 and to_call == 0:
            return True, "", "check"
        return False, "check needs amount 0 with to_call 0 (to_call=%d)" % to_call, None
    if action == "fold":
        if amount == 0 and to_call > 0:
            return True, "", "fold"
        return False, "fold needs amount 0 facing a bet (to_call=%d)" % to_call, None
    if action == "call":
        if amount > 0 and amount == min(to_call, stack):
            return True, "", "call"
        return False, "call amount must equal min(to_call=%d, stack=%d)" % (
            to_call, stack), None
    if action == "bet":
        if not (to_call == 0 and amount > 0):
            return False, "bet needs to_call 0 and amount>0 (to_call=%d)" % to_call, None
        if committed + amount <= level:
            return False, "bet must raise the level (amount_to=%d <= level=%d)" % (
                committed + amount, level), None
        return True, "", "bet"
    if action == "raise":
        if not (to_call > 0 and amount > 0):
            return False, "raise needs to_call>0 and amount>0 (to_call=%d)" % to_call, None
        if committed + amount <= level:
            return False, "raise must raise the level (amount_to=%d <= level=%d)" % (
                committed + amount, level), None
        return True, "", "raise"
    # all_in (non-coerced path: full-stack already returned above)
    if amount == stack and amount > 0:
        return True, "", "all_in"
    return False, "all_in must commit the full stack (%d)" % stack, None


def _dump(path, obj):
    # Preserve schema column order (validator V1 checks names AND order).
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=False, ensure_ascii=False, indent=2)
        f.write("\n")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _audit(audit_f, event):
    audit_f.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")


def _wait_for_action(path, timeout_s, poll_s=0.25):
    t0 = time.time()
    while True:
        if os.path.exists(path):
            try:
                obj = _load(path)
            except ValueError:
                pass  # half-written file; keep polling
            else:
                if isinstance(obj, dict) and "action" in obj and "amount" in obj:
                    return obj
        if time.time() - t0 > timeout_s:
            raise TimeoutError("no valid action file: %s" % path)
        time.sleep(poll_s)


class TurnDealer:
    """Deterministic turn-protocol dealer for one 6-max table."""

    def __init__(self, seed, table_idx=0, n_hands=60, roster=None, table_dir=None):
        self.seed = seed
        self.table_idx = table_idx
        self.n_hands = n_hands
        trng = stream("table", seed, table_idx)
        self.table_id = hex_id(trng, "T", 8)
        prng = stream("roster", seed, table_idx)
        if roster is None:
            roster = [hex_id(prng, "U", 12) for _ in range(N_SEATS)]
        self.roster = list(roster)
        self.table_dir = table_dir

    # -- roster attributes: same derivation as engine.players_table --
    def players_table(self):
        return TableEngine(self.seed, self.table_idx, self.n_hands,
                           roster=self.roster).players_table()

    def intake_step(self, seat, request, audit_f, player_fn=None,
                    timeout_s=86400, rig=None):
        """Write request, take ONE decision (retry+fallback inside).

        Returns (label, amount, reason, step_stat). step_stat counts
        rejections/retries/fallbacks for this step. rig, if given, is a
        test-only callable(seat, request, attempt) -> action-dict-or-None
        that overrides player_fn for that attempt (self-test scaffolding).
        """
        tdir = self.table_dir
        hid_n = request["hand_idx"]
        step = request["step"]
        req_path = os.path.join(
            tdir, "hand_%04d_step_%03d_request_seat%d.json" % (hid_n, step, seat))
        _dump(req_path, request)
        _audit(audit_f, {"event": "request", "hand_idx": hid_n,
                         "hand_id": request["hand_id"], "step": step,
                         "seat": seat, "street": request["street"],
                         "to_call": request["to_call"],
                         "legal": sorted(request["legal_actions"])})
        last_error = None
        for attempt in (0, 1):
            act_path = os.path.join(
                tdir, "hand_%04d_step_%03d_action_seat%d.json" % (hid_n, step, seat))
            if os.path.exists(act_path):
                os.remove(act_path)
            if rig is not None:
                forced = rig(seat, request, attempt)
                if forced is not None:
                    supplied = dict(forced)
                elif player_fn is not None:
                    supplied = dict(player_fn(seat, dict(request), attempt, last_error))
                else:
                    supplied = _wait_for_action(act_path, timeout_s)
                    supplied = dict(supplied)
                if "action" not in supplied or "amount" not in supplied:
                    supplied = dict(supplied)
                supplied.setdefault("seat_no", seat)
                supplied.setdefault("hand_id", request["hand_id"])
                supplied.setdefault("step", step)
                supplied.setdefault("attempt", attempt)
                _dump(act_path, {"protocol": PROTOCOL,
                                 "table_id": self.table_id,
                                 "hand_id": request["hand_id"],
                                 "hand_idx": hid_n, "step": step,
                                 "seat_no": seat,
                                 "player_id": self.roster[seat],
                                 "action": supplied.get("action"),
                                 "amount": supplied.get("amount"),
                                 "reason": clean_reason(supplied.get("reason", "")),
                                 "attempt": attempt})
            elif player_fn is not None:
                supplied = dict(player_fn(seat, dict(request), attempt, last_error))
                _dump(act_path, {"protocol": PROTOCOL,
                                 "table_id": self.table_id,
                                 "hand_id": request["hand_id"],
                                 "hand_idx": hid_n, "step": step,
                                 "seat_no": seat,
                                 "player_id": self.roster[seat],
                                 "action": supplied.get("action"),
                                 "amount": supplied.get("amount"),
                                 "reason": clean_reason(supplied.get("reason", "")),
                                 "attempt": attempt})
            else:
                supplied = dict(_wait_for_action(act_path, timeout_s))
            action = supplied.get("action")
            amount = supplied.get("amount")
            reason = clean_reason(supplied.get("reason", ""))
            ok, err, label = validate_action(
                action, amount, request["to_call"],
                request["stack_remaining"], request["street_committed"],
                request["level"], request["legal_actions"])
            _audit(audit_f, {"event": "action", "hand_idx": hid_n,
                             "hand_id": request["hand_id"], "step": step,
                             "seat": seat, "attempt": attempt,
                             "action": action, "amount": amount,
                             "reason": reason,
                             "accepted": bool(ok),
                             "label": label or "",
                             "reject_reason": "" if ok else err})
            if ok:
                return label, amount, reason, {"rejections": attempt,
                                               "retries": attempt,
                                               "fallbacks": 0}
            rej_path = os.path.join(
                tdir, "hand_%04d_step_%03d_rejected_seat%d_try%d.json" % (
                    hid_n, step, seat, attempt))
            _dump(rej_path, {"hand_id": request["hand_id"], "hand_idx": hid_n,
                             "step": step, "seat_no": seat, "attempt": attempt,
                             "received_action": action, "received_amount": amount,
                             "reason": err})
            _audit(audit_f, {"event": "reject", "hand_idx": hid_n,
                             "hand_id": request["hand_id"], "step": step,
                             "seat": seat, "attempt": attempt, "reason": err})
            last_error = err
            request = dict(request, attempt=attempt + 1,
                           last_rejection={"reason": err,
                                           "received_action": action,
                                           "received_amount": amount})
        fb = fallback_action(request["to_call"])
        _audit(audit_f, {"event": "fallback", "hand_idx": hid_n,
                         "hand_id": request["hand_id"], "step": step,
                         "seat": seat, "action": fb, "amount": 0,
                         "reason": "fallback after 2 illegal attempts: %s" % last_error})
        return fb, 0, "dealer-fallback %s (%s)" % (fb, last_error), {
            "rejections": 2, "retries": 1, "fallbacks": 1}

    def play_hand_protocol(self, hand_idx, audit_f, player_fn=None,
                           timeout_s=86400, rig=None):
        """Play one hand through the file protocol. Returns (hand, seats, acts)."""
        hrng = stream("hand", self.seed, self.table_idx, hand_idx)
        hand_id = hex_id(hrng, "H", 15)
        button = hand_idx % N_SEATS
        deck = new_deck(hrng)
        hole = {s: [deck.pop(), deck.pop()] for s in range(N_SEATS)}
        community = [deck.pop() for _ in range(5)]

        stacks = {s: START_STACK for s in range(N_SEATS)}
        contributed = {s: 0 for s in range(N_SEATS)}
        sb_seat = (button + 1) % N_SEATS
        bb_seat = (button + 2) % N_SEATS
        stacks[sb_seat] -= SB
        stacks[bb_seat] -= BB
        contributed[sb_seat] += SB
        contributed[bb_seat] += BB
        pot = SB + BB

        actions = []
        history = []
        folded = set()
        street_reached = "preflop"
        stats = {"rejections": 0, "retries": 0, "fallbacks": 0}
        step = 0

        def active():
            return [s for s in range(N_SEATS) if s not in folded]

        order0 = [(button + 3 + i) % N_SEATS for i in range(N_SEATS)]
        first = {"preflop": order0,
                 "flop": [(button + 1 + i) % N_SEATS for i in range(N_SEATS)],
                 "turn": [(button + 1 + i) % N_SEATS for i in range(N_SEATS)],
                 "river": [(button + 1 + i) % N_SEATS for i in range(N_SEATS)]}
        level = BB
        for street in STREETS:
            street_reached = street
            if len(active()) <= 1:
                break
            committed_st = {s: 0 for s in range(N_SEATS)}
            if street == "preflop":
                for s in range(N_SEATS):
                    committed_st[s] = SB if s == sb_seat else (BB if s == bb_seat else 0)
                level = BB
            else:
                level = 0
            acted = set()
            aggressions = 0
            ptr = 0
            order = first[street]
            guard = 0
            while guard < MAX_ACTIONS_PER_HAND:
                guard += 1
                if len(active()) <= 1:
                    break
                need = [s for s in active()
                        if stacks[s] > 0 and (committed_st[s] != level or s not in acted)]
                if not need:
                    break
                while order[ptr % N_SEATS] not in need:
                    ptr += 1
                seat = order[ptr % N_SEATS]
                ptr += 1
                to_call = level - committed_st[seat]
                board_now = community[:BOARD_LEN[street]] if street != "preflop" else []
                request = {"protocol": PROTOCOL, "table_id": self.table_id,
                           "hand_id": hand_id, "hand_idx": hand_idx,
                           "button_seat": button, "seat_no": seat,
                           "player_id": self.roster[seat],
                           "hole": list(hole[seat]), "board": list(board_now),
                           "street": street, "pot": pot, "level": level,
                           "to_call": to_call,
                           "stack_remaining": stacks[seat],
                           "street_committed": committed_st[seat],
                           "players_active": len(active()),
                           "active_seats": list(active()),
                           "stacks": {str(s): stacks[s] for s in range(N_SEATS)},
                           "history": [{"street": h[0], "player_id": h[1],
                                        "action": h[2], "amount": h[3]} for h in history],
                           "legal_actions": sorted(legal_set(to_call, aggressions)),
                           "small_blind": SB, "big_blind": BB,
                           "starting_stack": START_STACK, "step": step,
                           "attempt": 0, "last_rejection": None}
                label, amount, _reason, st = self.intake_step(
                    seat, request, audit_f, player_fn, timeout_s, rig)
                stats["rejections"] += st["rejections"]
                stats["retries"] += st["retries"]
                stats["fallbacks"] += st["fallbacks"]
                level_before = level
                committed_before = committed_st[seat]
                stack_before = stacks[seat]
                row = {"street": street, "player_id": self.roster[seat],
                       "action": label, "amount": amount,
                       "amount_to": committed_before + amount,
                       "pot_before": pot, "stack_before": stack_before,
                       "to_call": to_call, "players_active": len(active())}
                if label in ("bet", "raise", "all_in") and row["amount_to"] > level_before:
                    level = row["amount_to"]
                    aggressions += 1
                    acted = {seat}
                else:
                    if label == "fold":
                        folded.add(seat)
                    acted.add(seat)
                pot += amount
                committed_st[seat] = committed_before + amount
                contributed[seat] += amount
                stacks[seat] -= amount
                actions.append(row)
                history.append((street, self.roster[seat], label, amount))
                step += 1
                if len(active()) <= 1:
                    break
            if len(active()) <= 1:
                break

        board_out = community[:BOARD_LEN[street_reached]]
        if len(active()) == 1:
            winners = list(active())
            showdown = False
        else:
            if all(stacks[s] == 0 for s in active()):
                street_reached = "river"
                board_out = community[:5]
            scored = sorted(((score7(hole[s], board_out), s) for s in active()),
                            reverse=True)
            best = scored[0][0]
            winners = sorted(s for sc, s in scored if sc == best)
            showdown = True
        n_win = len(winners)
        share, rem = divmod(pot, n_win)
        won = {s: 0 for s in range(N_SEATS)}
        for i, s in enumerate(winners):
            won[s] = share + (1 if i < rem else 0)
        at_showdown = len(active()) if showdown else 0

        hand_row = {"hand_id": hand_id, "table_id": self.table_id,
                    "started_at": BASE_EPOCH_US + hand_idx * HAND_SPACING_US,
                    "phase": "synthetic", "button_seat": button,
                    "small_blind": SB, "big_blind": BB,
                    "board_cards": " ".join(board_out), "final_pot": pot,
                    "players_dealt": N_SEATS, "players_at_showdown": at_showdown}
        seat_rows = []
        for s in range(N_SEATS):
            seat_rows.append({"hand_id": hand_id, "player_id": self.roster[s],
                              "seat_no": s, "starting_stack": START_STACK,
                              "hole_card_1": hole[s][0], "hole_card_2": hole[s][1],
                              "total_contribution": contributed[s],
                              "net_chips": won[s] - contributed[s],
                              "folded": s in folded,
                              "went_to_showdown": showdown and s in active(),
                              "won_share": (won[s] / pot) if pot > 0 else 0.0})
        for i, r in enumerate(actions):
            full = {"hand_id": hand_id, "action_no": i}
            full.update(r)
            actions[i] = full
        _audit(audit_f, {"event": "hand_close", "hand_idx": hand_idx,
                         "hand_id": hand_id, "final_pot": pot,
                         "winners": sorted(winners), "showdown": showdown,
                         "n_actions": len(actions),
                         "rejections": stats["rejections"],
                         "fallbacks": stats["fallbacks"]})
        return hand_row, seat_rows, actions, stats

    def run_table(self, player_fn=None, timeout_s=86400, rig=None,
                  audit_name="audit.jsonl"):
        """Run all hands. Appends schema-exact rows to hands/seats/actions.json."""
        os.makedirs(self.table_dir, exist_ok=True)
        hands, seats, actions = [], [], []
        totals = {"rejections": 0, "retries": 0, "fallbacks": 0,
                  "requests": 0, "hand_close": 0}
        with open(os.path.join(self.table_dir, audit_name), "w",
                  encoding="utf-8") as audit_f:
            for h in range(self.n_hands):
                hr, sr, ar, st = self.play_hand_protocol(
                    h, audit_f, player_fn, timeout_s, rig)
                hands.append(hr)
                seats.extend(sr)
                actions.extend(ar)
                totals["rejections"] += st["rejections"]
                totals["retries"] += st["retries"]
                totals["fallbacks"] += st["fallbacks"]
                totals["requests"] += len(ar)
                totals["hand_close"] += 1
        for name, rows in (("hands", hands), ("seats", seats), ("actions", actions)):
            _dump(os.path.join(self.table_dir, name + ".json"), rows)
        _dump(os.path.join(self.table_dir, "players.json"), self.players_table())
        _dump(os.path.join(self.table_dir, "protocol_stats.json"),
              {"protocol": PROTOCOL, "seed": self.seed,
               "table_idx": self.table_idx, "n_hands": self.n_hands,
               "table_id": self.table_id, "roster": self.roster, **totals})
        return hands, seats, actions, self.players_table(), totals
