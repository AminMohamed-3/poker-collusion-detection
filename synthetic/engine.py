# -*- coding: utf-8 -*-
"""Deterministic seeded poker table engine (synthetic-data track, world lane).

Normal-play only. Family directors inject behavior through the hook API in
``directors.py``; this module owns ALL game logic, deck, and accounting.

Conventions (mirroring real tables, verified read-only from data/*.parquet):
- 6-max, blinds SB=1 BB=2 (real medians), fresh starting stack 300 (~median 303).
- Blinds are forced contributions, NOT logged actions; first logged action has
  pot_before == SB + BB.
- Per action: amount = chips added NOW; amount_to = street commitment AFTER the
  action (uniform for check/fold/call/bet/raise/all_in); pot chains by amount;
  players_active = 6 - folds-so-far; to_call = street level - own commitment.
- Board '' when the hand ends preflop, else 3/4/5 cards for flop/turn/river end.
- IDs are opaque hex in real shapes (H+15 / T+8 / U+12, uppercase), seeded.
- phase='synthetic'. Fresh stacks every hand (documented simplification).
- Determinism: every stochastic draw comes from random.Random streams seeded by
  SHA-256(seed/table/hand[/decision]); same seed -> bit-exact row dicts.

Only stdlib is used in this module (pyarrow only in to_parquet()).
"""
import hashlib
import random

try:
    from directors import DecisionContext
except ImportError:  # pragma: no cover - alternate sys.path layouts
    from synthetic.directors import DecisionContext

SB = 1
BB = 2
START_STACK = 300
N_SEATS = 6
BASE_EPOCH_US = 1735689600000000  # 2025-01-01T00:00:00Z, micros
HAND_SPACING_US = 180000000       # 3 minutes per hand
MAX_ACTIONS_PER_HAND = 200
MAX_AGGRESSIONS_PER_STREET = 4    # bet + raises cap -> termination guarantee

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_IDX = {r: i for i, r in enumerate(RANKS)}
STREETS = ("preflop", "flop", "turn", "river")
BOARD_LEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

HAND_COLS = ["hand_id", "table_id", "started_at", "phase", "button_seat",
             "small_blind", "big_blind", "board_cards", "final_pot",
             "players_dealt", "players_at_showdown"]
SEAT_COLS = ["hand_id", "player_id", "seat_no", "starting_stack", "hole_card_1",
             "hole_card_2", "total_contribution", "net_chips", "folded",
             "went_to_showdown", "won_share"]
ACTION_COLS = ["hand_id", "action_no", "street", "player_id", "action", "amount",
               "amount_to", "pot_before", "stack_before", "to_call",
               "players_active"]
PLAYER_COLS = ["player_id", "account_age_days", "experience_hands_bucket",
               "preferred_stake", "region_bucket", "client_family"]


def stream(*parts):
    """Deterministic independent RNG stream for the given key parts."""
    h = hashlib.sha256(("/".join(str(p) for p in parts)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def hex_id(rng, prefix, n):
    return prefix + "".join(rng.choice("0123456789ABCDEF") for _ in range(n))


# ---------------------------------------------------------------- cards

def new_deck(rng):
    deck = [r + s for r in RANKS for s in SUITS]
    rng.shuffle(deck)
    return deck


def _score5(cards):
    """cards: 5 x (rank_idx, suit). Returns comparable tuple (cat, tiebreaks...)."""
    rs = sorted((c[0] for c in cards), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    uniq = sorted(set(rs), reverse=True)
    straight_high = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:  # wheel A-5
            straight_high = 3
    counts = {}
    for r in rs:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    if straight_high is not None and flush:
        return (8, straight_high)
    if groups[0][1] == 4:
        return (7, groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, groups[0][0], groups[1][0])
    if flush:
        return (5,) + tuple(rs)
    if straight_high is not None:
        return (4, straight_high)
    if groups[0][1] == 3:
        kick = sorted((r for r in rs if r != groups[0][0]), reverse=True)
        return (3, groups[0][0]) + tuple(kick)
    if groups[0][1] == 2 and groups[1][1] == 2:
        pairs = sorted((groups[0][0], groups[1][0]), reverse=True)
        kick = [r for r in rs if r not in pairs][0]
        return (2,) + tuple(pairs) + (kick,)
    if groups[0][1] == 2:
        kick = sorted((r for r in rs if r != groups[0][0]), reverse=True)
        return (1, groups[0][0]) + tuple(kick)
    return (0,) + tuple(rs)


def score7(hole, board):
    """hole: 2 cards, board: 3-5 cards (str like 'As'). Best-5 comparable score."""
    from itertools import combinations
    allc = [((RANK_IDX[c[0]], c[1])) for c in hole + board]
    best = None
    for five in combinations(allc, 5):
        s = _score5(list(five))
        if best is None or s > best:
            best = s
    return best


def chen_strength(hole):
    """Preflop holding strength in [0,1] (Chen formula / 20, clipped)."""
    r = sorted((RANK_IDX[c[0]] for c in hole), reverse=True)
    high = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(r[0], (r[0] + 2) / 2.0)
    if r[0] == r[1]:
        return min(1.0, max(5.0, high * 2.0) / 20.0)
    score = high
    if hole[0][1] == hole[1][1]:
        score += 2.0
    gap = r[0] - r[1] - 1
    if gap == 0:
        score += 1.0 if r[0] < 10 else 0.0
    elif gap == 1:
        score -= 1.0
    elif gap == 2:
        score -= 2.0
    elif gap == 3:
        score -= 4.0
    else:
        score -= 5.0
    if gap <= 1 and r[0] < 10:
        score += 1.0
    return max(0.0, min(1.0, score / 20.0))


def postflop_strength(hole, board):
    s = score7(hole, board)
    top = s[1] if len(s) > 1 else 0
    return max(0.0, min(1.0, 0.15 * chen_strength(hole) + 0.75 * (s[0] / 8.0)
                        + 0.10 * (top / 12.0)))


def bet_size(pot, stack):
    return max(2 * BB, min(stack, (pot + 1) // 2 if pot >= 2 * BB else 2 * BB))


def raise_to_level(level, pot, committed, stack):
    target = max(level + 2 * BB, level + (pot + 1) // 2)
    need = target - committed
    if need >= stack:
        return committed + stack  # short/all-in
    return target


# ---------------------------------------------------------------- engine

class TableEngine:
    """One deterministic 6-max table. directors: {seat_no: director}."""

    def __init__(self, seed, table_idx=0, n_hands=5000, roster=None, directors=None):
        self.seed = seed
        self.table_idx = table_idx
        self.n_hands = n_hands
        trng = stream("table", seed, table_idx)
        self.table_id = hex_id(trng, "T", 8)
        prng = stream("roster", seed, table_idx)
        if roster is None:
            roster = [hex_id(prng, "U", 12) for _ in range(N_SEATS)]
        self.roster = list(roster)
        self.directors = dict(directors or {})
        self.hook_calls = 0
        self.hook_overrides = 0
        self.hook_fallbacks = 0

    # -- default normal-play policy (no game logic outside this method) --
    @staticmethod
    def default_policy(strength, to_call, pot, rng, street="preflop"):
        if to_call == 0:
            base = strength * 1.0 + (0.10 if street != "preflop" else -0.05)
            p_bet = max(0.05, min(0.80, base))
            return "bet" if rng.random() < p_bet else "check"
        price = to_call / (pot + to_call + 1.0)
        if street == "preflop":
            if strength > 0.88 and rng.random() < 0.02:
                return "all_in"
            p_raise = 0.24 if strength > 0.60 else 0.07
            if rng.random() < p_raise:
                return "raise"
            p_fold = max(0.02, min(0.95, 0.87 - strength * 1.15 + price * 0.7))
        else:
            if strength > 0.90 and rng.random() < 0.02:
                return "all_in"
            p_raise = 0.14 if strength > 0.72 else 0.03
            if rng.random() < p_raise:
                return "raise"
            p_fold = max(0.02, min(0.95, 0.75 - strength * 1.1 + price * 0.8))
        if rng.random() < p_fold:
            return "fold"
        return "call"

    def _decide(self, seat, dctx, legal, strength, to_call, pot, hrng, street):
        """Hook dispatch: director override or default policy. Always legal out."""
        director = self.directors.get(seat)
        if director is not None:
            self.hook_calls += 1
            try:
                want = director.decide(dctx, legal)
            except Exception:
                want = None
            if want in legal:
                self.hook_overrides += 1
                return want
            if want is not None:
                self.hook_fallbacks += 1
        return self.default_policy(strength, to_call, pot, hrng, street)

    def play_hand(self, hand_idx):
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

        actions = []          # row dicts (without hand_id/action_no yet)
        history = []          # lightweight tuples for director context
        folded = set()
        street_reached = "preflop"

        def active():
            return [s for s in range(N_SEATS) if s not in folded]

        def decide_seat(seat, street, level, committed_st, aggressions, started_us=None):
            nonlocal pot
            to_call = level - committed_st[seat]
            stack = stacks[seat]
            if stack <= 0:
                return None  # all-in, skipped
            if to_call == 0:
                legal = frozenset({"check", "bet", "all_in"})
            else:
                legal = frozenset({"fold", "call", "raise", "all_in"})
            if street == "preflop":
                strength = chen_strength(hole[seat])
            else:
                board = community[:BOARD_LEN[street]]
                strength = postflop_strength(hole[seat], board)
            if aggressions >= MAX_AGGRESSIONS_PER_STREET and to_call > 0:
                legal = frozenset({"fold", "call", "all_in"})
            elif aggressions >= MAX_AGGRESSIONS_PER_STREET:
                legal = frozenset({"check", "all_in"})
            board_now = community[:BOARD_LEN[street]] if street != "preflop" else []
            dctx = DecisionContext(
                hand_id=hand_id, hand_idx=hand_idx, street=street,
                seat_no=seat, player_id=self.roster[seat],
                partner_ids=[self.roster[s] for s in active() if s != seat],
                hole=tuple(hole[seat]), board=tuple(board_now),
                pot=pot, level=level, to_call=to_call, stack_remaining=stack,
                street_committed=committed_st[seat],
                players_active=len(active()), history=tuple(history),
                rng=stream("dec", self.seed, self.table_idx, hand_idx,
                           len(actions)))
            choice = self._decide(seat, dctx, legal, strength, to_call, pot, hrng, street)
            if choice not in legal:
                choice = self.default_policy(strength, to_call, pot, hrng)
                if choice not in legal:  # absolute safety net
                    choice = "check" if to_call == 0 else ("call" if stack > to_call else "all_in")
            # --- sizing / labeling (engine-owned, directors never size) ---
            committed_before = committed_st[seat]
            if choice == "all_in":
                amount = stack
            elif choice in ("fold", "check", "call"):
                amount = min(to_call, stack) if choice == "call" else 0
            elif choice == "bet":
                amount = bet_size(pot, stack)
            else:  # raise
                amount = raise_to_level(level, pot, committed_before, stack) - committed_before
            label = choice
            if amount == stack and amount > 0 and choice in ("call", "bet", "raise", "all_in"):
                label = "all_in"
            committed_st[seat] = committed_before + amount
            contributed[seat] += amount
            stacks[seat] -= amount
            return {"street": street, "player_id": self.roster[seat], "action": label,
                    "amount": amount, "amount_to": committed_st[seat],
                    "pot_before": pot, "stack_before": stack, "to_call": to_call,
                    "players_active": len(active())}

        # ---- streets ----
        order0 = [(button + 3 + i) % N_SEATS for i in range(N_SEATS)]  # UTG first
        first = {"preflop": order0,
                 "flop": [(button + 1 + i) % N_SEATS for i in range(N_SEATS)],
                 "turn": [(button + 1 + i) % N_SEATS for i in range(N_SEATS)],
                 "river": [(button + 1 + i) % N_SEATS for i in range(N_SEATS)]}
        level = BB
        showdown = False
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
            # rotate pointer to street start
            while len(actions) > MAX_ACTIONS_PER_HAND:
                break
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
                row = decide_seat(seat, street, level, committed_st, aggressions)
                if row is None:
                    acted.add(seat)
                    continue
                # aggression tracking BEFORE pot update (level_before = level)
                level_before = level
                if row["action"] in ("bet", "raise", "all_in") and row["amount_to"] > level_before:
                    level = row["amount_to"]
                    aggressions += 1
                    acted = {seat}
                else:
                    if row["action"] == "fold":
                        folded.add(seat)
                    acted.add(seat)
                pot += row["amount"]
                actions.append(row)
                history.append((street, self.roster[seat], row["action"], row["amount"]))
                if len(active()) <= 1:
                    break
            else:
                break
            if len(active()) <= 1:
                break
        # ---- settlement ----
        board_out = community[:BOARD_LEN[street_reached]]
        if len(active()) == 1:
            winners = list(active())
        else:
            # all remaining streets dealt out if everyone is all-in: board = full
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
        return hand_row, seat_rows, actions

    def run_table(self):
        hands, seats, actions = [], [], []
        for h in range(self.n_hands):
            hr, sr, ar = self.play_hand(h)
            hands.append(hr)
            seats.extend(sr)
            actions.extend(ar)
        return hands, seats, actions

    # -- players table (static per-table roster, real-marginal attributes) --
    def players_table(self):
        buckets_exp = ["new", "developing", "experienced", "veteran"]
        w_exp = [0.12, 0.27, 0.38, 0.23]
        stakes = ["micro", "low", "mid"]
        w_stake = [0.55, 0.35, 0.10]
        regions = ["americas", "europe", "apac", "other"]
        w_reg = [0.38, 0.34, 0.22, 0.06]
        clients = ["desktop", "mobile", "web"]
        w_cli = [0.49, 0.36, 0.15]
        rows = []
        for pid in self.roster:
            rng = stream("player-attr", self.seed, self.table_idx, pid)
            rows.append({"player_id": pid,
                         "account_age_days": rng.randint(30, 2500),
                         "experience_hands_bucket": rng.choices(buckets_exp, w_exp)[0],
                         "preferred_stake": rng.choices(stakes, w_stake)[0],
                         "region_bucket": rng.choices(regions, w_reg)[0],
                         "client_family": rng.choices(clients, w_cli)[0]})
        return rows


def to_parquet(hands, seats, actions, players, outdir):
    """Write schema-exact parquets (pyarrow). Returns {name: path}."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import os
    schemas = {
        "hands": pa.schema([("hand_id", pa.string()), ("table_id", pa.string()),
                            ("started_at", pa.timestamp("us", tz="UTC")),
                            ("phase", pa.string()), ("button_seat", pa.int64()),
                            ("small_blind", pa.int64()), ("big_blind", pa.int64()),
                            ("board_cards", pa.string()), ("final_pot", pa.int64()),
                            ("players_dealt", pa.int64()),
                            ("players_at_showdown", pa.int64())]),
        "seats": pa.schema([("hand_id", pa.string()), ("player_id", pa.string()),
                            ("seat_no", pa.int64()), ("starting_stack", pa.int64()),
                            ("hole_card_1", pa.string()), ("hole_card_2", pa.string()),
                            ("total_contribution", pa.int64()), ("net_chips", pa.int64()),
                            ("folded", pa.bool_()), ("went_to_showdown", pa.bool_()),
                            ("won_share", pa.float64())]),
        "actions": pa.schema([("hand_id", pa.string()), ("action_no", pa.int64()),
                              ("street", pa.string()), ("player_id", pa.string()),
                              ("action", pa.string()), ("amount", pa.int64()),
                              ("amount_to", pa.int64()), ("pot_before", pa.int64()),
                              ("stack_before", pa.int64()), ("to_call", pa.int64()),
                              ("players_active", pa.int64())]),
        "players": pa.schema([("player_id", pa.string()),
                              ("account_age_days", pa.int64()),
                              ("experience_hands_bucket", pa.string()),
                              ("preferred_stake", pa.string()),
                              ("region_bucket", pa.string()),
                              ("client_family", pa.string())]),
    }
    tables = {"hands": (hands, HAND_COLS), "seats": (seats, SEAT_COLS),
              "actions": (actions, ACTION_COLS), "players": (players, PLAYER_COLS)}
    paths = {}
    for name, (rows, cols) in tables.items():
        coldata = {c: [r[c] for r in rows] for c in cols}
        t = pa.Table.from_pydict(coldata, schema=schemas[name])
        p = os.path.join(outdir, "demo_%s.parquet" % name)
        pq.write_table(t, p)
        paths[name] = p
    return paths
