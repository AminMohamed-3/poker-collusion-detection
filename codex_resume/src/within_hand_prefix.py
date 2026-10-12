"""Pure causal public-action prefix extraction for a focal pair decision.

Tokens describe only actions with ``action_no < target_action_no``.  They mask
all private cards and terminal outcomes; they are not poker-data I/O or a model.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


MAX_PREFIX_EVENTS = 24
TOKEN_DIM = 18
_STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_RANKS = {rank: value for value, rank in enumerate("23456789TJQKA", start=2)}
_SUITS = {suit: index for index, suit in enumerate("cdhs")}


def _finite_nonnegative(event: Mapping[str, Any], name: str) -> float:
    try:
        value = float(event[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid {name}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _integer(value: Any, name: str, low: int | None = None, high: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(number) or number != round(number):
        raise ValueError(f"{name} must be an integer")
    result = int(number)
    if (low is not None and result < low) or (high is not None and result > high):
        raise ValueError(f"{name} is out of range")
    return result


def _street(value: Any) -> int:
    if isinstance(value, str):
        if value not in _STREETS:
            raise ValueError("unknown street")
        return _STREETS[value]
    return _integer(value, "street", 0, 3)


def _seat_map(seats: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> dict[Any, int]:
    if isinstance(seats, Mapping):
        rows = [{"player_id": player, **dict(row)} for player, row in seats.items()]
    else:
        rows = list(seats)
    result: dict[Any, int] = {}
    for row in rows:
        if not isinstance(row, Mapping) or "player_id" not in row:
            raise ValueError("each seat requires player_id")
        player = row["player_id"]
        if player in result:
            raise ValueError("duplicate player seat")
        result[player] = _integer(row.get("seat_no"), "seat_no", 0, 5)
    if not result:
        raise ValueError("at least one seat is required")
    return result


def _board_summary(board_cards: Any, street: int) -> tuple[float, float, float]:
    required = (0, 3, 4, 5)[street]
    if required == 0:
        return 0.0, 0.0, 0.0
    if board_cards is None:
        pieces: list[str] = []
    elif isinstance(board_cards, str):
        pieces = board_cards.split()
    else:
        raise ValueError("board_cards must be a whitespace string or None")
    if len(pieces) < required:
        raise ValueError("board is shorter than the event street requires")
    # Read exactly the legal board prefix.  Later suffix cards are deliberately
    # neither parsed nor validated, so they cannot affect an earlier prefix.
    prefix = pieces[:required]
    ranks: list[int] = []
    suits: list[int] = []
    for card in prefix:
        if len(card) != 2 or card[0] not in _RANKS or card[1] not in _SUITS:
            raise ValueError("unknown board card in required prefix")
        ranks.append(_RANKS[card[0]])
        suits.append(_SUITS[card[1]])
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    return max(ranks) / 14.0, sum(count - 1 for count in rank_counts.values()) / 2.0, max(suit_counts.values()) / 5.0


def _action_class(event: Mapping[str, Any], amount: float, to_call: float) -> tuple[int, float]:
    action = event.get("action")
    if action == "fold":
        return 0, 0.0
    if action == "check":
        return 1, 0.0
    if action == "call":
        return 2, 0.0
    if action == "all_in":
        return (2 if amount <= to_call else 3), 1.0
    if action in ("bet", "raise"):
        return 3, 0.0
    raise ValueError("unknown action")


def extract_prefix(
    hand: Mapping[str, Any],
    seats: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    focal_player: Any,
    partner_player: Any,
    target_action_no: int,
) -> tuple[np.ndarray, int]:
    """Return latest-24 causal public action tokens before a target action number.

    Player/table/hand IDs are grouping inputs only.  The function does not read
    hole cards, terminal outcomes, target-action contents, or later-action
    contents.  It intentionally performs no contribution/to-call replay.
    """
    if not isinstance(hand, Mapping):
        raise TypeError("hand must be a mapping")
    if focal_player == partner_player:
        raise ValueError("focal and partner players must differ")
    target = _integer(target_action_no, "target_action_no")
    try:
        blind = float(hand["big_blind"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("missing or invalid big_blind") from exc
    if not math.isfinite(blind) or blind <= 0:
        raise ValueError("big_blind must be finite and positive")
    seat_numbers = _seat_map(seats)
    if focal_player not in seat_numbers or partner_player not in seat_numbers:
        raise ValueError("focal and partner must have seats")

    # Deliberately filter using only action numbers before touching any event
    # payload.  This isolates a focal prefix from target/future mutations.
    prior: list[tuple[int, Mapping[str, Any]]] = []
    for event in actions:
        if not isinstance(event, Mapping):
            raise ValueError("actions must be mappings")
        number = event.get("action_no")
        try:
            is_prior = number < target
        except TypeError as exc:
            raise ValueError("action_no must be comparable to target_action_no") from exc
        if is_prior:
            prior.append((_integer(number, "action_no", 0), event))
    prior.sort(key=lambda item: item[0])
    numbers = [number for number, _ in prior]
    if len(numbers) != len(set(numbers)):
        raise ValueError("duplicate action_no in prefix")

    tokens: list[np.ndarray] = []
    last_street = -1
    for _number, event in prior:
        street = _street(event.get("street"))
        if street < last_street:
            raise ValueError("prefix returns to an earlier street")
        last_street = street
        actor = event.get("player_id")
        if actor not in seat_numbers:
            raise ValueError("action actor has no seat")
        amount = _finite_nonnegative(event, "amount")
        to_call = _finite_nonnegative(event, "to_call")
        pot_before = _finite_nonnegative(event, "pot_before")
        stack_before = _finite_nonnegative(event, "stack_before")
        active = _integer(event.get("players_active"), "players_active", 1, 6)
        action_class, is_all_in = _action_class(event, amount, to_call)
        board_high, board_pairs, board_flush = _board_summary(hand.get("board_cards"), street)
        role = 0 if actor == focal_player else 1 if actor == partner_player else 2
        token = np.zeros(TOKEN_DIM, dtype=np.float32)
        token[0] = street / 3.0
        token[1 + role] = 1.0
        token[4 + action_class] = 1.0
        token[8] = is_all_in
        token[9] = math.log1p(amount / blind) / 5.0
        token[10] = to_call / max(pot_before + to_call, 1.0)
        token[11] = math.log1p(pot_before / blind) / 5.0
        token[12] = math.log1p(stack_before / blind) / 5.0
        token[13] = active / 6.0
        token[14] = ((seat_numbers[actor] - seat_numbers[focal_player]) % 6) / 5.0
        token[15:18] = (board_high, board_pairs, board_flush)
        if not np.isfinite(token).all():
            raise ValueError("non-finite prefix token")
        tokens.append(token)

    selected = tokens[-MAX_PREFIX_EVENTS:]
    result = np.zeros((MAX_PREFIX_EVENTS, TOKEN_DIM), dtype=np.float32)
    if selected:
        result[: len(selected)] = np.stack(selected)
    return result, len(selected)
