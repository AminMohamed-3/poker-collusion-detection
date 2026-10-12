"""Deterministic synthetic poker teacher-mechanics fixtures and factuality gate.

This module contains no real poker data, labels, predictions, model code, or model
weights.  It creates twelve original one-hand mini-histories in six matched pairs.
Amounts are always additional chips committed by the named action.  A raise also
records ``to_amount``, the resulting total street commitment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "fixtures"
ALLOWED_TAGS = {"none", "chip_transfer_pattern", "soft_play_pattern", "isolation_pattern"}
UNCERTAINTY_ORDER = {"low": 0, "medium": 1, "high": 2}
STREETS = ("preflop", "flop", "turn", "river")
CARD_RE = re.compile(r"^(?:[2-9TJQKA])[cdhs]$")


RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Synthetic teacher mechanics response",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version", "case_id", "pair_id", "hand_ids", "cited_line_ids",
        "observed_facts", "unavailable_facts", "mechanism_assessment",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "case_id": {"type": "string"},
        "pair_id": {"type": "string"},
        "hand_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "cited_line_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "observed_facts": {
            "type": "object", "additionalProperties": False,
            "required": ["actions", "net_chips", "outcomes"],
            "properties": {
                "actions": {
                    "type": "array", "minItems": 4, "maxItems": 4,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["line_id", "hand_id", "street", "actor_alias", "action", "amount", "to_amount"],
                        "properties": {
                            "line_id": {"type": "string"}, "hand_id": {"type": "string"},
                            "street": {"enum": list(STREETS)}, "actor_alias": {"type": "string"},
                            "action": {"enum": ["fold", "check", "call", "bet", "raise"]},
                            "amount": {"type": "integer", "minimum": 0},
                            "to_amount": {"type": ["integer", "null"], "minimum": 0},
                        },
                    },
                },
                "net_chips": {"type": "object", "additionalProperties": {"type": "integer"}},
                "outcomes": {
                    "type": "array", "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["hand_id", "winner_aliases", "pot", "showdown", "returned_uncalled"],
                        "properties": {
                            "hand_id": {"type": "string"},
                            "winner_aliases": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "pot": {"type": "integer", "minimum": 0}, "showdown": {"type": "boolean"},
                            "returned_uncalled": {
                                "type": "array", "items": {
                                    "type": "object", "additionalProperties": False,
                                    "required": ["actor_alias", "amount"],
                                    "properties": {"actor_alias": {"type": "string"}, "amount": {"type": "integer", "minimum": 1}},
                                },
                            },
                        },
                    },
                },
            },
        },
        "unavailable_facts": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "mechanism_assessment": {
            "type": "object", "additionalProperties": False,
            "required": ["observable_tags", "evidence_for", "evidence_against", "uncertainty", "abstain", "abstention_reason"],
            "properties": {
                "observable_tags": {"type": "array", "items": {"enum": sorted(ALLOWED_TAGS)}, "uniqueItems": True},
                "evidence_for": {"$ref": "#/$defs/evidence"},
                "evidence_against": {"$ref": "#/$defs/evidence"},
                "uncertainty": {"enum": ["low", "medium", "high"]},
                "abstain": {"type": "boolean"},
                "abstention_reason": {"type": ["string", "null"]},
            },
        },
    },
    "$defs": {
        "evidence": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["claim", "cited_line_ids"],
                "properties": {
                    "claim": {"type": "string", "minLength": 1},
                    "cited_line_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True},
                },
            },
        },
    },
}


PROMPT_TEMPLATE = """You are checking observable mechanics in an original synthetic no-limit hold'em mini-history. Do not infer private intent or guilt. Return one JSON object only, matching schema_version 1.0.

Amount convention: `amount` is the additional chips committed by that action, never a street-total. Only raises have a non-null `to_amount`, which is the actor's resulting total commitment on that street. Net chips cover the complete hand from listed starting stacks through any uncalled-chip return and the terminal called-pot award.

Extract exactly the four designated decision lines, in listed order. Copy only visible facts. Cite only line IDs present in the source. `observable_tags` describe betting mechanics, not intent. Evidence prose is explanatory and is not a guilt label. Since intent is omitted, abstain from an intent conclusion and give a reason. Use high uncertainty when the source is observationally indistinguishable from its matched case. Source lines are chronological: never treat a later reveal or event as information available for an earlier decision. Assess only the two aliases on the review_pair metadata line.

Required top-level keys: schema_version, case_id, pair_id, hand_ids, cited_line_ids, observed_facts, unavailable_facts, mechanism_assessment.
`observed_facts` keys: actions, net_chips, outcomes.
Each action: line_id, hand_id, street, actor_alias, action, amount, to_amount.
Each outcome: hand_id, winner_aliases (array), pot, showdown, returned_uncalled (array of actor_alias and amount).
`mechanism_assessment`: observable_tags (none/chip_transfer_pattern/soft_play_pattern/isolation_pattern), evidence_for, evidence_against, uncertainty (low/medium/high), abstain, abstention_reason.
Each evidence item: claim and cited_line_ids. Evidence claims must remain observational.

CASE_ID: <<CASE_ID>>
PAIR_ID: <<PAIR_ID>>
HAND_IDS: <<HAND_IDS>>
DESIGNATED_DECISION_LINE_IDS: <<EXTRACTION_LINE_IDS>>

SOURCE:
<<SOURCE_TEXT>>
"""


ALIASES = [
    ({"A": "Aspen", "B": "Briar", "C": "Cedar"}, {"A": "Nile", "B": "Orchid", "C": "Pine"}),
    ({"A": "Dune", "B": "Ember", "C": "Flint"}, {"A": "Quartz", "B": "Reed", "C": "Sable"}),
    ({"A": "Grove", "B": "Harbor", "C": "Indigo"}, {"A": "Tamar", "B": "Umber", "C": "Vale"}),
    ({"A": "Juniper", "B": "Kestrel", "C": "Linden"}, {"A": "Willow", "B": "Xenia", "C": "Yarrow"}),
    ({"A": "Mesa", "B": "North", "C": "Osprey"}, {"A": "Aster", "B": "Brook", "C": "Cairn"}),
    ({"A": "Prairie", "B": "Raven", "C": "Spruce"}, {"A": "Delta", "B": "Elm", "C": "Fjord"}),
    ({"A": "Topaz", "B": "Alder", "C": "Birch"}, {"A": "Gale", "B": "Heath", "C": "Iris"}),
    ({"A": "Cobalt", "B": "Drift", "C": "Estuary"}, {"A": "Jade", "B": "Knoll", "C": "Lagoon"}),
    ({"A": "Fern", "B": "Granite", "C": "Hollow"}, {"A": "Marsh", "B": "Nova", "C": "Opal"}),
    ({"A": "Islet", "B": "Jetty", "C": "Karst"}, {"A": "Pecan", "B": "Ridge", "C": "Shoal"}),
    ({"A": "Larch", "B": "Moor", "C": "Nectar"}, {"A": "Thorn", "B": "Upland", "C": "Violet"}),
    ({"A": "Onyx", "B": "Pebble", "C": "Quill"}, {"A": "Wren", "B": "Xylem", "C": "Yucca"}),
]


def _post(actor: str, amount: int, blind: str) -> dict[str, Any]:
    return {"type": "action", "street": "preflop", "actor": actor, "action": f"post_{blind}_blind", "amount": amount, "to_amount": None}


def _act(street: str, actor: str, action: str, amount: int = 0, to_amount: int | None = None) -> dict[str, Any]:
    return {"type": "action", "street": street, "actor": actor, "action": action, "amount": amount, "to_amount": to_amount}


def _event(street: str, actor: str | None, code: str) -> dict[str, Any]:
    return {"type": "event", "street": street, "actor": actor, "event_code": code}


BASE_OMITTED = ["private_intent", "private_communications", "off_excerpt_history", "complete_hole_card_identities"]


RETURN_UNCALLED = [10, 0, 70, 100, 0, 100]
REVIEW_ROLES = [("B", "C"), ("B", "C"), ("A", "C"), ("B", "C"), ("A", "C"), ("A", "C")]

PAIR_SPECS: list[dict[str, Any]] = [
    {
        "actions": [
            _post("B", 5, "small"), _post("C", 10, "big"),
            _act("preflop", "A", "fold"), _act("preflop", "B", "call", 5), _act("preflop", "C", "check"),
            _act("flop", "B", "bet", 10), _act("flop", "C", "call", 10),
            _act("turn", "B", "bet", 20), _act("turn", "C", "call", 20),
            _act("river", "B", "bet", 10), "EVENT", _act("river", "C", "fold"),
        ],
        "event_explanation": _event("river", "C", "client_disconnect_auto_fold_announced"),
        "event_probe": None,
        "extract_positions": [7, 8, 9, 10],
        "winner": "B", "showdown": False,
        "tags": ("none", "chip_transfer_pattern"),
        "evidence": ("The visible automatic-fold notice supplies a direct non-intent explanation.", "A small river bet follows two called streets and wins by a voluntary fold."),
    },
    {
        "actions": [
            _post("B", 5, "small"), _post("C", 10, "big"),
            _act("preflop", "A", "fold"), _act("preflop", "B", "call", 5), _act("preflop", "C", "check"),
            _act("flop", "B", "bet", 10), _act("flop", "C", "call", 10),
            _act("turn", "B", "bet", 20), _act("turn", "C", "call", 20),
            _act("river", "B", "check"), "EVENT", _act("river", "C", "check"),
        ],
        "event_explanation": _event("river", "B", "client_disconnect_auto_check_announced"),
        "event_probe": None,
        "extract_positions": [7, 8, 9, 10],
        "winner": "C", "showdown": True,
        "tags": ("none", "soft_play_pattern"),
        "evidence": ("The visible automatic-check notice explains the river check.", "The prior aggressor voluntarily checks and the hand checks down to the other player."),
    },
    {
        "actions": [
            _post("B", 5, "small"), _post("C", 10, "big"),
            _act("preflop", "A", "raise", 30, 30), _act("preflop", "B", "call", 25),
            _act("preflop", "C", "raise", 90, 100), _act("preflop", "A", "fold"), _act("preflop", "B", "fold"), "EVENT",
        ],
        "event_explanation": _event("preflop", "C", "post_hand_two_aces_revealed"),
        "event_probe": None,
        "extract_positions": [2, 3, 4, 5],
        "winner": "C", "showdown": False,
        "tags": ("none", "isolation_pattern"),
        "evidence": ("The post-hand reveal supplies visible support for an ordinary value raise.", "A large re-raise removes both prior participants without showdown."),
    },
    {
        "actions": [
            _post("B", 5, "small"), _post("C", 10, "big"),
            _act("preflop", "A", "fold"), _act("preflop", "B", "call", 5), _act("preflop", "C", "check"),
            _act("flop", "B", "bet", 20), _act("flop", "C", "call", 20),
            _act("turn", "B", "bet", 40), _act("turn", "C", "call", 40),
            _act("river", "B", "check"), _act("river", "C", "bet", 100), "EVENT", _act("river", "B", "fold"),
        ],
        "event_explanation": _event("river", "B", "time_bank_expired_auto_fold_announced"),
        "event_probe": None,
        "extract_positions": [7, 8, 9, 11],
        "winner": "C", "showdown": False,
        "tags": ("none", "chip_transfer_pattern"),
        "evidence": ("The time-bank notice directly precedes an automatic fold.", "One player builds the pot, then voluntarily folds to a large river bet."),
    },
    {
        "actions": [
            _post("B", 5, "small"), _post("C", 10, "big"),
            _act("preflop", "A", "raise", 30, 30), _act("preflop", "B", "fold"), _act("preflop", "C", "call", 20),
            _act("flop", "C", "check"), _act("flop", "A", "bet", 40), _act("flop", "C", "call", 40),
            _act("turn", "C", "check"), _act("turn", "A", "check"),
            _act("river", "C", "check"), "EVENT", _act("river", "A", "check"),
        ],
        "event_explanation": _event("river", "A", "client_disconnect_auto_check_announced"),
        "event_probe": None,
        "extract_positions": [8, 9, 10, 11],
        "winner": "A", "showdown": True,
        "tags": ("none", "soft_play_pattern"),
        "evidence": ("The visible disconnect notice explains the terminal automatic check.", "The flop aggressor voluntarily checks back twice and still wins at showdown."),
    },
    {
        "actions": [
            _post("B", 5, "small"), _post("C", 10, "big"),
            _act("preflop", "A", "call", 10), _act("preflop", "B", "call", 5), _act("preflop", "C", "check"),
            _act("flop", "B", "check"), _act("flop", "C", "bet", 20), _act("flop", "A", "call", 20), _act("flop", "B", "fold"),
            _act("turn", "C", "check"), _act("turn", "A", "bet", 40), _act("turn", "C", "call", 40),
            _act("river", "C", "bet", 100), _act("river", "A", "fold"),
        ],
        "event_explanation": None,
        "event_probe": None,
        "extract_positions": [9, 10, 12, 13],
        "winner": "C", "showdown": False,
        "tags": ("chip_transfer_pattern", "chip_transfer_pattern"),
        "evidence": ("The sequence moves a large pot to the river bettor, but supplies no cause.", "The sequence moves a large pot to the river bettor, but supplies no cause."),
        "unidentifiable": True,
    },
]


def _materialize_actions(spec: dict[str, Any], role_index: int) -> list[dict[str, Any]]:
    event = spec["event_explanation"] if role_index == 0 else spec["event_probe"]
    out: list[dict[str, Any]] = []
    for item in spec["actions"]:
        if item == "EVENT":
            if event is not None:
                out.append(copy.deepcopy(event))
        else:
            out.append(copy.deepcopy(item))
    return out


def _assign_line_ids(case_num: int, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, item in enumerate(items, 1):
        item["line_id"] = f"tm_l{case_num:03d}_{index:02d}"
    return items


def _ledger_result(hand: dict[str, Any]) -> tuple[dict[str, int], int]:
    contributions = {p: 0 for p in hand["players"]}
    for item in hand["chronology"]:
        if item["type"] == "action":
            contributions[item["actor"]] += item["amount"]
        elif item["type"] == "return":
            contributions[item["actor"]] -= item["amount"]
    pot = sum(contributions.values())
    winner = hand["outcome"]["winner"]
    net = {p: -contributions[p] for p in hand["players"]}
    net[winner] += pot
    return net, pot


def _metadata_lines(case: dict[str, Any], variant: bool) -> list[str]:
    prefix = case["case_id"].replace("tm_case_", "tm_m")
    aliases = case["variant_aliases"] if variant else case["aliases"]
    hand = case["mechanics_ledger"]["hands"][0]
    values = {
        "case": f"{prefix}_01 | case_id={case['case_id']}",
        "pair": f"{prefix}_02 | pair_id={case['pair_id']}",
        "hands": f"{prefix}_03 | hand_ids={','.join(case['hand_ids'])}",
        "amount": f"{prefix}_04 | amount=additional_chips; raise_to=resulting_street_commitment",
        "omitted": f"{prefix}_05 | omitted={','.join(case['expected_facts']['unavailable_facts'])}",
        "table": f"{prefix}_06 | button={aliases[hand['button']]}; blinds={hand['small_blind']}/{hand['big_blind']}; starting_stacks=" + ",".join(f"{aliases[p]}:{hand['starting_stacks'][p]}" for p in hand["players"]),
        "review": f"{prefix}_07 | review_pair=" + ",".join(aliases[p] for p in case["review_roles"]),
    }
    order = ["case", "pair", "hands", "table", "review", "amount", "omitted"] if not variant else ["amount", "omitted", "review", "table", "hands", "pair", "case"]
    return [values[key] for key in order]


def render_case(case: dict[str, Any], variant: bool = False) -> str:
    aliases = case["variant_aliases"] if variant else case["aliases"]
    hand = case["mechanics_ledger"]["hands"][0]
    lines = _metadata_lines(case, variant)
    lines.append("chronology:")
    for item in hand["chronology"]:
        actor = aliases[item["actor"]] if item.get("actor") else "-"
        if item["type"] == "event":
            lines.append(f"{item['line_id']} | {hand['hand_id']} | {item['street']} | event | actor={actor} | code={item['event_code']}")
        elif item["type"] == "return":
            lines.append(f"{item['line_id']} | {hand['hand_id']} | terminal | return | actor={actor} | amount={item['amount']}")
        else:
            to_text = str(item["to_amount"]) if item["to_amount"] is not None else "-"
            lines.append(
                f"{item['line_id']} | {hand['hand_id']} | {item['street']} | actor={actor} | "
                f"action={item['action']} | add={item['amount']} | to={to_text}"
            )
    outcome = hand["outcome"]
    lines.append(
        f"{outcome['line_id']} | {hand['hand_id']} | terminal | winners={aliases[outcome['winner']]} | "
        f"pot={outcome['pot']} | showdown={str(outcome['showdown']).lower()}"
    )
    return "\n".join(lines)


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    case_num = 0
    for pair_index, spec in enumerate(PAIR_SPECS, 1):
        for role_index, role in enumerate(("matched_explanation", "mechanics_probe")):
            case_num += 1
            aliases, variant_aliases = ALIASES[case_num - 1]
            raw_items = _materialize_actions(spec, role_index)
            returned_amount = RETURN_UNCALLED[pair_index - 1]
            if returned_amount:
                last_street = next(x["street"] for x in reversed(raw_items) if isinstance(x, dict) and x["type"] == "action")
                raw_items.append({"type": "return", "street": last_street, "actor": spec["winner"], "amount": returned_amount})
            items = _assign_line_ids(case_num, raw_items)
            hand_id = f"tm_hand_{case_num:03d}"
            outcome_line = f"tm_l{case_num:03d}_{len(items) + 1:02d}"
            hand = {
                "hand_id": hand_id,
                "players": ["A", "B", "C"],
                "starting_stacks": {"A": 200, "B": 200, "C": 200},
                "button": "A", "small_blind": 5, "big_blind": 10,
                "chronology": items,
                "known_cards": [],
                "outcome": {"line_id": outcome_line, "winner": spec["winner"], "pot": 0, "showdown": spec["showdown"], "returned_uncalled": []},
            }
            net, pot = _ledger_result(hand)
            hand["outcome"]["pot"] = pot
            hand["outcome"]["returned_uncalled"] = [
                {"line_id": x["line_id"], "actor": x["actor"], "amount": x["amount"]}
                for x in items if x["type"] == "return"
            ]
            action_items = [x for x in items if x["type"] == "action"]
            designated = [action_items[i] for i in spec["extract_positions"]]
            expected_actions = [
                {
                    "line_id": x["line_id"], "hand_id": hand_id, "street": x["street"],
                    "actor_alias": aliases[x["actor"]], "action": x["action"], "amount": x["amount"],
                    "to_amount": x["to_amount"],
                }
                for x in designated
            ]
            tag = spec["tags"][role_index]
            unavailable = list(BASE_OMITTED)
            if spec.get("unidentifiable"):
                unavailable.append("cause_of_observed_sequence")
            elif role_index == 1:
                unavailable.append("cause_of_selected_action")
            required_uncertainty = "high" if spec.get("unidentifiable") else "medium"
            case = {
                "case_id": f"tm_case_{case_num:03d}",
                "pair_id": f"tm_pair_{pair_index:02d}",
                "hand_ids": [hand_id],
                "design_role": role,
                "unidentifiable_pair": bool(spec.get("unidentifiable", False)),
                "review_roles": list(REVIEW_ROLES[pair_index - 1]),
                "author_mechanism_note": {
                    "non_gold": True, "suggested_tags": [tag], "evidence_seed": spec["evidence"][role_index],
                    "note": "Optional fixture-author description; excluded from prompts and factual pass/fail.",
                },
                "aliases": aliases,
                "variant_aliases": variant_aliases,
                "extraction_line_ids": [x["line_id"] for x in designated],
                "source_text": "",
                "variant_text": "",
                "expected_facts": {
                    "actions": expected_actions,
                    "net_chips": {aliases[p]: net[p] for p in hand["players"]},
                    "outcomes": [{
                        "hand_id": hand_id, "winner_aliases": [aliases[spec["winner"]]],
                        "pot": pot, "showdown": spec["showdown"],
                        "returned_uncalled": [
                            {"actor_alias": aliases[x["actor"]], "amount": x["amount"]}
                            for x in items if x["type"] == "return"
                        ],
                    }],
                    "unavailable_facts": unavailable,
                    "required_assessment": {
                        "abstain": True, "minimum_uncertainty": required_uncertainty,
                    },
                },
                "mechanics_ledger": {"hands": [hand]},
            }
            case["source_text"] = render_case(case, variant=False)
            case["variant_text"] = render_case(case, variant=True)
            cases.append(case)
    return cases


def build_prompt(case: dict[str, Any], variant: bool = False) -> str:
    text = case["variant_text"] if variant else case["source_text"]
    return (
        PROMPT_TEMPLATE.replace("<<CASE_ID>>", case["case_id"])
        .replace("<<PAIR_ID>>", case["pair_id"])
        .replace("<<HAND_IDS>>", json.dumps(case["hand_ids"], separators=(",", ":")))
        .replace("<<EXTRACTION_LINE_IDS>>", json.dumps(case["extraction_line_ids"], separators=(",", ":")))
        .replace("<<SOURCE_TEXT>>", text)
    )


def _all_line_ids(case: dict[str, Any]) -> set[str]:
    ids = {line.split(" | ", 1)[0] for line in case["source_text"].splitlines() if " | " in line}
    return ids


def _validate_unique_cards(hand: dict[str, Any], errors: list[str]) -> None:
    cards = hand.get("known_cards", [])
    if any(not isinstance(card, str) or not CARD_RE.match(card) for card in cards):
        errors.append(f"{hand['hand_id']}: invalid card code")
    if len(cards) != len(set(cards)):
        errors.append(f"{hand['hand_id']}: duplicate known card")


def validate_hand(hand: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    players = hand["players"]
    if len(players) != 3 or len(set(players)) != 3:
        errors.append(f"{hand['hand_id']}: expected three unique players")
        return errors
    stacks = dict(hand["starting_stacks"])
    if set(stacks) != set(players) or any(not isinstance(v, int) or v <= 0 for v in stacks.values()):
        errors.append(f"{hand['hand_id']}: invalid starting stacks")
        return errors
    _validate_unique_cards(hand, errors)
    committed_total = {p: 0 for p in players}
    committed_street = {p: 0 for p in players}
    active = set(players)
    current_bet = 0
    last_full_raise = hand["big_blind"]
    current_street = "preflop"
    positions = list(players)
    button_index = positions.index(hand["button"])
    expected_blinds = [
        ("post_small_blind", positions[(button_index + 1) % len(positions)]),
        ("post_big_blind", positions[(button_index + 2) % len(positions)]),
    ]
    blind_posts: list[tuple[str, str]] = []
    return_records: list[dict[str, Any]] = []
    expected_actor = positions[(button_index + 3) % len(positions)]
    seen_lines: set[str] = set()

    def next_active(after: str) -> str:
        index = positions.index(after)
        for offset in range(1, len(positions) + 1):
            candidate = positions[(index + offset) % len(positions)]
            if candidate in active:
                return candidate
        raise AssertionError("no active player")
    for item in hand["chronology"]:
        if return_records:
            errors.append(f"{hand['hand_id']}: chronology continues after terminal return")
        line_id = item.get("line_id")
        if not line_id or line_id in seen_lines:
            errors.append(f"{hand['hand_id']}: missing or duplicate line ID {line_id!r}")
        seen_lines.add(line_id)
        if item["street"] not in STREETS:
            errors.append(f"{hand['hand_id']}: invalid street {item['street']}")
            continue
        if STREETS.index(item["street"]) < STREETS.index(current_street):
            errors.append(f"{hand['hand_id']}: street order regressed")
        if item["street"] != current_street:
            unsettled = [p for p in active if committed_street[p] != current_bet and stacks[p] > 0]
            if unsettled:
                errors.append(f"{hand['hand_id']}: prior street unsettled for {unsettled}")
            current_street = item["street"]
            committed_street = {p: 0 for p in players}
            current_bet = 0
            last_full_raise = hand["big_blind"]
            expected_actor = next_active(hand["button"])
        if item["type"] == "event":
            if item.get("actor") not in players and item.get("actor") is not None:
                errors.append(f"{hand['hand_id']}: event has unknown actor")
            continue
        if item["type"] == "return":
            actor = item.get("actor")
            amount = item.get("amount")
            if actor not in active or len(active) != 1 or not isinstance(amount, int) or amount <= 0:
                errors.append(f"{hand['hand_id']}: invalid terminal uncalled-chip return")
                continue
            matched_level = max(committed_street[p] for p in players if p != actor)
            expected_return = committed_street[actor] - matched_level
            if amount != expected_return:
                errors.append(f"{hand['hand_id']}: returned {amount}, expected unmatched {expected_return}")
            if amount > committed_total[actor]:
                errors.append(f"{hand['hand_id']}: return exceeds contribution")
                continue
            stacks[actor] += amount
            committed_total[actor] -= amount
            committed_street[actor] -= amount
            current_bet = matched_level
            return_records.append({"line_id": item["line_id"], "actor": actor, "amount": amount})
            continue
        if item["type"] != "action":
            errors.append(f"{hand['hand_id']}: unknown chronology item type {item['type']}")
            continue
        actor = item["actor"]
        action = item["action"]
        amount = item["amount"]
        if actor not in active:
            errors.append(f"{hand['hand_id']}: inactive actor {actor}")
            continue
        if not isinstance(amount, int) or amount < 0 or amount > stacks[actor]:
            errors.append(f"{hand['hand_id']}: invalid amount for {actor}")
            continue
        before = committed_street[actor]
        to_call = current_bet - before
        if action.startswith("post_"):
            blind_posts.append((action, actor))
            expected = hand["small_blind"] if action == "post_small_blind" else hand["big_blind"]
            if current_street != "preflop" or amount != expected or before != 0:
                errors.append(f"{hand['hand_id']}: illegal blind post")
        else:
            if blind_posts != expected_blinds:
                errors.append(f"{hand['hand_id']}: blinds missing, repeated, or out of order")
            if actor != expected_actor:
                errors.append(f"{hand['hand_id']}: out-of-turn action by {actor}; expected {expected_actor}")
        if action.startswith("post_"):
            pass
        elif action == "fold":
            if amount != 0 or item["to_amount"] is not None:
                errors.append(f"{hand['hand_id']}: fold commits chips")
            active.remove(actor)
            if active:
                expected_actor = next_active(actor)
            continue
        elif action == "check":
            if amount != 0 or to_call != 0 or item["to_amount"] is not None:
                errors.append(f"{hand['hand_id']}: illegal check by {actor}")
        elif action == "call":
            expected = min(to_call, stacks[actor])
            if to_call <= 0 or amount != expected or item["to_amount"] is not None:
                errors.append(f"{hand['hand_id']}: illegal call by {actor}")
        elif action == "bet":
            if current_bet != 0 or amount <= 0 or item["to_amount"] is not None:
                errors.append(f"{hand['hand_id']}: illegal bet by {actor}")
            if amount < hand["big_blind"] and amount != stacks[actor]:
                errors.append(f"{hand['hand_id']}: bet below minimum")
            current_bet = before + amount
            last_full_raise = current_bet
        elif action == "raise":
            new_total = before + amount
            raise_size = new_total - current_bet
            if current_bet <= 0 or new_total <= current_bet or item["to_amount"] != new_total:
                errors.append(f"{hand['hand_id']}: illegal raise by {actor}")
            if raise_size < last_full_raise and amount != stacks[actor]:
                errors.append(f"{hand['hand_id']}: raise below minimum")
            if raise_size >= last_full_raise:
                last_full_raise = raise_size
            current_bet = new_total
        else:
            errors.append(f"{hand['hand_id']}: unknown action {action}")
        stacks[actor] -= amount
        committed_total[actor] += amount
        committed_street[actor] += amount
        if action.startswith("post_"):
            current_bet = max(current_bet, committed_street[actor])
        elif active:
            expected_actor = next_active(actor)
    if blind_posts != expected_blinds:
        errors.append(f"{hand['hand_id']}: expected exactly one small and big blind post")
    if len(active) > 1:
        unsettled = [p for p in active if committed_street[p] != current_bet and stacks[p] > 0]
        if unsettled:
            errors.append(f"{hand['hand_id']}: terminal street unsettled for {unsettled}")
        if not hand["outcome"]["showdown"]:
            errors.append(f"{hand['hand_id']}: multiple active players without showdown")
    elif hand["outcome"]["showdown"]:
        errors.append(f"{hand['hand_id']}: showdown declared after folds leave one player")
    else:
        sole = next(iter(active))
        residual_unmatched = committed_street[sole] - max(committed_street[p] for p in players if p != sole)
        if residual_unmatched != 0:
            errors.append(f"{hand['hand_id']}: missing terminal return of {residual_unmatched}")
    if hand["outcome"].get("returned_uncalled", []) != return_records:
        errors.append(f"{hand['hand_id']}: outcome return records differ from ledger")
    pot = sum(committed_total.values())
    if hand["outcome"]["pot"] != pot:
        errors.append(f"{hand['hand_id']}: outcome pot {hand['outcome']['pot']} != ledger pot {pot}")
    winner = hand["outcome"]["winner"]
    if winner not in active:
        errors.append(f"{hand['hand_id']}: winner is folded or unknown")
    final = {p: stacks[p] + (pot if p == winner else 0) for p in players}
    if sum(final.values()) != sum(hand["starting_stacks"].values()):
        errors.append(f"{hand['hand_id']}: chips are not conserved")
    if any(value < 0 for value in final.values()):
        errors.append(f"{hand['hand_id']}: negative final stack")
    return errors


def validate_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    neutral_id_text = " ".join([case["case_id"], case["pair_id"], *case["hand_ids"], *case["extraction_line_ids"]]).lower()
    if re.search(r"innocent|mechanic|transfer|soft|isolation|collu|guilt", neutral_id_text):
        errors.append(f"{case['case_id']}: input ID encodes a role or mechanism")
    if len(case["extraction_line_ids"]) != 4 or len(set(case["extraction_line_ids"])) != 4:
        errors.append(f"{case['case_id']}: extraction IDs must be four unique lines")
    if case["source_text"] != render_case(case, False):
        errors.append(f"{case['case_id']}: source rendering drift")
    if case["variant_text"] != render_case(case, True):
        errors.append(f"{case['case_id']}: variant rendering drift")
    for hand in case["mechanics_ledger"]["hands"]:
        errors.extend(validate_hand(hand))
    all_ids = _all_line_ids(case)
    if not set(case["extraction_line_ids"]).issubset(all_ids):
        errors.append(f"{case['case_id']}: extraction line absent from source")
    expected = case["expected_facts"]
    if [a["line_id"] for a in expected["actions"]] != case["extraction_line_ids"]:
        errors.append(f"{case['case_id']}: expected action order differs from requested order")
    if sum(expected["net_chips"].values()) != 0:
        errors.append(f"{case['case_id']}: expected net chips are not zero-sum")
    for outcome in expected["outcomes"]:
        if len(outcome["winner_aliases"]) != 1:
            errors.append(f"{case['case_id']}: fixture unexpectedly contains a split pot")
    note = case["author_mechanism_note"]
    if not note.get("non_gold") or any(tag not in ALLOWED_TAGS for tag in note.get("suggested_tags", [])):
        errors.append(f"{case['case_id']}: invalid non-gold author mechanism note")
    prompt = build_prompt(case)
    if len(prompt) > 12000 or len(prompt.split()) > 3000:
        errors.append(f"{case['case_id']}: prompt exceeds conservative 3,000-token proxy")
    return errors


def validate_bundle(cases: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(cases) != 12:
        errors.append(f"expected 12 cases, found {len(cases)}")
    for field in ("case_id", "hand_ids"):
        flat: list[str] = []
        for case in cases:
            value = case[field]
            flat.extend(value if isinstance(value, list) else [value])
        if len(flat) != len(set(flat)):
            errors.append(f"duplicate {field}")
    roles = [case["design_role"] for case in cases]
    if roles.count("matched_explanation") != 6 or roles.count("mechanics_probe") != 6:
        errors.append("design roles are not balanced 6/6")
    pair_ids = sorted({case["pair_id"] for case in cases})
    if len(pair_ids) != 6 or any(sum(c["pair_id"] == pid for c in cases) != 2 for pid in pair_ids):
        errors.append("expected six two-case pairs")
    all_line_ids: list[str] = []
    for case in cases:
        errors.extend(validate_case(case))
        all_line_ids.extend(_all_line_ids(case))
    if len(all_line_ids) != len(set(all_line_ids)):
        errors.append("line IDs are not globally unique")
    indistinguishable = [c for c in cases if c["unidentifiable_pair"]]
    if len(indistinguishable) != 2:
        errors.append("expected one explicitly unidentifiable pair")
    else:
        def signature(c: dict[str, Any]) -> list[tuple[Any, ...]]:
            hand = c["mechanics_ledger"]["hands"][0]
            return [
                (x["type"], x["street"], x.get("actor"), x.get("action"), x.get("amount"), x.get("to_amount"), x.get("event_code"))
                for x in hand["chronology"]
            ]
        if signature(indistinguishable[0]) != signature(indistinguishable[1]):
            errors.append("unidentifiable pair does not share observable chronology")
        for case in indistinguishable:
            req = case["expected_facts"]["required_assessment"]
            if not req["abstain"] or req["minimum_uncertainty"] != "high":
                errors.append("unidentifiable pair must require high-uncertainty abstention")
    return errors


def _require_exact_keys(obj: Any, required: set[str], path: str, errors: list[str]) -> bool:
    if not isinstance(obj, dict):
        errors.append(f"{path}: expected object")
        return False
    extra = set(obj) - required
    missing = required - set(obj)
    if extra:
        errors.append(f"{path}: unexpected keys {sorted(extra)}")
    if missing:
        errors.append(f"{path}: missing keys {sorted(missing)}")
    return not extra and not missing


def _variant_expected(case: dict[str, Any]) -> dict[str, Any]:
    expected = copy.deepcopy(case["expected_facts"])
    source_to_role = {name: role for role, name in case["aliases"].items()}
    def remap(name: str) -> str:
        return case["variant_aliases"][source_to_role[name]]
    for action in expected["actions"]:
        action["actor_alias"] = remap(action["actor_alias"])
    expected["net_chips"] = {remap(name): value for name, value in expected["net_chips"].items()}
    for outcome in expected["outcomes"]:
        outcome["winner_aliases"] = [remap(name) for name in outcome["winner_aliases"]]
        for returned in outcome["returned_uncalled"]:
            returned["actor_alias"] = remap(returned["actor_alias"])
    return expected


def evaluate_response(case: dict[str, Any], response: str | dict[str, Any], variant: bool = False) -> list[str]:
    """Return factuality/schema errors. An empty list is a pass."""
    errors: list[str] = []
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as exc:
            return [f"response: invalid JSON: {exc}"]
    top_keys = {"schema_version", "case_id", "pair_id", "hand_ids", "cited_line_ids", "observed_facts", "unavailable_facts", "mechanism_assessment"}
    if not _require_exact_keys(response, top_keys, "response", errors):
        return errors
    for field, expected_value in (
        ("schema_version", SCHEMA_VERSION), ("case_id", case["case_id"]),
        ("pair_id", case["pair_id"]), ("hand_ids", case["hand_ids"]),
    ):
        if response[field] != expected_value:
            errors.append(f"{field}: expected {expected_value!r}, got {response[field]!r}")
    valid_ids = _all_line_ids(case)
    cited = response["cited_line_ids"]
    if not isinstance(cited, list) or len(cited) != len(set(cited)) or any(x not in valid_ids for x in cited):
        errors.append("cited_line_ids: contains duplicate or unknown ID")
    expected = _variant_expected(case) if variant else case["expected_facts"]
    facts = response["observed_facts"]
    if _require_exact_keys(facts, {"actions", "net_chips", "outcomes"}, "observed_facts", errors):
        if facts["actions"] != expected["actions"]:
            errors.append("observed_facts.actions: exact designated facts mismatch")
        if facts["net_chips"] != expected["net_chips"]:
            errors.append("observed_facts.net_chips: full-hand ledger mismatch")
        if facts["outcomes"] != expected["outcomes"]:
            errors.append("observed_facts.outcomes: terminal outcome mismatch")
        hand_outcome = case["mechanics_ledger"]["hands"][0]["outcome"]
        required_fact_ids = set(case["extraction_line_ids"]) | {hand_outcome["line_id"]}
        required_fact_ids.update(item["line_id"] for item in hand_outcome["returned_uncalled"])
        if isinstance(cited, list) and not required_fact_ids.issubset(cited):
            errors.append("cited_line_ids: must include extracted actions, outcome, and any return line")
    if response["unavailable_facts"] != expected["unavailable_facts"]:
        errors.append("unavailable_facts: exact set/order mismatch")
    assessment = response["mechanism_assessment"]
    assessment_keys = {"observable_tags", "evidence_for", "evidence_against", "uncertainty", "abstain", "abstention_reason"}
    if _require_exact_keys(assessment, assessment_keys, "mechanism_assessment", errors):
        tags = assessment["observable_tags"]
        if not isinstance(tags, list) or not tags or len(tags) != len(set(tags)) or any(tag not in ALLOWED_TAGS for tag in tags):
            errors.append("mechanism_assessment.observable_tags: invalid enum or duplicate")
        elif "none" in tags and len(tags) != 1:
            errors.append("mechanism_assessment.observable_tags: none cannot be combined")
        uncertainty = assessment["uncertainty"]
        minimum = expected["required_assessment"]["minimum_uncertainty"]
        if uncertainty not in UNCERTAINTY_ORDER or UNCERTAINTY_ORDER.get(uncertainty, -1) < UNCERTAINTY_ORDER[minimum]:
            errors.append(f"mechanism_assessment.uncertainty: must be at least {minimum}")
        if assessment["abstain"] is not expected["required_assessment"]["abstain"]:
            errors.append("mechanism_assessment.abstain: mismatch")
        reason = assessment["abstention_reason"]
        if assessment["abstain"] and (not isinstance(reason, str) or not reason.strip()):
            errors.append("mechanism_assessment.abstention_reason: required for abstention")
        for direction in ("evidence_for", "evidence_against"):
            evidence = assessment[direction]
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"mechanism_assessment.{direction}: at least one item required")
                continue
            for index, item in enumerate(evidence):
                if not _require_exact_keys(item, {"claim", "cited_line_ids"}, f"{direction}[{index}]", errors):
                    continue
                if not isinstance(item["claim"], str) or not item["claim"].strip():
                    errors.append(f"{direction}[{index}].claim: non-empty prose required")
                ids = item["cited_line_ids"]
                if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or any(x not in valid_ids for x in ids):
                    errors.append(f"{direction}[{index}].cited_line_ids: duplicate, empty, or unknown")
    return errors


def _canonical_observed(case: dict[str, Any], response: dict[str, Any], variant: bool) -> dict[str, Any]:
    aliases = case["variant_aliases"] if variant else case["aliases"]
    to_role = {name: role for role, name in aliases.items()}
    facts = copy.deepcopy(response["observed_facts"])
    for action in facts["actions"]:
        action["actor_alias"] = to_role[action["actor_alias"]]
    facts["net_chips"] = {to_role[name]: value for name, value in facts["net_chips"].items()}
    for outcome in facts["outcomes"]:
        outcome["winner_aliases"] = [to_role[name] for name in outcome["winner_aliases"]]
        for returned in outcome["returned_uncalled"]:
            returned["actor_alias"] = to_role[returned["actor_alias"]]
    return facts

def evaluate_invariance(
    case: dict[str, Any], source_response: dict[str, Any], variant_response: dict[str, Any]
) -> list[str]:
    """Check name/presentation invariance after each response passes factuality."""
    errors = [f"source: {e}" for e in evaluate_response(case, source_response, False)]
    errors.extend(f"variant: {e}" for e in evaluate_response(case, variant_response, True))
    if errors:
        return errors
    if _canonical_observed(case, source_response, False) != _canonical_observed(case, variant_response, True):
        errors.append("canonical observed facts changed under alias/presentation variant")
    source_assessment = source_response["mechanism_assessment"]
    variant_assessment = variant_response["mechanism_assessment"]
    for field in ("observable_tags", "uncertainty", "abstain"):
        if source_assessment[field] != variant_assessment[field]:
            errors.append(f"mechanism_assessment.{field} changed under variant")
    if source_response["unavailable_facts"] != variant_response["unavailable_facts"]:
        errors.append("unavailable_facts changed under variant")
    return errors

def reference_response(case: dict[str, Any], variant: bool = False) -> dict[str, Any]:
    expected = _variant_expected(case) if variant else case["expected_facts"]
    all_ids = _all_line_ids(case)
    omission_id = next(x for x in sorted(all_ids) if x.startswith(case["case_id"].replace("tm_case_", "tm_m")) and x.endswith("_05"))
    outcome = case["mechanics_ledger"]["hands"][0]["outcome"]
    outcome_id = outcome["line_id"]
    return_ids = [item["line_id"] for item in outcome["returned_uncalled"]]
    action_ids = list(case["extraction_line_ids"])
    required = expected["required_assessment"]
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case["case_id"], "pair_id": case["pair_id"], "hand_ids": case["hand_ids"],
        "cited_line_ids": action_ids + return_ids + [outcome_id, omission_id],
        "observed_facts": {
            "actions": copy.deepcopy(expected["actions"]),
            "net_chips": copy.deepcopy(expected["net_chips"]),
            "outcomes": copy.deepcopy(expected["outcomes"]),
        },
        "unavailable_facts": list(expected["unavailable_facts"]),
        "mechanism_assessment": {
            "observable_tags": list(case["author_mechanism_note"]["suggested_tags"]),
            "evidence_for": [{"claim": case["author_mechanism_note"]["evidence_seed"], "cited_line_ids": [action_ids[-1], outcome_id]}],
            "evidence_against": [{"claim": "Private intent and off-excerpt history are explicitly unavailable.", "cited_line_ids": [omission_id]}],
            "uncertainty": required["minimum_uncertainty"], "abstain": True,
            "abstention_reason": "The observed mechanics do not identify private intent.",
        },
    }


def run_self_tests(cases: list[dict[str, Any]]) -> list[str]:
    errors = validate_bundle(cases)
    for case in cases:
        reference = reference_response(case)
        response_errors = evaluate_response(case, reference)
        if response_errors:
            errors.append(f"{case['case_id']}: valid reference rejected: {response_errors}")
        variant_reference = reference_response(case, variant=True)
        variant_errors = evaluate_response(case, variant_reference, variant=True)
        if variant_errors:
            errors.append(f"{case['case_id']}: valid variant reference rejected: {variant_errors}")
        invariance_errors = evaluate_invariance(case, reference, variant_reference)
        if invariance_errors:
            errors.append(f"{case['case_id']}: valid invariant pair rejected: {invariance_errors}")
    first = cases[0]
    valid = reference_response(first)
    bad_id = copy.deepcopy(valid)
    bad_id["cited_line_ids"][0] = "tm_l999_99"
    if not evaluate_response(first, bad_id):
        errors.append("negative test failed: unknown citation ID accepted")
    bad_amount = copy.deepcopy(valid)
    bad_amount["observed_facts"]["actions"][0]["amount"] += 1
    if not evaluate_response(first, bad_amount):
        errors.append("negative test failed: incorrect action amount accepted")
    bad_net = copy.deepcopy(valid)
    first_alias = next(iter(bad_net["observed_facts"]["net_chips"]))
    bad_net["observed_facts"]["net_chips"][first_alias] += 1
    if not evaluate_response(first, bad_net):
        errors.append("negative test failed: incorrect net accepted")
    corrupt_hand = copy.deepcopy(first["mechanics_ledger"]["hands"][0])
    corrupt_hand["outcome"]["pot"] += 1
    if not validate_hand(corrupt_hand):
        errors.append("negative test failed: corrupt terminal pot accepted")
    corrupt_return = copy.deepcopy(first["mechanics_ledger"]["hands"][0])
    next(item for item in corrupt_return["chronology"] if item["type"] == "return")["amount"] += 1
    if not validate_hand(corrupt_return):
        errors.append("negative test failed: incorrect uncalled-chip return accepted")
    bad_variant = reference_response(first, variant=True)
    bad_variant["mechanism_assessment"]["observable_tags"] = ["isolation_pattern"]
    if not evaluate_invariance(first, valid, bad_variant):
        errors.append("negative test failed: variant tag drift accepted")
    corrupt_cards = copy.deepcopy(first["mechanics_ledger"]["hands"][0])
    corrupt_cards["known_cards"] = ["As", "As"]
    if not validate_hand(corrupt_cards):
        errors.append("negative test failed: duplicate known cards accepted")
    return errors


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_artifacts(cases: list[dict[str, Any]]) -> list[Path]:
    """Write once with exclusive creation; refusing to overwrite freezes reviewed files."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, bytes] = {
        "cases.json": _json_bytes({"fixture_version": SCHEMA_VERSION, "cases": cases}),
        "response_schema.json": _json_bytes(RESPONSE_SCHEMA),
        "reference_responses.json": _json_bytes({
            "schema_version": SCHEMA_VERSION,
            "responses": [reference_response(case) for case in cases],
        }),
        "prompt_template.txt": PROMPT_TEMPLATE.encode("utf-8"),
    }
    written: list[Path] = []
    for name, payload in artifacts.items():
        path = FIXTURE_DIR / name
        with path.open("xb") as handle:
            handle.write(payload)
        written.append(path)
    hashes = "".join(f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in sorted(artifacts.items()))
    hash_path = FIXTURE_DIR / "HASHES.sha256"
    with hash_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(hashes)
    written.append(hash_path)
    readme = (
        "# Synthetic teacher-mechanics pilot\n\n"
        "These 12 original synthetic cases contain no real poker records, predictions, labels, or teacher outputs. "
        "They form six matched two-case patterns. `design_role` is fixture-only construction metadata and is never placed in a prompt. "
        "Observable mechanism tags are not intent or guilt labels. Every case requires abstention from intent; pair 06 is deliberately observationally unidentifiable and requires high uncertainty.\n\n"
        "Amounts are additional chips committed by an action. Only a raise has `to_amount`, the resulting street commitment. "
        "The reported pot is the called pot: any unmatched terminal bet or raise is emitted and checked as a separate `returned_uncalled` fact. "
        "Net chips cover both the return and terminal award; split pots are excluded. Complete card identities, intent, private communications, and outside history are explicitly unavailable.\n\n"
        "Run `python teacher_mechanics_fixtures.py --self-test` before inference. The standalone `evaluate_response` function is the factuality gate. "
        "It rejects wrong IDs, actions, amounts, returns, full-hand nets, outcomes, unavailable-fact sets, missing abstention, and source/variant factual drift. "
        "Evidence prose is checked only for non-empty text and valid citations. `author_mechanism_note` tags are optional descriptions excluded from the factual admission gate; they are neither coordination nor guilt labels.\n\n"
        "All cases require intent abstention, so this pilot measures instruction following, chronology, arithmetic, source grounding, and alias invariance. "
        "The one-hand examples cannot establish benign-versus-cheating discrimination, calibrated suspicion, or student utility. Those require a separately frozen experiment.\n\n"
        "`HASHES.sha256` freezes the fixture, prompt, response schema, and reference responses. Root review of these files and hashes is required before any teacher inference. "
        "Use one fixed greedy non-thinking decode, a 1,024-token output cap, and no post-hoc prompt repair or variant sampling.\n"
    )
    readme_path = FIXTURE_DIR / "README.md"
    with readme_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(readme)
    written.append(readme_path)
    return written


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--write-fixtures", action="store_true")
    parser.add_argument("--preview-pair", type=int)
    args = parser.parse_args(argv)
    cases = build_cases()
    errors = run_self_tests(cases)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    if args.preview_pair is not None:
        selected = [case for case in cases if case["pair_id"] == f"tm_pair_{args.preview_pair:02d}"]
        print(json.dumps(selected, indent=2, sort_keys=True))
    if args.write_fixtures:
        for path in write_artifacts(cases):
            print(path)
    if args.self_test or not (args.preview_pair or args.write_fixtures):
        max_chars = max(len(build_prompt(case)) for case in cases)
        max_words = max(len(build_prompt(case).split()) for case in cases)
        print(f"PASS: {len(cases)} cases; max prompt {max_chars} chars/{max_words} whitespace tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
