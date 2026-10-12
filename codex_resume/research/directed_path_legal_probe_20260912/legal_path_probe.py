"""One-shot fixed synthetic probe of the HF10 directed-path symmetry mismatch.

No real poker data, labels, model code, inference, or external observations are used.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import statistics
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
POKER_ROOT = HERE.parents[2]
SOURCE_RECEIPT = HERE / "SOURCES.json"
OUTPUTS = (HERE / "REPORT.json", HERE / "REPORT.md", HERE / "RUN_RECEIPT.json")
EPS = 1e-9
STRENGTHS = (1, 2, 4)
KINDS = ("one_way", "reversed", "disconnected", "shared_receiver", "no_path", "direct_transfer")
REVIEW_PAIR = ("A", "B")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_teacher_validator():
    path = POKER_ROOT / "Codex Resume/research/teacher_mechanics_20260912/src/teacher_mechanics_fixtures.py"
    spec = importlib.util.spec_from_file_location("frozen_teacher_mechanics_fixtures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_hand


def action(line_id: str, street: str, actor: str, move: str, amount: int = 0, to_amount: int | None = None) -> dict[str, Any]:
    return {"type": "action", "line_id": line_id, "street": street, "actor": actor,
            "action": move, "amount": amount, "to_amount": to_amount}


def hand_ledger(hand: dict[str, Any]) -> tuple[dict[str, int], dict[str, int], int]:
    contributions = {p: 0 for p in hand["players"]}
    for item in hand["chronology"]:
        if item["type"] == "action":
            contributions[item["actor"]] += item["amount"]
        elif item["type"] == "return":
            contributions[item["actor"]] -= item["amount"]
    pot = sum(contributions.values())
    net = {p: -contributions[p] for p in hand["players"]}
    net[hand["outcome"]["winner"]] += pot
    return contributions, net, pot


def make_leg(scenario_id: str, hand_number: int, payer: str, receiver: str, bystander: str,
             strength: int, bankroll: dict[str, int]) -> dict[str, Any]:
    """Create a legal three-player leg: payer builds a called pot, receiver bets river, payer folds."""
    unit = 10 * strength
    preflop_total = 2 * unit
    prefix = f"{scenario_id}_h{hand_number:02d}"
    players = [payer, bystander, receiver]  # seat order; payer is button, then SB, then BB
    chronology = [
        action(f"{prefix}_l01", "preflop", bystander, "post_small_blind", 5),
        action(f"{prefix}_l02", "preflop", receiver, "post_big_blind", 10),
        action(f"{prefix}_l03", "preflop", payer, "raise", preflop_total, preflop_total),
        action(f"{prefix}_l04", "preflop", bystander, "fold"),
        action(f"{prefix}_l05", "preflop", receiver, "call", preflop_total - 10),
        action(f"{prefix}_l06", "flop", receiver, "check"),
        action(f"{prefix}_l07", "flop", payer, "bet", unit),
        action(f"{prefix}_l08", "flop", receiver, "call", unit),
        action(f"{prefix}_l09", "turn", receiver, "check"),
        action(f"{prefix}_l10", "turn", payer, "bet", unit),
        action(f"{prefix}_l11", "turn", receiver, "call", unit),
        action(f"{prefix}_l12", "river", receiver, "bet", unit),
        action(f"{prefix}_l13", "river", payer, "fold"),
        {"type": "return", "line_id": f"{prefix}_l14", "street": "river", "actor": receiver, "amount": unit},
    ]
    hand = {
        "hand_id": f"{prefix}", "players": players,
        "seat_order": [{"seat": i + 1, "player": p} for i, p in enumerate(players)],
        "starting_stacks": {p: bankroll[p] for p in players},
        "button": payer, "small_blind": 5, "big_blind": 10,
        "chronology": chronology, "known_cards": [],
        "outcome": {
            "line_id": f"{prefix}_l15", "winner": receiver, "pot": 0, "showdown": False,
            "returned_uncalled": [{"line_id": f"{prefix}_l14", "actor": receiver, "amount": unit}],
            "terminal_awards": [], "side_pots": [],
        },
    }
    contributions, net, pot = hand_ledger(hand)
    hand["outcome"]["pot"] = pot
    hand["outcome"]["terminal_awards"] = [{"actor": receiver, "amount": pot}]
    hand["verified_contributions"] = contributions
    hand["verified_net"] = net
    return hand


def legs_for(kind: str) -> list[tuple[str, str, str]]:
    return {
        "one_way": [("A", "C", "B"), ("C", "B", "A")],
        "reversed": [("B", "C", "A"), ("C", "A", "B")],
        "disconnected": [("A", "C", "B"), ("D", "B", "A")],
        "shared_receiver": [("A", "C", "B"), ("B", "C", "A")],
        "no_path": [("C", "A", "B")],
        "direct_transfer": [("A", "B", "C")],
    }[kind]


def build_scenario(kind: str, strength: int) -> dict[str, Any]:
    scenario_id = f"fx_{kind}_s{strength}"
    involved = sorted({p for leg in legs_for(kind) for p in leg})
    bankroll = {p: 1000 for p in involved}
    initial = dict(bankroll)
    hands = []
    continuity = []
    for index, (payer, receiver, bystander) in enumerate(legs_for(kind), 1):
        before = dict(bankroll)
        hand = make_leg(scenario_id, index, payer, receiver, bystander, strength, bankroll)
        assert all(hand["starting_stacks"][p] == before[p] for p in hand["players"])
        for player, value in hand["verified_net"].items():
            bankroll[player] += value
        continuity.append({"hand_id": hand["hand_id"], "before": before,
                           "after": dict(bankroll), "seated": list(hand["players"])})
        hands.append(hand)
    return {"scenario_id": scenario_id, "kind": kind, "strength": strength,
            "review_pair": list(REVIEW_PAIR), "initial_bankroll": initial,
            "hands": hands, "bankroll_continuity": continuity, "final_bankroll": bankroll}


def observable_input(scenario: dict[str, Any]) -> dict[str, Any]:
    """The detector gets only complete observable hand histories and the reviewed pair."""
    return {"review_pair": scenario["review_pair"], "hands": scenario["hands"]}


def mechanics(observed: dict[str, Any]) -> dict[str, Any]:
    hands = observed["hands"]
    players = sorted({p for hand in hands for p in hand["players"]})
    index = {p: i for i, p in enumerate(players)}
    n = len(players)
    transfer = [[0.0] * n for _ in range(n)]  # T[i][k] = chips i took from k
    scale = [0.0] * n
    n_both = [[0] * n for _ in range(n)]
    all_nonzero_nets: list[float] = []
    hand_records = []
    for hand in hands:
        contributions, net, pot = hand_ledger(hand)
        assert net == hand["verified_net"] and contributions == hand["verified_contributions"]
        losses = {p: max(-v, 0) for p, v in net.items()}
        gains = {p: max(v, 0) for p, v in net.items()}
        loss_total = sum(losses.values())
        assert abs(sum(net.values())) < EPS and abs(sum(gains.values()) - loss_total) < EPS
        for p, c in contributions.items():
            scale[index[p]] += c
        for p in hand["players"]:
            for q in hand["players"]:
                n_both[index[p]][index[q]] += 1
        for winner, gain in gains.items():
            for loser, loss in losses.items():
                if winner != loser and loss_total:
                    transfer[index[winner]][index[loser]] += gain * loss / loss_total
        all_nonzero_nets.extend(abs(v) for v in net.values() if v)
        hand_records.append({"hand_id": hand["hand_id"], "pot": pot,
                             "contributions": contributions, "net": net})
    scale = [max(v, 1.0) for v in scale]
    norm = [[math.sqrt(scale[i] * scale[j]) for j in range(n)] for i in range(n)]
    flow = [[transfer[i][j] - transfer[j][i] for j in range(n)] for i in range(n)]
    fpos = [[max(flow[i][j], 0.0) / norm[i][j] for j in range(n)] for i in range(n)]
    pot_unit = max(float(statistics.median(all_nonzero_nets)), 1.0)
    zf = [[flow[i][j] / (pot_unit * math.sqrt(n_both[i][j] + 1.0)) for j in range(n)] for i in range(n)]

    def chain_matrix(edge: list[list[float]]) -> tuple[list[list[float]], list[list[str | None]]]:
        out = [[0.0] * n for _ in range(n)]
        via: list[list[str | None]] = [[None] * n for _ in range(n)]
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                for c in range(n):
                    value = min(edge[c][a], edge[b][c])  # a -> c -> b
                    if value > out[a][b]:
                        out[a][b], via[a][b] = value, players[c]
        return out, via

    chain, via = chain_matrix(fpos)
    chain_z, via_z = chain_matrix(zf)
    pa, pb = (index[p] for p in observed["review_pair"])
    old = min(chain[pa][pb], chain[pb][pa])
    corrected = max(chain[pa][pb], chain[pb][pa])
    old_z = min(chain_z[pa][pb], chain_z[pb][pa])
    corrected_z = max(chain_z[pa][pb], chain_z[pb][pa])
    selected_forward = chain[pa][pb] >= chain[pb][pa]
    src, dst = (pa, pb) if selected_forward else (pb, pa)
    mid_name = via[src][dst]
    mid = index[mid_name] if mid_name is not None else None
    leg_scores = None if mid is None else {
        "source_to_intermediary": fpos[mid][src], "intermediary_to_destination": fpos[dst][mid],
        "minimum": min(fpos[mid][src], fpos[dst][mid]),
        "roles": [players[src], players[mid], players[dst]],
    }
    return {
        "players": players, "hand_records": hand_records,
        "T": {players[i]: {players[j]: transfer[i][j] for j in range(n)} for i in range(n)},
        "flow": {players[i]: {players[j]: flow[i][j] for j in range(n)} for i in range(n)},
        "total_contribution_scale": dict(zip(players, scale)),
        "direct_review_pair": abs(flow[pa][pb]) / norm[pa][pb],
        "directed_chain": {f"{players[pa]}_to_{players[pb]}": chain[pa][pb],
                           f"{players[pb]}_to_{players[pa]}": chain[pb][pa]},
        "old_min_both": old, "corrected_max_either": corrected,
        "directed_chain_z": {f"{players[pa]}_to_{players[pb]}": chain_z[pa][pb],
                             f"{players[pb]}_to_{players[pa]}": chain_z[pb][pa]},
        "old_min_both_z": old_z, "corrected_max_either_z": corrected_z,
        "selected_legs": leg_scores,
        "selected_via_z": via_z[src][dst],
    }


def rename_observed(observed: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    value = deepcopy(observed)
    value["review_pair"] = [mapping[p] for p in value["review_pair"]]
    for hand in value["hands"]:
        hand["players"] = [mapping[p] for p in hand["players"]]
        for seat in hand["seat_order"]:
            seat["player"] = mapping[seat["player"]]
        hand["starting_stacks"] = {mapping[p]: v for p, v in hand["starting_stacks"].items()}
        hand["button"] = mapping[hand["button"]]
        for item in hand["chronology"]:
            item["actor"] = mapping[item["actor"]]
        hand["outcome"]["winner"] = mapping[hand["outcome"]["winner"]]
        for item in hand["outcome"]["returned_uncalled"]:
            item["actor"] = mapping[item["actor"]]
        for item in hand["outcome"]["terminal_awards"]:
            item["actor"] = mapping[item["actor"]]
        hand["verified_contributions"] = {mapping[p]: v for p, v in hand["verified_contributions"].items()}
        hand["verified_net"] = {mapping[p]: v for p, v in hand["verified_net"].items()}
    return value


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def run() -> dict[str, Any]:
    if any(path.exists() for path in OUTPUTS) or (HERE / "FAILURE.json").exists():
        raise RuntimeError("one-shot output already exists; refusing to rerun")
    receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    checked_hashes = {}
    for relative, expected in receipt["sha256"].items():
        actual = sha256_path(POKER_ROOT / relative)
        checked_hashes[relative] = actual
        assert actual == expected, f"source hash drift: {relative}"
    validate_hand = load_teacher_validator()
    scenarios = [build_scenario(kind, strength) for strength in STRENGTHS for kind in KINDS]
    metrics = {}
    validation = {"hands": 0, "returns": 0, "zero_sum_hands": 0, "continuous_transitions": 0,
                  "no_side_pots_or_all_ins": 0, "eligible_hands": 0, "role_permutations": 0}
    mapping = {"A": "B", "B": "C", "C": "A", "D": "D"}
    for scenario in scenarios:
        previous = dict(scenario["initial_bankroll"])
        for hand in scenario["hands"]:
            errors = validate_hand(hand)
            assert errors == [], f"frozen validator rejected {hand['hand_id']}: {errors}"
            assert set(REVIEW_PAIR).issubset(hand["players"]), f"ineligible cited hand {hand['hand_id']}"
            assert hand["outcome"]["side_pots"] == []
            assert all(hand["verified_contributions"][p] < hand["starting_stacks"][p] for p in hand["players"])
            assert sum(hand["verified_net"].values()) == 0
            assert all(hand["starting_stacks"][p] == previous[p] for p in hand["players"])
            for p, value in hand["verified_net"].items():
                previous[p] += value
            validation["hands"] += 1
            validation["returns"] += len(hand["outcome"]["returned_uncalled"])
            validation["zero_sum_hands"] += 1
            validation["continuous_transitions"] += 1
            validation["no_side_pots_or_all_ins"] += 1
            validation["eligible_hands"] += 1
        assert previous == scenario["final_bankroll"]
        observed = observable_input(scenario)
        score = mechanics(observed)
        metrics[scenario["scenario_id"]] = score
        renamed = mechanics(rename_observed(observed, mapping))
        for field in ("direct_review_pair", "old_min_both", "corrected_max_either",
                      "old_min_both_z", "corrected_max_either_z"):
            assert close(score[field], renamed[field]), f"role permutation drift: {scenario['scenario_id']} {field}"
        validation["role_permutations"] += 1

    for strength in STRENGTHS:
        one = metrics[f"fx_one_way_s{strength}"]
        rev = metrics[f"fx_reversed_s{strength}"]
        assert close(one["old_min_both"], 0.0) and one["corrected_max_either"] > 0
        assert close(rev["old_min_both"], 0.0) and rev["corrected_max_either"] > 0
        assert close(one["corrected_max_either"], rev["corrected_max_either"])
        assert close(one["corrected_max_either_z"], rev["corrected_max_either_z"])
        for kind in ("disconnected", "shared_receiver", "no_path", "direct_transfer"):
            assert close(metrics[f"fx_{kind}_s{strength}"]["corrected_max_either"], 0.0), kind
        direct = metrics[f"fx_direct_transfer_s{strength}"]
        assert direct["direct_review_pair"] > 0 and close(direct["corrected_max_either"], 0.0)
        for kind in ("one_way", "reversed"):
            item = metrics[f"fx_{kind}_s{strength}"]
            assert item["selected_legs"] is not None
            assert close(item["corrected_max_either"], item["selected_legs"]["minimum"])

    intent_checks = []
    for strength in STRENGTHS:
        scenario = next(x for x in scenarios if x["scenario_id"] == f"fx_one_way_s{strength}")
        observed = observable_input(scenario)
        # These interpretations are intentionally never arguments to mechanics().
        views = [
            {"non_observed_interpretation": "ordinary_or_innocent", "observed": observed},
            {"non_observed_interpretation": "coordinated", "observed": deepcopy(observed)},
        ]
        fingerprints = [hashlib.sha256(canonical_bytes(view["observed"])).hexdigest() for view in views]
        scores = [mechanics(view["observed"])["corrected_max_either"] for view in views]
        assert fingerprints[0] == fingerprints[1] and close(scores[0], scores[1])
        intent_checks.append({"strength": strength, "observable_fingerprint": fingerprints[0],
                              "scores": scores, "identical": True,
                              "interpretations_excluded_from_detector": True})

    compact = []
    for scenario in scenarios:
        score = metrics[scenario["scenario_id"]]
        compact.append({
            "scenario_id": scenario["scenario_id"], "kind": scenario["kind"], "strength": scenario["strength"],
            "hand_ids": [h["hand_id"] for h in scenario["hands"]],
            "initial_bankroll": scenario["initial_bankroll"], "final_bankroll": scenario["final_bankroll"],
            "terminal_nets": [h["verified_net"] for h in scenario["hands"]],
            "T": score["T"], "flow": score["flow"], "scale": score["total_contribution_scale"],
            "direct": score["direct_review_pair"], "directed_chain": score["directed_chain"],
            "old": score["old_min_both"], "corrected": score["corrected_max_either"],
            "old_z": score["old_min_both_z"], "corrected_z": score["corrected_max_either_z"],
            "selected_legs": score["selected_legs"],
        })
    path_rows = [x for x in compact if x["kind"] in ("one_way", "reversed")]
    control_rows = [x for x in compact if x["kind"] in ("disconnected", "shared_receiver", "no_path")]
    direct_rows = [x for x in compact if x["kind"] == "direct_transfer"]
    return {
        "scope": "Measured mechanics of fixed synthetic legal examples only; no real-world power, prevalence, hidden-positive recovery, or public gain.",
        "design": {"strengths": list(STRENGTHS), "kinds": list(KINDS), "review_pair": list(REVIEW_PAIR),
                   "intended_leg_called_chips": {str(s): 40 * s for s in STRENGTHS}},
        "source_hashes_verified": checked_hashes,
        "validation": validation,
        "fixed_example_mechanics": {
            "corrected_path_recovery": {"passed": sum(x["corrected"] > 0 and close(x["old"], 0.0) for x in path_rows),
                                        "total": len(path_rows)},
            "no_path_control_rejections": {"passed": sum(close(x["corrected"], 0.0) for x in control_rows),
                                            "total": len(control_rows)},
            "direct_control_separation": {"passed": sum(x["direct"] > 0 and close(x["corrected"], 0.0) for x in direct_rows),
                                          "total": len(direct_rows)},
            "direction_reversal_pairs_equal": 3,
        },
        "intent_identifiability": {"checks": intent_checks, "insensitive": True,
                                    "conclusion": "Observer-identical histories receive identical scores; the corrected feature cannot identify intent."},
        "leg_explainability": {"all_path_scores_equal_selected_known_leg_minimum": True,
                               "conclusion": "Every positive corrected path score is exactly the minimum of two observable directed-transfer leg scores."},
        "decision": "STOP fourth-type promotion. The corrected feature repairs the directional mechanics mismatch, but is intention-insensitive and entirely explained by known directed-transfer legs.",
        "future_experiment": "Only as a separate hypothesis: freeze and test an evidence feature tied to observable cross-hand linkage beyond the two transfer legs, with observer-identical intent controls. Do not relabel this path score as other_coordination.",
        "scenarios": compact,
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Directed-path legal probe — 12 September 2026", "",
        "**The direction fix passes all frozen legal examples, but it does not identify a fourth coordination type. Stop promotion.**",
        "",
        report["scope"], "",
        "## Measured fixed-example mechanics", "",
        "| check | result |", "|---|---:|",
    ]
    fm = report["fixed_example_mechanics"]
    lines += [
        f"| one-way/reversed paths: old=0 and corrected>0 | {fm['corrected_path_recovery']['passed']}/{fm['corrected_path_recovery']['total']} |",
        f"| disconnected/shared-receiver/no-path controls: corrected=0 | {fm['no_path_control_rejections']['passed']}/{fm['no_path_control_rejections']['total']} |",
        f"| direct-transfer controls: direct>0 and corrected path=0 | {fm['direct_control_separation']['passed']}/{fm['direct_control_separation']['total']} |",
        f"| direction-reversal equality | {fm['direction_reversal_pairs_equal']}/3 |", "",
        "All 30 hands passed the frozen legality validator, zero-sum ledger, persistent-stack/no-redraw check, no-side-pot/no-all-in check, exact return/award check, A/B evidence-eligibility check, and canonical role-renaming check. Each hand includes a real unmatched river bet and explicit return.", "",
        "At strengths 1, 2, and 4, each intended payer loses exactly 40, 80, or 160 called chips on its leg. Proportional terminal-net attribution retains the 5-chip folded blind spillover. `REPORT.json` records every terminal net, T matrix, antisymmetric flow, HF10 contribution scale, both directed chains, old minimum, corrected maximum, z variant, and selected legs.", "",
        "## Identifiability and decision", "",
        "The ordinary/innocent and coordinated interpretations use byte-identical observable histories at all three strengths. Interpretation metadata is never passed to the detector, and scores are identical. The feature is therefore insensitive to intention.", "",
        "Every positive corrected score is exactly the weaker of two normalized, observable directed-transfer legs chosen through C. The fix is a valid repair of the one-way-direction bug, but its signal is fully explained by known legs; it supplies no distinct evidence of `other_coordination`.", "",
        "**Decision: stop fourth-type promotion.** Toy recovery is not real-world detector power and says nothing about prevalence, hidden positives, evidence recovery, or public score.", "",
        "A future experiment is warranted only as a clearly separate hypothesis: pre-freeze an observable cross-hand linkage feature beyond the two transfer legs and test it with the same observer-identical intent controls. Do not rename this path score as a new unknown label.", "",
        "## Reproduction receipt", "",
        "`DESIGN.md` and `SOURCES.json` were written before the single execution. Input hashes matched. `RUN_RECEIPT.json` records the program hash and run time. Outputs were created exclusively, and reruns are refused. Root must inspect the source and results before relying on them.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    try:
        report = run()
        finished = datetime.now(timezone.utc).isoformat()
        program_hash = sha256_path(Path(__file__))
        with (HERE / "REPORT.json").open("xb") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        with (HERE / "REPORT.md").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(report_markdown(report))
        run_receipt = {"run_count": 1, "started_utc": started, "finished_utc": finished,
                       "status": "PASS", "program_sha256": program_hash,
                       "report_json_sha256": sha256_path(HERE / "REPORT.json"),
                       "report_md_sha256": sha256_path(HERE / "REPORT.md"),
                       "python": sys.version}
        with (HERE / "RUN_RECEIPT.json").open("xb") as handle:
            handle.write(json.dumps(run_receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        print(json.dumps({"status": "PASS", "decision": report["decision"],
                          "mechanics": report["fixed_example_mechanics"]}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"started_utc": started, "failed_utc": datetime.now(timezone.utc).isoformat(),
                   "status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc(),
                   "program_sha256": sha256_path(Path(__file__))}
        path = HERE / "FAILURE.json"
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(json.dumps(failure, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        print(json.dumps({"status": "FAIL", "error": repr(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
