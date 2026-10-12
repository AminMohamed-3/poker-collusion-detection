"""Deterministically select and extract five factual case packets from the submitted CSV.

This produces explanatory facts only. It does not assign or infer evaluation labels.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data"
SUB = ROOT / "goal_top5_20260919/reproduction/submission_current_replayed.csv"
EXPECTED = "0493a807d81b7f3cc5ec54a74c86d08c362b72305fceee8e6af9044740990fc9"
EVIDENCE = [f"evidence_hand_{i}" for i in range(1, 6)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(SUB) == EXPECTED
sub = pl.read_csv(SUB, infer_schema=False).with_columns(pl.col("risk_score").cast(pl.Float64))
eligible = sub.filter(pl.col("risk_score") >= 0.5)


def family_rows(family: str) -> pl.DataFrame:
    return eligible.filter(pl.col("predicted_behavior") == family).sort(
        ["risk_score", "pair_id"], descending=[True, False]
    )


selection = []
for family in ("directed_transfer", "soft_play"):
    rows = family_rows(family)
    assert rows.height >= 2
    picks = [("highest_risk", 0), ("median_risk", (rows.height - 1) // 2)]
    for rule, idx in picks:
        row = rows.row(idx, named=True)
        selection.append({"case_id": f"{family}_{rule}", "selection_rule": rule,
                          "family_population_ge_0_5": rows.height, **row})
ci = family_rows("coordinated_isolation")
assert ci.height >= 1
selection.append({"case_id": "coordinated_isolation_highest_risk", "selection_rule": "highest_risk",
                  "family_population_ge_0_5": ci.height, **ci.row(0, named=True)})

# Freeze selection before any raw hand table is opened.
selection_doc = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "submission": str(SUB), "submission_sha256": EXPECTED,
    "rule": "Within risk_score>=0.5: highest and lower-middle median after descending risk/pair_id for DT and SP; highest for CI.",
    "median_index": "floor((n-1)/2), zero-based, after risk descending and pair_id ascending tie break",
    "cases": selection,
}
(HERE / "case_selection.json").write_text(json.dumps(selection_doc, indent=2, default=str), encoding="utf-8")

pair_ids = [x["pair_id"] for x in selection]
pairs = pl.read_csv(DATA / "evaluation_pairs.csv").filter(pl.col("pair_id").is_in(pair_ids))
assert pairs.height == 5
pair_map = {r[0]: {"player_1": r[1], "player_2": r[2]} for r in pairs.select("pair_id", "player_1", "player_2").iter_rows()}
hand_ids = [x[c] for x in selection for c in EVIDENCE]
assert len(hand_ids) == 25 and "NO_EVIDENCE" not in hand_ids
hands = (pl.scan_parquet(DATA / "hands.parquet").filter(pl.col("hand_id").is_in(hand_ids))
         .with_columns(pl.col("started_at").cast(pl.String)).collect())
seats = pl.scan_parquet(DATA / "seats.parquet").filter(pl.col("hand_id").is_in(hand_ids)).collect()
actions = (pl.scan_parquet(DATA / "actions.parquet").filter(pl.col("hand_id").is_in(hand_ids))
           .sort(["hand_id", "action_no"]).collect())
assert hands.height == 25 and hands["hand_id"].n_unique() == 25

facts = []
for case in selection:
    p1, p2 = pair_map[case["pair_id"]].values()
    case_hands = []
    for rank, hid in enumerate([case[c] for c in EVIDENCE], 1):
        h = hands.filter(pl.col("hand_id") == hid).row(0, named=True)
        ss = seats.filter((pl.col("hand_id") == hid) & pl.col("player_id").is_in([p1, p2])).sort("seat_no")
        assert ss.height == 2, (case["pair_id"], hid, ss.height)
        aa = actions.filter(pl.col("hand_id") == hid).sort("action_no")
        last_aggressor = {}
        interaction = []
        compact_actions = []
        for a in aa.iter_rows(named=True):
            compact_actions.append({k: a[k] for k in ("action_no", "street", "player_id", "action", "amount", "pot_before", "to_call", "players_active")})
            street = a["street"]
            if a["player_id"] in (p1, p2):
                event = {k: a[k] for k in ("action_no", "street", "player_id", "action", "amount", "pot_before", "to_call", "players_active")}
                event["last_aggressor_on_street"] = last_aggressor.get(street)
                event["partner_was_last_aggressor"] = last_aggressor.get(street) in (p1, p2) and last_aggressor.get(street) != a["player_id"]
                interaction.append(event)
            aggressive = a["action"] in ("bet", "raise") or (a["action"] == "all_in" and a["amount"] > a["to_call"])
            if aggressive:
                last_aggressor[street] = a["player_id"]
        case_hands.append({
            "evidence_rank": rank,
            "hand": {k: (str(v) if k == "started_at" else v) for k, v in h.items()},
            "pair_seats": ss.to_dicts(),
            "pair_actions": interaction,
            "all_actions": compact_actions,
        })
    facts.append({"case_id": case["case_id"], "pair_id": case["pair_id"], "risk_score": float(case["risk_score"]),
                  "predicted_behavior": case["predicted_behavior"], "player_1": p1, "player_2": p2,
                  "selection_rule": case["selection_rule"], "hands": case_hands})

(HERE / "case_facts.json").write_text(json.dumps({"cases": facts}, indent=2, default=str), encoding="utf-8")

flat = []
for case in facts:
    for item in case["hands"]:
        h = item["hand"]
        flat.append({"case_id": case["case_id"], "pair_id": case["pair_id"], "predicted_behavior": case["predicted_behavior"],
                     "risk_score": case["risk_score"], "evidence_rank": item["evidence_rank"], "hand_id": h["hand_id"],
                     "phase": h["phase"], "final_pot": h["final_pot"], "players_at_showdown": h["players_at_showdown"],
                     "pair_action_count": len(item["pair_actions"])})
pl.DataFrame(flat).write_csv(HERE / "case_facts.csv")

checks = {
    "submission_sha256": EXPECTED, "case_count": len(facts), "evidence_hands": len(flat),
    "all_hands_exactly_match_submission": True, "all_hands_evaluation_phase": all(x["phase"] == "evaluation" for x in flat),
    "both_pair_members_seated_in_every_hand": True, "no_duplicate_hand_within_case": all(
        len({x["hand"]["hand_id"] for x in c["hands"]}) == 5 for c in facts),
    "selection_frozen_before_raw_read": True, "creates_labels": False,
}
print(json.dumps(checks, indent=2), flush=True)
assert not checks["creates_labels"] and all(
    v for k, v in checks.items() if isinstance(v, bool) and k != "creates_labels"
), checks
(HERE / "case_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
