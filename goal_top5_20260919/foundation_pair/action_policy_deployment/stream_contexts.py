"""Bounded table-wise evaluation context extraction; contains no policy convention."""
from __future__ import annotations
import hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
import polars as pl
from conditional_action_features import INPUTS, action_class, card_features, extract_hand

RAW = ROOT / "data"
PAIR_HANDS = ROOT / "outputs_v2/hands_evaluation"
OUTPUT_COLUMNS = ["pair_id", "hand_id", "table_id", "direction", "action_no", "action_class",
                  "partner_strength", "observed_log_amount_bb", *INPUTS]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_style_counts(destination: Path):
    """Evaluation-phase public action totals; no labels and no pair IDs."""
    assert not destination.exists()
    hands = pl.scan_parquet(RAW / "hands.parquet").filter(pl.col("phase") == "evaluation").select("hand_id")
    actions = pl.scan_parquet(RAW / "actions.parquet").join(hands, on="hand_id", how="semi")
    kind = (pl.when(pl.col("action") == "fold").then(0)
            .when(pl.col("action") == "check").then(1)
            .when((pl.col("action") == "call") | ((pl.col("action") == "all_in") &
                                                   (pl.col("amount") <= pl.col("to_call")))).then(2)
            .otherwise(3))
    frame = actions.with_columns(kind.alias("kind")).group_by("player_id").agg(
        *[(pl.col("kind") == k).sum().alias(f"n{k}") for k in range(4)]).collect()
    frame.write_parquet(destination)
    return {"players": frame.height, "sha256": sha(destination)}

def table_ids():
    paths = sorted(PAIR_HANDS.glob("batch_*.parquet"))
    tables = sorted(set(pl.concat([pl.read_parquet(p, columns=["table_id"]).unique() for p in paths])
                        ["table_id"].to_list()))
    assert len(tables) == 400
    return tables

def _pairs_for_table(table_id):
    scans = [pl.scan_parquet(p).filter(pl.col("table_id") == table_id).select("pair_id", "hand_id")
             for p in sorted(PAIR_HANDS.glob("batch_*.parquet"))]
    keys = pl.concat(scans).collect()
    pairs = pl.read_csv(RAW / "evaluation_pairs.csv").select("pair_id", "player_1", "player_2")
    return keys.join(pairs, on="pair_id", validate="m:1")

def extract_table(table_id: str, style_counts_path: Path):
    """Materialize one table only. IDs are output keys, never model features."""
    keys = _pairs_for_table(table_id)
    ids = keys.select("hand_id").unique()
    hands = (pl.scan_parquet(RAW / "hands.parquet").filter(pl.col("phase") == "evaluation")
             .filter(pl.col("table_id") == table_id).select("hand_id", "table_id", "board_cards",
                                                            "button_seat", "big_blind").collect())
    assert hands["hand_id"].n_unique() == hands.height and set(ids["hand_id"]) <= set(hands["hand_id"])
    seats = pl.scan_parquet(RAW / "seats.parquet").join(ids.lazy(), on="hand_id", how="semi").collect()
    actions = (pl.scan_parquet(RAW / "actions.parquet").join(ids.lazy(), on="hand_id", how="semi")
               .collect().sort("hand_id", "action_no"))
    totals_frame = pl.read_parquet(style_counts_path)
    totals = {r[0]: np.asarray(r[1:], dtype=np.int64) for r in totals_frame.iter_rows()}
    hp, hs, ha = defaultdict(list), defaultdict(list), defaultdict(list)
    for pair, hand, first, second in keys.iter_rows(): hp[hand].append((pair, first, second))
    for row in seats.iter_rows(named=True): hs[row["hand_id"]].append(row)
    for row in actions.iter_rows(named=True): ha[row["hand_id"]].append(row)
    rows = []
    for hand in hands.iter_rows(named=True):
        hid = hand["hand_id"]; observed = ha[hid]
        current = defaultdict(lambda: np.zeros(4, dtype=np.int64))
        for action in observed: current[action["player_id"]][action_class(action)] += 1
        styles = {}
        for seat in hs[hid]:
            player = seat["player_id"]
            counts = totals.get(player, np.zeros(4, dtype=np.int64)) - current[player]
            assert counts.min() >= 0
            styles[player] = ((counts + 1) / (counts.sum() + 4)).tolist()
        rows.extend(extract_hand(hand, hs[hid], observed, hp[hid], styles))
    card_features.cache_clear()
    if not rows:
        return pl.DataFrame(schema={c: pl.String for c in OUTPUT_COLUMNS})
    frame = pl.DataFrame(rows).select(OUTPUT_COLUMNS).with_columns(
        pl.col(INPUTS + ["partner_strength", "observed_log_amount_bb"]).cast(pl.Float32))
    assert frame.select(pl.struct("pair_id", "direction", "hand_id", "action_no").n_unique()).item() == frame.height
    assert frame.select(pl.col(INPUTS).is_finite().all()).row(0) == (True,) * len(INPUTS)
    return frame
