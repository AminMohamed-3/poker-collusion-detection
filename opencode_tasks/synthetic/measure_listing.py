"""Listing-process measurements on DEV confirmed targets only (read-only inputs).

Reads: data/development_labels.csv, data/development_evidence.csv,
       data/hands.parquet, data/seats.parquet (all read-only).
Writes: listing_measurements.json (this folder only).
CPU-light: seats filtered to target members before any join; 2 threads.
"""
import json
import polars as pl

OUT = "listing_measurements.json"

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[4]  # synthetic->tasks->continuous->.opencode->poker
lab = pl.read_csv(ROOT / "data" / "development_labels.csv")
ev = pl.read_csv(ROOT / "data" / "development_evidence.csv")
targets = lab.filter(pl.col("label") == 1).select("pair_id", "player_1", "player_2", "behavior_family")
members = pl.concat([
    targets.select(pl.col("player_1").alias("player_id"), "pair_id"),
    targets.select(pl.col("player_2").alias("player_id"), "pair_id"),
])

seats = (
    pl.scan_parquet(ROOT / "data" / "seats.parquet")
    .filter(pl.col("player_id").is_in(members["player_id"].implode() if False else members.select("player_id").to_series().to_list()))
    .join(members.lazy(), on="player_id", how="inner")
    .collect()
)
# shared hands: both pair members seated
shared = (
    seats.group_by(["pair_id", "hand_id"])
    .agg(n_members=pl.col("player_id").n_unique())
    .filter(pl.col("n_members") == 2)
    .select("pair_id", "hand_id")
)
hands = (
    pl.scan_parquet(ROOT / "data" / "hands.parquet")
    .select("hand_id", "table_id", "started_at", "final_pot")
    .collect()
)
sh = shared.join(hands, on="hand_id", how="inner").sort(["pair_id", "started_at"])
sh = sh.with_columns(
    chrono_rank=pl.col("started_at").cum_count().over("pair_id") - 1,
    n_hands=pl.col("hand_id").count().over("pair_id"),
).with_columns(chrono_frac=pl.col("chrono_rank") / (pl.col("n_hands") - 1).clip(lower_bound=1))

listed = set(map(tuple, ev.select("pair_id", "hand_id").rows()))
sh = sh.with_columns(is_listed=pl.struct(["pair_id", "hand_id"]).map_elements(
    lambda s: (s["pair_id"], s["hand_id"]) in listed, return_dtype=pl.Boolean))

# per-pair table count (pool proxy)
tables = sh.group_by("pair_id").agg(n_tables=pl.col("table_id").n_unique(), n_shared=pl.len()).join(
    ev.group_by("pair_id").agg(n_listed=pl.len()), on="pair_id", how="left").fill_null(0)

# fold-event proxies from seats (member-level)
sm = seats.join(shared, on=["pair_id", "hand_id"], how="inner")


def hole_strong(h1, h2):
    if h1 is None or h2 is None:
        return False
    r = "23456789TJQKA"
    r1, r2 = h1[:-1], h2[:-1]
    suited = h1[-1] == h2[-1]
    pair = r1 == r2
    broad = r1 in "TJQKA" and r2 in "TJQKA"
    ace_hi = r1 == "A" or r2 == "A"
    return bool(pair or (suited and (broad or ace_hi)) or broad)


sm = sm.with_columns(
    strong_hole=pl.struct(["hole_card_1", "hole_card_2"]).map_elements(
        lambda s: hole_strong(s["hole_card_1"], s["hole_card_2"]), return_dtype=pl.Boolean)
)
pair_hand = sm.group_by(["pair_id", "hand_id"]).agg(
    any_fold=pl.col("folded").any(),
    n_fold=pl.col("folded").sum(),
    n_showdown=pl.col("went_to_showdown").sum(),
    strong_fold=((pl.col("folded") & pl.col("strong_hole")).sum() > 0),
)
# fold-while-partner-present: exactly one folded, other reached showdown
pair_hand = pair_hand.with_columns(
    fold_partner_present=(pl.col("n_fold") == 1) & (pl.col("n_showdown") == 1))
sh2 = sh.join(pair_hand, on=["pair_id", "hand_id"], how="left")

q = sh2.group_by("pair_id").agg(
    n_shared=pl.len(),
    n_listed=pl.col("is_listed").sum(),
    n_foldcand=pl.col("any_fold").sum(),
    n_fpp=pl.col("fold_partner_present").sum(),
    n_strongfold=pl.col("strong_fold").sum(),
).join(tables.select("pair_id", "n_tables"), on="pair_id")

lrow = sh2.filter(pl.col("is_listed"))
urow = sh2.filter(~pl.col("is_listed"))
listed_fracs = sorted(lrow["chrono_frac"].to_list())
dec = [listed_fracs[int(i * (len(listed_fracs) - 1) / 9)] for i in range(10)] if listed_fracs else []
quart = lrow.with_columns(q4=(pl.col("chrono_frac") * 4).floor().clip(0, 3).cast(pl.Int64)) \
    .group_by("q4").agg(n=pl.len()).sort("q4")
uquart = urow.with_columns(q4=(pl.col("chrono_frac") * 4).floor().clip(0, 3).cast(pl.Int64)) \
    .group_by("q4").agg(n=pl.len()).sort("q4")
qd = {r["q4"]: r["n"] for r in quart.rows(named=True)}
ud = {r["q4"]: r["n"] for r in uquart.rows(named=True)}
listed_rate_q = {str(k): round(qd.get(k, 0) / max(1, qd.get(k, 0) + ud.get(k, 0)), 4) for k in range(4)}

res = {
    "n_target_pairs": targets.height,
    "n_evidence_rows": ev.height,
    "per_pair": {
        "n_shared_median": float(q["n_shared"].median()),
        "n_shared_mean": float(q["n_shared"].mean()),
        "n_listed_median": float(q["n_listed"].median()),
        "n_listed_mean": float(q["n_listed"].mean()),
        "n_listed_hist": q.group_by("n_listed").agg(n=pl.len()).sort("n_listed").rows(named=True),
        "n_foldcand_median": float(q["n_foldcand"].median()),
        "n_foldcand_mean": float(q["n_foldcand"].mean()),
        "n_fpp_median": float(q["n_fpp"].median()),
        "n_fpp_mean": float(q["n_fpp"].mean()),
        "n_strongfold_median": float(q["n_strongfold"].median()),
        "n_strongfold_mean": float(q["n_strongfold"].mean()),
        "n_tables_all1": bool((q["n_tables"] == 1).all()),
    },
    "listed_chrono_frac_fullhist": {
        "median": float(lrow["chrono_frac"].median()),
        "mean": float(lrow["chrono_frac"].mean()),
        "deciles": [round(float(x), 4) for x in dec],
        "listed_rate_by_quartile": listed_rate_q,
        "unlisted_chrono_median": float(urow["chrono_frac"].median()),
    },
    "pot_bb": {
        "listed_median": float(lrow["final_pot"].median()),
        "unlisted_median": float(urow["final_pot"].median()),
    },
    "fold_proxy_rates": {
        "listed_any_fold": float(lrow["any_fold"].mean()),
        "unlisted_any_fold": float(urow["any_fold"].mean()),
        "listed_fpp": float(lrow["fold_partner_present"].mean()),
        "unlisted_fpp": float(urow["fold_partner_present"].mean()),
        "listed_strongfold": float(lrow["strong_fold"].mean()),
        "unlisted_strongfold": float(urow["strong_fold"].mean()),
    },
    "by_family": (
        sh2.join(targets.select("pair_id", "behavior_family"), on="pair_id")
        .group_by("behavior_family").agg(
            pairs=pl.col("pair_id").n_unique(),
            listed_chrono_med=pl.col("chrono_frac").filter(pl.col("is_listed")).median(),
            unlisted_chrono_med=pl.col("chrono_frac").filter(~pl.col("is_listed")).median(),
            listed_fpp=pl.col("fold_partner_present").filter(pl.col("is_listed")).mean(),
            unlisted_fpp=pl.col("fold_partner_present").filter(~pl.col("is_listed")).mean())
        .rows(named=True)
    ),
}
with open(OUT, "w") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps(res, indent=1, default=str)[:3000])
