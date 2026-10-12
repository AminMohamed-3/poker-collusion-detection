"""GATE-1 soundness checker (kill-gate harness, CPU only, 2 threads).

Usage (PowerShell, from repo root):
  $env:POLARS_MAX_THREADS=2; $env:OMP_NUM_THREADS=2
  & .\\.venv\\Scripts\\python.exe .opencode\\continuous\\tasks\\synthetic\\soundness_check.py [--data-dir data/synthetic] [--report <path.json>]

Exit codes: 0 PASS | 1 FAIL | 2 PENDING-ENGINE (no demo_*.parquet found).
Reads only; never writes outside --report. No Kaggle/VM/GPU.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

ALLOWED_ACTIONS = {"fold", "check", "call", "bet", "raise", "all_in"}
EPS = 1e-6

# Minimal declared schema the world engine must satisfy exactly (extra cols OK, flagged).
REQUIRED = {
    "hands": ["hand_id"],
    "seats": ["hand_id", "player_id"],
    "actions": ["hand_id", "seq_no", "player_id", "action"],
}

ROLE_HINTS = (("action", "actions"), ("seat", "seats"), ("hand", "hands"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_frame(path: str):
    try:
        import polars as pl

        return pl.read_parquet(path), "polars"
    except ImportError:
        import pandas as pd

        return pd.read_parquet(path), "pandas"


def cols_of(df, backend: str):
    return list(df.columns) if backend == "polars" else list(df.columns.tolist())


def to_records(df, backend: str, columns):
    if backend == "polars":
        return df.select(columns).to_dicts()
    return df[columns].to_dict(orient="records")


def classify(path: str, columns) -> str:
    low = " ".join(columns).lower() + " " + path.lower()
    for hint, role in ROLE_HINTS:
        if hint in low:
            return role
    if "seq_no" in columns or "action" in columns:
        return "actions"
    if "player_id" in columns and "stack_before" in low:
        return "seats"
    return "hands"


def check(path: str) -> dict:
    res = {"file": path, "role": None, "sha256": sha256_file(path),
           "sha_stable": sha256_file(path) == sha256_file(path),
           "schema": {}, "chip_conservation": {}, "action_legality": {}, "status": "PASS"}
    fails: list[str] = []
    skips: list[str] = []
    try:
        df, backend = load_frame(path)
    except Exception as e:  # noqa: BLE001
        res.update(status="FAIL", error=f"unreadable: {e}")
        return res
    columns = cols_of(df, backend)
    role = classify(path, columns)
    res["role"] = role
    missing = [c for c in REQUIRED[role] if c not in columns]
    extra = [c for c in columns if c not in REQUIRED[role]]
    res["schema"] = {"required": REQUIRED[role], "missing": missing, "extra": extra, "n_rows": len(df)}
    if missing:
        fails.append(f"schema missing {missing}")

    def has(*names):
        return all(n in columns for n in names)

    # --- chip conservation (seats role): sum(net) per hand ~= 0
    if role == "seats":
        if has("net_bb"):
            recs = to_records(df, backend, ["hand_id", "net_bb"])
            bad = 0
            hands: dict = {}
            for r in recs:
                try:
                    hands.setdefault(str(r["hand_id"]), 0.0)
                    v = r["net_bb"]
                    hands[str(r["hand_id"])] += 0.0 if v is None else float(v)
                except (TypeError, ValueError):
                    bad += 1
            viol = sorted(k for k, v in hands.items() if abs(v) > EPS)[:10]
            res["chip_conservation"] = {"hands": len(hands), "violations": viol, "non_numeric": bad}
            if viol or bad:
                fails.append(f"chip-conservation {len(viol)} hands, {bad} non-numeric")
        elif has("stack_before_bb", "stack_after_bb"):
            recs = to_records(df, backend, ["hand_id", "stack_before_bb", "stack_after_bb"])
            hands2: dict = {}
            for r in recs:
                try:
                    hands2.setdefault(str(r["hand_id"]), 0.0)
                    hands2[str(r["hand_id"])] += float(r["stack_after_bb"]) - float(r["stack_before_bb"])
                except (TypeError, ValueError):
                    pass
            viol = sorted(k for k, v in hands2.items() if abs(v) > EPS)[:10]
            res["chip_conservation"] = {"hands": len(hands2), "violations": viol, "via": "stack_after-before"}
            if viol:
                fails.append(f"chip-conservation {len(viol)} hands")
        else:
            res["chip_conservation"] = {"verdict": "SKIP", "reason": "no net_bb / stack cols"}
            skips.append("chip-conservation SKIP (no chip cols)")
    else:
        res["chip_conservation"] = {"verdict": "SKIP", "reason": f"role={role}"}

    # --- action legality (actions role)
    if role == "actions":
        want = [c for c in ("hand_id", "seq_no", "player_id", "action", "amount_bb",
                            "to_call_bb", "stack_before_bb", "street") if c in columns]
        recs = to_records(df, backend, want)
        by_hand: dict = {}
        for r in recs:
            by_hand.setdefault(str(r["hand_id"]), []).append(r)
        bad_action = sum(1 for r in recs if str(r["action"]) not in ALLOWED_ACTIONS)
        bad_amt, bad_stack, bad_seq, act_after_fold = 0, 0, 0, 0
        for rows in by_hand.values():
            try:
                seqs = [int(r["seq_no"]) for r in rows]
            except (TypeError, ValueError):
                bad_seq += 1
                continue
            if sorted(seqs) != seqs or len(set(seqs)) != len(seqs):
                bad_seq += 1
            folded: set = set()
            for r in sorted(rows, key=lambda x: int(x["seq_no"])):
                a = str(r["action"])
                amt = r.get("amount_bb")
                try:
                    amt_f = 0.0 if amt is None else float(amt)
                except (TypeError, ValueError):
                    bad_amt += 1
                    continue
                if not (amt_f >= -EPS and amt_f == amt_f):
                    bad_amt += 1
                if a == "fold" and abs(amt_f) > EPS:
                    bad_amt += 1
                if a in ("call", "bet", "raise", "all_in") and r.get("to_call_bb") is not None:
                    try:
                        tc = float(r["to_call_bb"])
                        if a == "call" and abs(amt_f - tc) > max(EPS, 1e-9 * abs(tc)):
                            bad_amt += 1
                        if a in ("bet", "raise") and amt_f + EPS < tc:
                            bad_amt += 1
                    except (TypeError, ValueError):
                        bad_amt += 1
                if r.get("stack_before_bb") is not None:
                    try:
                        if amt_f - float(r["stack_before_bb"]) > EPS:
                            bad_stack += 1
                    except (TypeError, ValueError):
                        bad_stack += 1
                pid = str(r["player_id"])
                if pid in folded:
                    act_after_fold += 1
                if a == "fold":
                    folded.add(pid)
        res["action_legality"] = {"hands": len(by_hand), "bad_action": bad_action, "bad_amount": bad_amt,
                                  "bad_stack": bad_stack, "bad_seq": bad_seq,
                                  "act_after_fold": act_after_fold}
        for k in ("bad_action", "bad_amount", "bad_stack", "bad_seq", "act_after_fold"):
            if res["action_legality"][k]:
                fails.append(f"action-legality {k}={res['action_legality'][k]}")
    else:
        res["action_legality"] = {"verdict": "SKIP", "reason": f"role={role}"}

    # --- ID leakage: opaque IDs only (format/order must not encode label)
    id_cols = [c for c in columns if "id" in c.lower()]
    res["id_columns"] = id_cols
    res["skips"] = skips
    res["failures"] = fails
    if fails:
        res["status"] = "FAIL"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "demo_*.parquet")))
    manifest = os.path.join(args.data_dir, "demo_manifest.json")
    manifest_info = None
    if os.path.exists(manifest):
        try:
            with open(manifest, encoding="utf-8") as f:
                m = json.load(f)
            manifest_info = {"seed": m.get("seed"), "engine_sha": m.get("engine_sha"),
                             "rerun_cmd": m.get("rerun_cmd"), "full_rerun": "PENDING-ENGINE"}
        except (OSError, ValueError) as e:  # noqa: BLE001
            manifest_info = {"error": str(e)}
    if not files:
        out = {"verdict": "PENDING-ENGINE", "reason": "no demo_*.parquet in " + args.data_dir,
               "manifest": manifest_info,
               "determinism": {"full_bit_exact_rerun": "PENDING-ENGINE (engine absent)"}}
        print(json.dumps(out, indent=2))
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        return 2
    per_file = [check(p) for p in files]
    verdict = "PASS" if all(r["status"] == "PASS" for r in per_file) else "FAIL"
    out = {"verdict": verdict, "n_files": len(files), "files": per_file, "manifest": manifest_info,
           "determinism": {"sha_stable_reread": all(r["sha_stable"] for r in per_file),
                           "full_bit_exact_rerun": "PENDING-ENGINE (needs engine rerun_cmd + seed)"}}
    print(json.dumps(out, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
