"""Rebuild the submitted composition from the replayed STEP272 and frozen evidence branches."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CV = ROOT / "Claude Version"
sys.path.insert(0, str(ROOT / "src"))
from validation import validate_submission

SOURCES = {
    "base": (HERE / "submission_step272_replayed.csv", "d5891a68919d51e2204d72c1515950e7b01b64b6c1b3242a5bc3b37ca18a53f5"),
    "sp": (CV / "outputs_step277_20260919/submission_step277_spRule_on_step272.csv", "462da01535bf45e9d6a7a67043ea0f9cfc52a4fe1e4e3864ee55a9f7252a32f8"),
    "ci": (CV / "outputs_step276_20260919/submission_step276_cigamma2_on_step272.csv", "3c38a93ed93db1c59c85ce235df85081d2b1ee3124170e860469768611c3045d"),
}
EXPECTED = "0493a807d81b7f3cc5ec54a74c86d08c362b72305fceee8e6af9044740990fc9"


def load(path: Path, expected: str):
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected
    reader = csv.reader(io.StringIO(payload.decode("utf-8")))
    return next(reader), list(reader)


loaded = {k: load(*v) for k, v in SOURCES.items()}
header = loaded["base"][0]
assert all(x[0] == header for x in loaded.values())
rows = {k: v[1] for k, v in loaded.items()}
out = []
changed = {"soft_play": 0, "coordinated_isolation": 0}
for base, sp, ci in zip(rows["base"], rows["sp"], rows["ci"]):
    assert base[:3] == sp[:3] == ci[:3]
    chosen = sp if base[2] == "soft_play" else ci if base[2] == "coordinated_isolation" else base
    if chosen != base:
        changed[base[2]] += 1
    out.append(chosen)
assert changed == {"soft_play": 3857, "coordinated_isolation": 278}
buffer = io.StringIO(newline="")
writer = csv.writer(buffer, lineterminator="\n")
writer.writerow(header)
writer.writerows(out)
payload = buffer.getvalue().encode("utf-8")
digest = hashlib.sha256(payload).hexdigest()
assert digest == EXPECTED, digest
target = HERE / "submission_current_replayed.csv"
target.write_bytes(payload)
report = validate_submission(target, ROOT / "data/evaluation_pairs.csv", ROOT / "data/hands.parquet",
                             ROOT / "data/seats.parquet", ROOT / "data/sample_submission.csv")
assert report.valid, report.to_dict()
receipt = {"output": str(target), "sha256": digest, "bytes": len(payload), "changed_rows": changed,
           "validation": report.to_dict(),
           "claim": "artifact replay from recovered STEP272 rank artifact plus frozen SP/CI branch CSVs; not raw retraining"}
(HERE / "current_composition_replay_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
