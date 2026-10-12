"""Compose fixed, disjoint SP and CI evidence changes; retain exact CSV field text."""
import runtime
from runtime import ROOT
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from validation import validate_submission

HERE = Path(__file__).resolve().parent
CV = ROOT / "Claude Version"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
SPECS = {
    "base": ("outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv", "d5891a68919d51e2204d72c1515950e7b01b64b6c1b3242a5bc3b37ca18a53f5"),
    "sp": ("outputs_step277_20260919/submission_step277_spRule_on_step272.csv", "462da01535bf45e9d6a7a67043ea0f9cfc52a4fe1e4e3864ee55a9f7252a32f8"),
    "ci": ("outputs_step276_20260919/submission_step276_cigamma2_on_step272.csv", "3c38a93ed93db1c59c85ce235df85081d2b1ee3124170e860469768611c3045d"),
}
protocol = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Single public test of two previously built, family-disjoint evidence changes on STEP272",
    "source_sha256": {k: v[1] for k, v in SPECS.items()},
    "frozen_recipe": "SP rows from STEP277 at its existing weight .30; CI rows from STEP276 at its existing gamma2; all other rows and all risk/behavior from STEP272",
    "expectation": "Exploratory modest positive gain, primarily SP; prior SP estimate +.0003 to +.0008 composite, CI near public rounding/noise. Exact composition has no model interaction because row scopes are disjoint.",
    "uncertainties": "Reused audit; SP exact event-fusion composition not jointly validated; CI is a decoder adjustment with weak transfer history. Public/private gains not guaranteed.",
    "decision": "One fixed submission if allowance remains. Keep incumbent unchanged. No weight/gamma tuning from this result. Do not infer individual component labels from feedback.",
}
freeze = OUT / "composition_protocol.json"
if freeze.exists():
    old = json.loads(freeze.read_text(encoding="utf-8"))
    assert {k: v for k, v in old.items() if k != "created_utc"} == {k: v for k, v in protocol.items() if k != "created_utc"}
else:
    freeze.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

rows = {}
headers = {}
for key, (relative, expected) in SPECS.items():
    payload = (CV / relative).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected
    reader = csv.reader(io.StringIO(payload.decode("utf-8")))
    headers[key] = next(reader)
    rows[key] = list(reader)
assert headers["base"] == headers["sp"] == headers["ci"]
assert len(rows["base"]) == len(rows["sp"]) == len(rows["ci"]) == 112540

combined = []
counts = {"soft_play": 0, "coordinated_isolation": 0}
for base, sp, ci in zip(rows["base"], rows["sp"], rows["ci"]):
    assert len(base) == len(sp) == len(ci) == 8
    assert base[:3] == sp[:3] == ci[:3]
    family = base[2]
    assert sp == base or family == "soft_play"
    assert ci == base or family == "coordinated_isolation"
    chosen = sp if family == "soft_play" else ci if family == "coordinated_isolation" else base
    if chosen != base:
        counts[family] += 1
    combined.append(chosen)
assert counts == {"soft_play": 3857, "coordinated_isolation": 278}, counts
csv_path = OUT / "submission_sp277_ci276_on_step272.csv"
buffer = io.StringIO(newline="")
writer = csv.writer(buffer, lineterminator="\n")
writer.writerow(headers["base"])
writer.writerows(combined)
payload = buffer.getvalue().encode("utf-8")
if csv_path.exists():
    assert csv_path.read_bytes() == payload
else:
    csv_path.write_bytes(payload)
report = validate_submission(csv_path, ROOT / "data/evaluation_pairs.csv", ROOT / "data/hands.parquet", ROOT / "data/seats.parquet", ROOT / "data/sample_submission.csv")
assert report.valid, report.to_dict()
for key, (relative, expected) in SPECS.items():
    assert hashlib.sha256((CV / relative).read_bytes()).hexdigest() == expected
receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "path": str(csv_path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "changed_rows": counts, "total_changed": sum(counts.values()), "risk_behavior_field_text_exact": True, "disjoint_family_composition": True, "source_files_unchanged": True, "validation": report.to_dict(), "submitted": False}
(OUT / "composition_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
