"""Fresh artifact integrity, exact change scope, and full legal-hand validation."""
import runtime
from runtime import ROOT
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
from validation import validate_submission, EVIDENCE_COLUMNS

HERE = Path(__file__).resolve().parent
CV = ROOT / "Claude Version"
FILES = {
    "incumbent272": ("outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv", "d5891a68919d51e2204d72c1515950e7b01b64b6c1b3242a5bc3b37ca18a53f5"),
    "candidate276": ("outputs_step276_20260919/submission_step276_cigamma2_on_step272.csv", "3c38a93ed93db1c59c85ce235df85081d2b1ee3124170e860469768611c3045d"),
    "candidate277": ("outputs_step277_20260919/submission_step277_spRule_on_step272.csv", "462da01535bf45e9d6a7a67043ea0f9cfc52a4fe1e4e3864ee55a9f7252a32f8"),
}
results = {"checked_utc": datetime.now(timezone.utc).isoformat(), "candidates": {}}
base = pl.read_csv(CV / FILES["incumbent272"][0], infer_schema=False)
for name, (relative, expected) in FILES.items():
    path = CV / relative
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == expected, (name, before)
    frame = pl.read_csv(path, infer_schema=False)
    assert frame.select("pair_id", "risk_score", "predicted_behavior").equals(base.select("pair_id", "risk_score", "predicted_behavior"))
    changed = pl.Series("changed", [a != b for a, b in zip(frame.select(EVIDENCE_COLUMNS).iter_rows(), base.select(EVIDENCE_COLUMNS).iter_rows())])
    scope = frame.filter(changed).group_by("predicted_behavior").len().to_dicts()
    allowed = {"incumbent272": None, "candidate276": "coordinated_isolation", "candidate277": "soft_play"}[name]
    assert all(row["predicted_behavior"] == allowed for row in scope), scope
    report = validate_submission(path, ROOT / "data/evaluation_pairs.csv", ROOT / "data/hands.parquet", ROOT / "data/seats.parquet", ROOT / "data/sample_submission.csv")
    assert report.valid, report.to_dict()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    results["candidates"][name] = {"path": str(path), "sha256": before, "scope": scope, "risk_behavior_exact": True, "validation": report.to_dict()}
    (HERE / "ready_validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(name, json.dumps(results["candidates"][name]), flush=True)

