"""Verify the 42 recovered chronology shards against their original temp copies."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = Path(
    r"C:\Users\USER\AppData\Local\Temp\claude\C--Data-Projects-Money-Making-work-poker"
    r"\f2bb4f17-323f-4806-98b3-984023c70212\scratchpad\step257"
)
RECOVERED = HERE / "recovered_chronology"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
for short, old_name in (("development", "chrono_development"), ("evaluation", "chrono_evaluation")):
    originals = sorted(p for p in (ORIGINAL / old_name).iterdir() if p.is_file())
    copies = sorted(p for p in (RECOVERED / short).iterdir() if p.is_file())
    assert [p.name for p in originals] == [p.name for p in copies]
    for old, new in zip(originals, copies):
        old_hash, new_hash = sha(old), sha(new)
        assert old_hash == new_hash and old.stat().st_size == new.stat().st_size
        rows.append({"phase": short, "name": old.name, "bytes": new.stat().st_size,
                     "sha256": new_hash, "original": str(old), "copy": str(new)})

receipt = {"verified_utc": datetime.now(timezone.utc).isoformat(), "files": len(rows),
           "bytes": sum(x["bytes"] for x in rows), "all_hashes_match": True, "members": rows}
(HERE / "recovered_chronology_manifest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps({k: receipt[k] for k in ("files", "bytes", "all_hashes_match")}, indent=2))
