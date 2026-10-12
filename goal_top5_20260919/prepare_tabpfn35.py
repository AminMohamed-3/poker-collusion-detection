"""Stage verified public model and source in a separate project directory."""
import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tabpfn35"
receipt = json.loads((OUT / "source_receipt.json").read_text(encoding="utf-8"))
wheel = OUT / receipt["wheel"]["filename"]
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == receipt["wheel"]["digests"]["sha256"]
vendor = (OUT / "vendor").resolve()
vendor.mkdir(exist_ok=True)
with zipfile.ZipFile(wheel) as archive:
    for member in archive.infolist():
        target = (vendor / member.filename).resolve()
        assert target.is_relative_to(vendor), member.filename
    archive.extractall(vendor)

model = OUT / "tabpfn-v3.5-fast-20260909.safetensors"
if not model.exists():
    partial = model.with_suffix(".partial")
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(receipt["model_url"], timeout=60) as response, partial.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            if size % (64 * 1024 * 1024) == 0:
                print(f"Downloaded {size // (1024 * 1024)} MiB", flush=True)
    assert digest.hexdigest() == receipt["expected_model_sha256"], digest.hexdigest()
    partial.replace(model)
digest = hashlib.sha256(model.read_bytes()).hexdigest()
assert digest == receipt["expected_model_sha256"]
result = {"prepared_utc": datetime.now(timezone.utc).isoformat(), "model": str(model), "model_bytes": model.stat().st_size, "model_sha256": digest, "vendor": str(vendor), "version": receipt["version"], "existing_environments_changed": False, "poker_data_sent": False}
(OUT / "prepared.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
