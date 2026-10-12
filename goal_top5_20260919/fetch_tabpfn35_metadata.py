"""Download official public metadata/source wheel; never sends competition data."""
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tabpfn35"
OUT.mkdir(exist_ok=True)
def read(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()

meta = json.loads(read("https://pypi.org/pypi/tabpfn/json"))
version = meta["info"]["version"]
wheels = [x for x in meta["urls"] if x["filename"].endswith(".whl")]
assert len(wheels) == 1, [x["filename"] for x in wheels]
wheel = wheels[0]
body = read(wheel["url"])
assert hashlib.sha256(body).hexdigest() == wheel["digests"]["sha256"]
(OUT / wheel["filename"]).write_bytes(body)
license_url = "https://huggingface.co/Prior-Labs/tabpfn_3_5/raw/main/LICENSE"
license_body = read(license_url)
(OUT / "MODEL_LICENSE.txt").write_bytes(license_body)
result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "version": version,
          "requires_python": meta["info"]["requires_python"], "requires_dist": meta["info"]["requires_dist"],
          "wheel": {k: wheel[k] for k in ("filename", "url", "size", "digests")},
          "license_url": license_url, "license_sha256": hashlib.sha256(license_body).hexdigest(),
          "model_url": "https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/tabpfn-v3.5-fast-20260909.safetensors",
          "expected_model_sha256": "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550",
          "model_not_downloaded": True}
(OUT / "source_receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
