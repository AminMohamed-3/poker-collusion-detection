import hashlib
import json
import urllib.request
from pathlib import Path

out = Path(__file__).resolve().parent / "tabpfn35"
with urllib.request.urlopen("https://pypi.org/pypi/skrub/0.10.1/json", timeout=60) as response:
    meta = json.load(response)
wheel, = [x for x in meta["urls"] if x["filename"].endswith(".whl")]
with urllib.request.urlopen(wheel["url"], timeout=60) as response:
    payload = response.read()
assert hashlib.sha256(payload).hexdigest() == wheel["digests"]["sha256"]
(out / wheel["filename"]).write_bytes(payload)
receipt = {"wheel": wheel["filename"], "sha256": wheel["digests"]["sha256"], "requires_dist": meta["info"]["requires_dist"]}
(out / "skrub_source.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
