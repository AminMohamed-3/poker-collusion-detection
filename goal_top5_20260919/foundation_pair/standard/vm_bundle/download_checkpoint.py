"""Download only the official Prior Labs checkpoint and verify its LFS SHA256."""
import hashlib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = HERE / "tabpfn-v3.5-20260909_multiclass.safetensors"
URL = "https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/tabpfn-v3.5-20260909_multiclass.safetensors"
EXPECTED = "4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149"
assert not DESTINATION.exists(), f"Refusing to overwrite {DESTINATION}"
urllib.request.urlretrieve(URL, DESTINATION)
with DESTINATION.open("rb") as stream:
    actual = hashlib.file_digest(stream, "sha256").hexdigest()
assert actual == EXPECTED, (actual, EXPECTED)
print({"path": str(DESTINATION), "sha256": actual, "bytes": DESTINATION.stat().st_size})
