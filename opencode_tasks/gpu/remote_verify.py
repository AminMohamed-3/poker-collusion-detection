import hashlib
import json
from pathlib import Path

r = Path("/workspace/poker/codex/tabpfn3_production_20260913")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


f = json.loads((r / "production-freeze.json").read_text())
bad = [n for n, d in f["files"].items() if sha(r / n) != d]
print("freeze_files: " + str(len(f["files"])) + " mismatches: " + str(bad))
print("freeze_sha: " + sha(r / "production-freeze.json"))
print("expect_freeze: c4f9527f1dbe527f447c267c2f711931f13de73ac6407730515ef544ef0589b3")
print("ckpt: " + sha(r / "checkpoint/tabpfn-v3-classifier-v3_default.ckpt"))
print("expect_ckpt: " + f["checkpoint_sha256"])
import tabpfn

print("tabpfn: " + tabpfn.__version__ + " at " + tabpfn.__file__)
import torch

print("torch: " + torch.__version__ + " cuda: " + str(torch.cuda.is_available()))
