import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
O = ROOT / "Codex Resume" / "outputs_tabpfn3_default_20260913"
inv = json.loads((O / "transfer-plan.json").read_text(encoding="utf-8"))
R = "/workspace/poker/codex/tabpfn3_production_20260913"
P = O / "vm-package"
lines = []
for n in inv["files"]:
    src = (P / n).as_posix()
    lines.append("put \"" + src + "\" \"" + R + "/" + n + "\"\n")
out = ROOT / ".opencode" / "continuous" / "tasks" / "gpu" / "sftp_batch.txt"
out.write_text("".join(lines), encoding="utf-8")
print("batch lines: " + str(len(lines)))
