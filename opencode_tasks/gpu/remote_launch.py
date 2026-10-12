import json
import subprocess
from pathlib import Path

r = Path("/workspace/poker/codex/tabpfn3_production_20260913")
name = "opencode_tabpfn_production"
config = Path("/etc/supervisor/conf.d/opencode-tabpfn-production.conf")
assert not config.exists(), "supervisor job already exists: inspect before retry"
python = "/venv/main/bin/python"
command = (
    "/usr/bin/timeout --signal=TERM --kill-after=15s 10800 "
    + python + " -u " + str(r / "tabpfn3_production_vm.py") + " all --root " + str(r)
)
config.write_text(
    "[program:" + name + "]\n"
    + "command=" + command + "\n"
    + "directory=" + str(r) + "\n"
    + "autostart=false\n"
    + "autorestart=false\n"
    + "startsecs=0\n"
    + "stdout_logfile=" + str(r / "production.log") + "\n"
    + "redirect_stderr=true\n"
    + "stdout_logfile_maxbytes=0\n"
)
receipt = {"supervisor_job": name, "command": command, "config": str(config)}
for c in [["supervisorctl", "reread"], ["supervisorctl", "update"], ["supervisorctl", "start", name]]:
    q = subprocess.run(c, capture_output=True, text=True)
    assert q.returncode == 0, (c, q.stdout, q.stderr)
    receipt[" ".join(c)] = (q.stdout + q.stderr).strip()
q = subprocess.run(["supervisorctl", "status", name], capture_output=True, text=True)
receipt["status"] = q.stdout.strip()
print(json.dumps(receipt, indent=2))
