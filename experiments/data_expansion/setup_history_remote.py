from pathlib import Path
import hashlib,json,subprocess,tarfile
R=Path('/workspace/data-expansion-20260910');A=R/'history-repair-package.tar.gz'
assert hashlib.sha256(A.read_bytes()).hexdigest()=='b33971221584ba71328540a368e0b131bfccf1d107209602c3c955654264a537'
H=R/'history_repair';H.mkdir(exist_ok=False)
with tarfile.open(A) as t:t.extractall(H,filter='data')
p=json.loads((H/'pilot-plan.json').read_text());assert all(hashlib.sha256((H/n).read_bytes()).hexdigest()==v for n,v in p['source_hashes'].items())
subprocess.run(['/venv/main/bin/python',str(H/'test_generator.py')],check=True)
s=H/'worker.sh';s.write_text('''#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh ""
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /workspace/data-expansion-20260910/history_repair
pty /venv/main/bin/python -u run_pilot.py 2>&1 | tee generation.log
''');s.chmod(0o755)
c=Path('/etc/supervisor/conf.d/poker_history_pilot.conf');assert not c.exists()
c.write_text('''[program:poker_history_pilot]
environment=PROC_NAME="poker_history_pilot"
command=/workspace/data-expansion-20260910/history_repair/worker.sh
directory=/workspace/data-expansion-20260910/history_repair
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
''')
for cmd in [['reread'],['update','poker_history_pilot'],['start','poker_history_pilot']]:subprocess.run(['supervisorctl',*cmd],check=True)
