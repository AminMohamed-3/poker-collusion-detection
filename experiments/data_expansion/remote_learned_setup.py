from pathlib import Path
import hashlib,json,subprocess,tarfile
r=Path('/workspace/data-expansion-20260910');a=r/'learned-package.tar.gz'
assert hashlib.sha256(a.read_bytes()).hexdigest()=='f3867ce4917e8cf9a5f90610215383730bb316eb8eaa21e47c13909715a22367'
d=r/'learned';d.mkdir(exist_ok=False)
with tarfile.open(a) as t:t.extractall(d,filter='data')
p=json.loads((d/'pilot-plan.json').read_text());assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in p['source_hashes'].items())
subprocess.run(['/venv/main/bin/python',str(d/'test_generator.py')],check=True)
s=d/'worker.sh';s.write_text('''#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh ""
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /workspace/data-expansion-20260910/learned
pty /venv/main/bin/python -u run_pilot.py 2>&1 | tee generation.log
''');s.chmod(0o755)
c=Path('/etc/supervisor/conf.d/poker_learned_pilot.conf');assert not c.exists()
c.write_text('''[program:poker_learned_pilot]
environment=PROC_NAME="poker_learned_pilot"
command=/workspace/data-expansion-20260910/learned/worker.sh
directory=/workspace/data-expansion-20260910/learned
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
''')
for cmd in [['reread'],['update','poker_learned_pilot'],['start','poker_learned_pilot']]:subprocess.run(['supervisorctl',*cmd],check=True)
