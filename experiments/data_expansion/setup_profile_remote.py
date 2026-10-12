from pathlib import Path
import hashlib,json,subprocess,tarfile
R=Path('/workspace/data-expansion-20260910');A=R/'profile-repair-package.tar.gz'
assert hashlib.sha256(A.read_bytes()).hexdigest()=='49a7a3e75717f1fec5187aa3a5bffca08a401594490efe5ef52890d38ec337d4'
H=R/'profile_repair';H.mkdir(exist_ok=False)
with tarfile.open(A) as t:t.extractall(H,filter='data')
p=json.loads((H/'pilot-plan.json').read_text());assert all(hashlib.sha256((H/n).read_bytes()).hexdigest()==v for n,v in p['source_hashes'].items())
subprocess.run(['/venv/main/bin/python',str(H/'test_generator.py')],check=True)
s=H/'worker.sh';s.write_text('''#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh ""
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /workspace/data-expansion-20260910/profile_repair
pty /venv/main/bin/python -u run_pilot.py 2>&1 | tee generation.log
''');s.chmod(0o755)
c=Path('/etc/supervisor/conf.d/poker_profile_pilot.conf');assert not c.exists()
c.write_text('''[program:poker_profile_pilot]
environment=PROC_NAME="poker_profile_pilot"
command=/workspace/data-expansion-20260910/profile_repair/worker.sh
directory=/workspace/data-expansion-20260910/profile_repair
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
''')
for cmd in [['reread'],['update','poker_profile_pilot'],['start','poker_profile_pilot']]:subprocess.run(['supervisorctl',*cmd],check=True)
