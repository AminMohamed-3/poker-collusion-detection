from pathlib import Path
import hashlib,json,subprocess,tarfile
H=Path('/workspace/mhc-20260910')
with tarfile.open(H/'omitted-package.tar.gz') as t:t.extractall(H,filter='data')
m=json.loads((H/'omitted-manifest.json').read_text());assert all(hashlib.sha256((H/n).read_bytes()).hexdigest()==v for n,v in m.items())
s=H/'omitted_worker.sh';s.write_text('''#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh ""
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /workspace/mhc-20260910
pty /venv/main/bin/python -u run_omitted_remote.py 2>&1 | tee omitted-worker.log
''');s.chmod(0o755)
c=Path('/etc/supervisor/conf.d/poker_mhc_omitted.conf');assert not c.exists()
c.write_text('''[program:poker_mhc_omitted]
environment=PROC_NAME="poker_mhc_omitted"
command=/workspace/mhc-20260910/omitted_worker.sh
directory=/workspace/mhc-20260910
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
''')
for cmd in [['reread'],['update','poker_mhc_omitted'],['start','poker_mhc_omitted']]:subprocess.run(['supervisorctl',*cmd],check=True)
