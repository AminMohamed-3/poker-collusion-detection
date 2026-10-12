from pathlib import Path
import hashlib,json,subprocess,tarfile
H=Path('/workspace/poker-augmentation-20260910')
with tarfile.open(H/'package.tar.gz') as t:t.extractall(H,filter='data')
m=json.loads((H/'remote-manifest.json').read_text());assert all(hashlib.sha256((H/n).read_bytes()).hexdigest()==v for n,v in m.items())
s=H/'worker.sh';s.write_text('''#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh ""
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /workspace/poker-augmentation-20260910
pty /venv/main/bin/python -u run_remote.py 2>&1 | tee worker.log
''');s.chmod(0o755)
c=Path('/etc/supervisor/conf.d/poker_augmentation.conf');assert not c.exists()
c.write_text('''[program:poker_augmentation]
environment=PROC_NAME="poker_augmentation"
command=/workspace/poker-augmentation-20260910/worker.sh
directory=/workspace/poker-augmentation-20260910
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
''')
for cmd in [['reread'],['update','poker_augmentation'],['start','poker_augmentation']]:subprocess.run(['supervisorctl',*cmd],check=True)
