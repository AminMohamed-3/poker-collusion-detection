from pathlib import Path
import hashlib, json, subprocess, tarfile
H=Path('/workspace/history-augmentation-20260910')
H.mkdir(exist_ok=False)
with tarfile.open('/workspace/history-augmentation-package.tar.gz') as t:
    t.extractall(H,filter='data')
m=json.loads((H/'generation-manifest.json').read_text())
assert all(hashlib.sha256((H/n).read_bytes()).hexdigest()==v for n,v in m.items())
(H/'vendor').symlink_to('/workspace/data-expansion-20260910/vendor',target_is_directory=True)
subprocess.run(['/venv/main/bin/python','-c',"import torch,polars; print(torch.__version__,torch.cuda.is_available(),polars.__version__); print(torch.ones(1,device='cuda').item())"],check=True)
wrapper=Path('/opt/supervisor-scripts/poker_history_variation.sh')
wrapper.write_text('''#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh ""
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /workspace/history-augmentation-20260910/varied
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1
pty /venv/main/bin/python -u run_pilot.py 2>&1 | tee worker.log
''')
wrapper.chmod(0o755)
Path('/etc/supervisor/conf.d/poker_history_variation.conf').write_text('''[program:poker_history_variation]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/poker_history_variation.sh
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
''')
for args in [['reread'],['update'],['start','poker_history_variation']]:
    subprocess.run(['supervisorctl',*args],check=True)
