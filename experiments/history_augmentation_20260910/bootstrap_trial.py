from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
BASE=Path('/workspace/history-augmentation-20260910');H=BASE/'trial'
H.mkdir(exist_ok=False)
with tarfile.open('/workspace/history-trial-package.tar.gz') as t:t.extractall(H,filter='data')
m=json.loads((H/'remote-manifest.json').read_text())
assert all(hashlib.sha256((H/n).read_bytes()).hexdigest()==v for n,v in m.items())
shutil.copy2('/workspace/stress_inference.py',H/'stress_inference.py')
(H/'collect_results.py').write_text('''from pathlib import Path
import json,hashlib,tarfile
H=Path(__file__).resolve().parent
files=[p for p in H.rglob('*') if p.is_file() and p.name not in ['package.tar.gz','returned-manifest.json'] and '__pycache__' not in p.parts]
# Original training inputs are already safely local; preserve all new models,
# predictions, logs, manifests and code in the return archive.
files=[p for p in files if p.name!='train-arrays.npz']
m={p.relative_to(H).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(H/'returned-manifest.json').write_text(json.dumps(m,indent=2))
with tarfile.open(H.parent/'trial-results.tar.gz','w:gz') as t:
    for p in files+[H/'returned-manifest.json']:t.add(p,arcname=p.relative_to(H))
print('RESULTS_ARCHIVED',flush=True)
''')
wrapper=Path('/opt/supervisor-scripts/poker_history_trial.sh')
wrapper.write_text('''#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh ""
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /workspace/history-augmentation-20260910/trial
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2
pty /venv/main/bin/python -u run_remote.py 2>&1 | tee worker.log
pty /venv/main/bin/python -u stress_inference.py 2>&1 | tee stress.log
pty /venv/main/bin/python -u collect_results.py 2>&1 | tee collection.log
''')
wrapper.chmod(0o755)
Path('/etc/supervisor/conf.d/poker_history_trial.conf').write_text('''[program:poker_history_trial]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/poker_history_trial.sh
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
''')
for args in [['reread'],['update'],['start','poker_history_trial']]:subprocess.run(['supervisorctl',*args],check=True)
