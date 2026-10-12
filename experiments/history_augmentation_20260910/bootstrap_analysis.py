from pathlib import Path
import hashlib,json,subprocess,tarfile
H=Path('/workspace/history-augmentation-20260910')
with tarfile.open('/workspace/history-analysis-package.tar.gz') as t:t.extractall(H,filter='data')
m=json.loads((H/'analysis-manifest.json').read_text())
assert all(hashlib.sha256((H/n).read_bytes()).hexdigest()==v for n,v in m.items())
E=H/'work/poker/experiments/history_augmentation_20260910'
(E/'varied').symlink_to(H/'varied',target_is_directory=True)
runner=E/'run_analysis.py'
runner.write_text('''from pathlib import Path
import subprocess,sys,time,json,tarfile,hashlib
H=Path(__file__).resolve().parent
D=H/'varied'
start=time.monotonic()
while not (D/'pilot-generation.json').exists():
    assert time.monotonic()-start<1800,'generation timeout'
    time.sleep(5)
for name,args in [('audit_varied_pilot.py',[]),('replay_varied_pilot.py',[]),('audit_seat_schedule.py',['varied']),('prepare_varied_features.py',[]),('varied_pair_domain_audit.py',[]),('temporal_audit.py',['varied'])]:
    print('START',name,flush=True)
    with (H/(name+'.log')).open('w') as f:subprocess.run([sys.executable,'-u',str(H/name),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',name,flush=True)
files=list(D.glob('*.json'))+[D/'features/synthetic-pairs.npz',D/'features/metadata.json',D/'temporal-features.parquet']
files += list(D.glob('pilot/pool-*/provenance.json'))+list(D.glob('pilot/pool-*/receipt.json'))
manifest={p.relative_to(D).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(D/'returned-manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open('/workspace/history-augmentation-20260910/varied-results.tar.gz','w:gz') as t:
    for p in files+[D/'returned-manifest.json']:t.add(p,arcname=p.relative_to(D))
print('AUDITS_AND_FEATURES_COMPLETE',flush=True)
''')
wrapper=Path('/opt/supervisor-scripts/poker_history_analysis.sh')
wrapper.write_text('''#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh ""
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /workspace/history-augmentation-20260910/work/poker/experiments/history_augmentation_20260910
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2
pty /venv/main/bin/python -u run_analysis.py 2>&1 | tee worker.log
''')
wrapper.chmod(0o755)
Path('/etc/supervisor/conf.d/poker_history_analysis.conf').write_text('''[program:poker_history_analysis]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/poker_history_analysis.sh
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
''')
for args in [['reread'],['update'],['start','poker_history_analysis']]:subprocess.run(['supervisorctl',*args],check=True)
