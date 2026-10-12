"""Private supervisor worker; no changes to other running services."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
p=argparse.ArgumentParser(); p.add_argument('--job',required=True); p.add_argument('--sha256',required=True); args=p.parse_args()
assert args.job.replace('_','').isalnum()
root=Path('/workspace/poker-research'); archive=root/('tabpfn-'+args.job+'.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest()==args.sha256
dest=root/('tabpfn-'+args.job+'-v1'); dest.mkdir(exist_ok=False)
with tarfile.open(archive) as t: t.extractall(dest,filter='data')
name='poker-tabpfn-'+args.job; config=Path('/etc/supervisor/conf.d')/(name+'.conf'); assert not config.exists()
config.write_text(f'''[program:{name}]
command=/venv/main/bin/python -u {dest}/run_tabpfn_v2.py
directory={dest}
autostart=false
autorestart=false
startsecs=0
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={dest}/run.log
stdout_logfile_maxbytes=0
environment=PYTHONUNBUFFERED="1",TABPFN_DISABLE_TELEMETRY="1"
''')
subprocess.run(['supervisorctl','reread'],check=True)
subprocess.run(['supervisorctl','update',name],check=True)
# Explicit start is separate, after the previous GPU batch completes.
print(name+' prepared, not started')
