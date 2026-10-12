"""Reconstruct a verified subset package, then start its private worker."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
p=argparse.ArgumentParser(); p.add_argument('--job',required=True); p.add_argument('--sha256',required=True); args=p.parse_args()
assert args.job.startswith('omit_') and args.job.replace('_','').isalnum()
root=Path('/workspace/poker-research'); archive=root/('mil-'+args.job+'-delta.tar.gz'); assert hashlib.sha256(archive.read_bytes()).hexdigest()==args.sha256
dest=root/('mil-'+args.job+'-v1')
if dest.exists():
    assert not (dest/'results').exists() and not (Path('/etc/supervisor/conf.d')/('poker-mil-'+args.job+'.conf')).exists()
else: dest.mkdir()
with tarfile.open(archive) as t: t.extractall(dest,filter='data')
subprocess.run(['/venv/main/bin/python',str(dest/'rebuild_mil_delta.py')],check=True)
name='poker-mil-'+args.job; config=Path('/etc/supervisor/conf.d')/(name+'.conf'); assert not config.exists()
config.write_text(f'''[program:{name}]
command=/venv/main/bin/python -u {dest}/run_mil_pairs.py
directory={dest}
autostart=false
autorestart=false
startsecs=0
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={dest}/run.log
stdout_logfile_maxbytes=0
environment=PYTHONUNBUFFERED="1"
''')
subprocess.run(['supervisorctl','reread'],check=True); subprocess.run(['supervisorctl','update',name],check=True); subprocess.run(['supervisorctl','start',name],check=True)
