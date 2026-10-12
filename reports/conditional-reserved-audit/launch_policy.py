from pathlib import Path
import hashlib
import subprocess
import tarfile

root=Path('/workspace/poker-research/conditional-audit169-v1')
assert Path(__file__).resolve().parent==root
archive=root/'policy-bundle.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='45c496041b2a2ccb4f484cba899825fef803cb8b9cc666711fcb61f960f2d098'
with tarfile.open(archive) as t: t.extractall(root,filter='data')
name='poker-audit169-policy'; config=Path('/etc/supervisor/conf.d')/(name+'.conf')
assert not config.exists()
config.write_text(f'''[program:{name}]
command=/venv/main/bin/python -u {root}/run_conditional_policy_screen.py
directory={root}
autostart=false
autorestart=false
startsecs=0
redirect_stderr=true
stdout_logfile={root}/run.log
stdout_logfile_maxbytes=0
stopasgroup=true
killasgroup=true
''')
for args in [['supervisorctl','reread'],['supervisorctl','update',name],['supervisorctl','start',name]]: subprocess.run(args,check=True)
