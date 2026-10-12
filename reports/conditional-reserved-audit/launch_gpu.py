from pathlib import Path
import hashlib
import subprocess
import tarfile

root=Path('/workspace/poker-research/conditional-audit169-v1'); assert Path(__file__).resolve().parent==root
archive=root/'gpu-bundle.tar.gz'; assert hashlib.sha256(archive.read_bytes()).hexdigest()=='1df7454086c753685cd3ae79ee833b37a199b6ddcdee4b1c9735908874bd2101'
gpu=root/'gpu'; gpu.mkdir()
with tarfile.open(archive) as t: t.extractall(gpu,filter='data')
name='poker-audit169-tabicl'; config=Path('/etc/supervisor/conf.d')/(name+'.conf'); assert not config.exists()
config.write_text(f'''[program:{name}]
command=/venv/main/bin/python -u {gpu}/run_reserved_tabicl.py
directory={gpu}
autostart=false
autorestart=false
startsecs=0
redirect_stderr=true
stdout_logfile={gpu}/run.log
stdout_logfile_maxbytes=0
stopasgroup=true
killasgroup=true
''')
for args in [['supervisorctl','reread'],['supervisorctl','update',name],['supervisorctl','start',name]]: subprocess.run(args,check=True)
assert (root/'policy-screen/summary.json').is_file()
result=root/'policy-results.tar.gz'
with tarfile.open(result,'w:gz') as t:
    t.add(root/'policy-screen',arcname='policy-screen'); t.add(root/'run.log',arcname='policy-run.log')
print('policy-results-sha256='+hashlib.sha256(result.read_bytes()).hexdigest(),flush=True)
