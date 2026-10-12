import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root=Path('/workspace/data-expansion-20260910')
archive=root/'pilot-package.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='58b2d7b94223a966dba4d356e6221f849d9c2975e7c37a30c1d96076c7985462'
with tarfile.open(archive) as tar: tar.extractall(root,filter='data')
manifest=json.loads((root/'manifest.json').read_text())
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
subprocess.run(['uv','pip','install','--python','/venv/main/bin/python','polars==1.44.1','phevaluator==0.6.0','lightgbm==4.7.0','scikit-learn==1.9.0','joblib==1.6.0','tzdata'],check=True)
subprocess.run(['/venv/main/bin/python','-m','pip','install','--no-index','--no-deps','--target',str(root/'vendor'),str(root/'pokerkit-0.7.5-py3-none-any.whl')],check=True)
subprocess.run(['/venv/main/bin/python','-u',str(root/'test_generator.py')],check=True)
script=root/'remote_pilot.sh';script.write_bytes(script.read_bytes().replace(b'\r\n',b'\n'));script.chmod(0o755)
config=Path('/etc/supervisor/conf.d/poker_data_pilot.conf');assert not config.exists()
config.write_text('''[program:poker_data_pilot]
environment=PROC_NAME="poker_data_pilot"
command=/workspace/data-expansion-20260910/remote_pilot.sh
directory=/workspace/data-expansion-20260910
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
''')
subprocess.run(['supervisorctl','reread'],check=True)
subprocess.run(['supervisorctl','update','poker_data_pilot'],check=True)
subprocess.run(['supervisorctl','start','poker_data_pilot'],check=True)
