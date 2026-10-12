import hashlib
from pathlib import Path
import subprocess
import tarfile

root = Path('/workspace/mhc-20260910')
archive = root/'remote-package.tar.gz'
proxy = root/'remote-package-proxy.tar.gz'
if proxy.exists():
    assert hashlib.sha256(proxy.read_bytes()).hexdigest() == '84555a4d3427d82c994f48635720804585df1902a2978a3a9fa05fb5ab4f1e17'
    if archive.exists(): archive.rename(root/'partial-direct-transfer.tar.gz')
    proxy.rename(archive)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '84555a4d3427d82c994f48635720804585df1902a2978a3a9fa05fb5ab4f1e17'
with tarfile.open(archive) as tar:
    tar.extractall(root,filter='data')
script = root/'remote_worker.sh'
script.write_bytes(script.read_bytes().replace(b'\r\n',b'\n'))
script.chmod(0o755)
config = Path('/etc/supervisor/conf.d/poker_mhc.conf')
assert not config.exists()
config.write_text('''[program:poker_mhc]
environment=PROC_NAME="poker_mhc"
command=/workspace/mhc-20260910/remote_worker.sh
directory=/workspace/mhc-20260910
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
''')
subprocess.run(['supervisorctl','reread'],check=True)
subprocess.run(['supervisorctl','update','poker_mhc'],check=True)
subprocess.run(['supervisorctl','start','poker_mhc'],check=True)
