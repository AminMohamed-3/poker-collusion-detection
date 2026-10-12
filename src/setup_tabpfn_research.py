"""Install a private package overlay; preserve the main training environment."""
from pathlib import Path
import hashlib
import json
import subprocess
import urllib.request
import zipfile

root=Path('/workspace/poker-research/tabpfn-v2'); root.mkdir(exist_ok=False)
libs=root/'libs'
packages=['tabpfn==2.2.1','scikit-learn==1.6.1','huggingface-hub==0.36.2','pandas==2.3.3','backoff==2.2.1','distro==1.9.0','eval-type-backport==0.4.0','nvidia-ml-py==13.610.43','posthog==7.48.0','pydantic-settings==2.15.0','python-dotenv==1.2.3','pytz==2026.3.post1','tabpfn-common-utils==0.2.23','tzdata==2026.3']
subprocess.run(['uv','pip','install','--python','/venv/main/bin/python','--target',str(libs),'--no-deps',*packages],check=True)
revision='f851f2a3c941544733b712d8c0f96dfae9b28862'
prefix=f'https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/{revision}/'
checkpoint=root/'tabpfn-v2-classifier-v2_default.ckpt'
for name in ['tabpfn-v2-classifier-v2_default.ckpt','LICENSE.txt','README.md']:
    urllib.request.urlretrieve(prefix+name,root/name)
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert digest=='cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49'
(root/'installation.json').write_text(json.dumps({'checkpoint_revision':revision,'checkpoint_sha256':digest,'packages':packages,'isolated_overlay':True,'attribution':'Built with PriorLabs-TabPFN; research only. Distributed models must use the TabPFN prefix and retain this license.'},indent=2)+'\n')
print(json.dumps({'setup_complete':True,'checkpoint_sha256':digest}))
