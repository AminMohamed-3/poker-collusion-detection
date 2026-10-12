"""Verified one-shot remote worker. No held-label access or evaluation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE/'remote-package-manifest.json').read_text())
assert all(hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
assert not (HERE/'fold0/held-metadata.json').exists()
capabilities = json.loads(subprocess.check_output(['vast-capabilities'],text=True))
instance = capabilities.get('instance',{})
receipt = {'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
           'cuda':torch.version.cuda,'python':sys.version,
           'cpu_affinity_count':len(os.sched_getaffinity(0)),
           'container_id':os.environ.get('CONTAINER_ID'),
           'workspace_is_volume':instance.get('workspace_is_volume'),
           'held_labels_absent':True,'execution_concurrency':3,
           'queue_sha256':hashlib.sha256((HERE/'run_remote_parallel.py').read_bytes()).hexdigest()}
(HERE/'remote-runtime.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
scripts=[]
if not (HERE/'tests.json').exists() or not json.loads((HERE/'tests.json').read_text())['passed']: scripts.append('test_model.py')
if not (HERE/'runtime-check.json').exists() or not json.loads((HERE/'runtime-check.json').read_text())['passed']: scripts.append('runtime_check.py')
for script in scripts+['run_remote_parallel.py']:
    subprocess.run([sys.executable,'-u',str(HERE/script)],check=True)
assert len(json.loads((HERE/'fold0/queue.json').read_text())['completed'])==12
print('All twelve frozen fits completed. Held evaluation must run locally.',flush=True)
