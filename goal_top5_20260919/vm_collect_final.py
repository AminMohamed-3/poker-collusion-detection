"""Archive final non-secret receipts and verify the GPU is no longer in use."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/workspace/poker_top5_20260919')
gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                      '--format=csv,noheader'], capture_output=True, text=True, check=True)
assert not gpu.stdout.strip(), 'A GPU process is still active; do not destroy yet'
jobs = ('standard', 'tabm', 'standard_citation', 'global_pu')
receipts = {}
for name in jobs:
    p = ROOT/name/'transfer_receipt.json'
    r = json.loads(p.read_text())
    archive = Path(r['archive'])
    assert archive.resolve().is_relative_to(ROOT.resolve())
    with archive.open('rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == r['sha256']
    assert r['inputs_exact']
    receipts[name] = r
(ROOT/'all_jobs_receipt.json').write_text(json.dumps({'gpu_idle': True, 'jobs': receipts}, indent=2))
archive = ROOT/'final_receipts.tar.gz'
with tarfile.open(archive, 'w:gz') as target:
    target.add(ROOT/'all_jobs_receipt.json', arcname='all_jobs_receipt.json')
    for name in jobs:
        target.add(ROOT/name/'transfer_receipt.json', arcname=name+'/transfer_receipt.json')
    for name in ('install.log','checkpoint.log','versions.log'):
        if (ROOT/'standard'/name).exists():
            target.add(ROOT/'standard'/name, arcname='standard/'+name)
with archive.open('rb') as source:
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': hashlib.file_digest(source, 'sha256').hexdigest(), 'gpu_idle': True}), flush=True)
