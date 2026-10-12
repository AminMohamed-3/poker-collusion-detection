"""One frozen Standard citation screen; use the existing verified checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import zipfile

ROOT = Path('/workspace/poker_top5_20260919/standard_citation')
ZIP = Path('/workspace/poker_standard_citation_20260919.zip')
EXPECTED = '4a9f11b35672cd48f1da3aa07ea435614c68f8ef66f6b84119fc0f6e656b5c2c'
STANDARD = ROOT.parent/'standard'

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

assert sha(ZIP) == EXPECTED
assert not ROOT.exists(), 'No retries or overwrite'
ROOT.mkdir()
with zipfile.ZipFile(ZIP) as archive:
    for member in archive.infolist():
        assert (ROOT/member.filename).resolve().is_relative_to(ROOT.resolve())
    archive.extractall(ROOT)
manifest = json.loads((ROOT/'MANIFEST.json').read_text())
for entry in manifest['files']:
    assert sha(ROOT/entry['path']) == entry['sha256']
model = 'tabpfn-v3.5-20260909_multiclass.safetensors'
assert sha(STANDARD/model) == '4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149'
os.symlink(STANDARD/model, ROOT/model)
env = os.environ.copy()
env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
results = []
for mode in ('run', 'replay'):
    t = time.monotonic()
    with (ROOT/(mode+'.log')).open('w') as stream:
        process = subprocess.Popen([str(STANDARD/'.venv/bin/python'), '-B', 'predict_standard.py', mode],
                                   cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        timed_out = False
        try:
            code = process.wait(timeout=1000)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            code = process.wait()
    result = {'mode': mode, 'pid': process.pid, 'exit_code': code, 'timed_out': timed_out,
              'seconds': time.monotonic()-t}
    results.append(result)
    print(json.dumps({**result, 'tail': (ROOT/(mode+'.log')).read_text(errors='replace').splitlines()[-5:]}), flush=True)
    if code != 0:
        break
(ROOT/'terminal.json').write_text(json.dumps(results, indent=2))
for entry in manifest['files']:
    assert sha(ROOT/entry['path']) == entry['sha256']
archive = ROOT/'citation_outputs.tar.gz'
with tarfile.open(archive, 'w:gz') as target:
    for name in ('outputs','terminal.json','run.log','replay.log'):
        if (ROOT/name).exists():
            target.add(ROOT/name, arcname=name)
receipt = {'archive': str(archive), 'bytes': archive.stat().st_size,
           'sha256': sha(archive), 'inputs_exact': True, 'terminal': results}
(ROOT/'transfer_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
