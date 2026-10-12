"""Run the immutable Global-PU screen and replay, preserving failed outputs too."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path('/workspace/poker_top5_20260919/global_pu')
PYTHON = ROOT.parent/'standard/.venv/bin/python'

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

setup = json.loads((ROOT/'setup_receipt.json').read_text())
assert setup['inputs_exact'] and setup['bundle_sha256'] == '5522436d8408e54e6e0aa6f42ccb2cf91b2e0d4d24c7c7330d58eada37504555'
manifest = json.loads((ROOT/'MANIFEST.json').read_text())
for entry in manifest['files']:
    assert sha(ROOT/entry['path']) == entry['sha256']
assert not (ROOT/'outputs').exists(), 'Only a fresh invocation is authorized here'
env = os.environ.copy()
env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
terminal = []
for mode, limit in [('run', 3900), ('replay', 600)]:
    started = time.monotonic()
    with (ROOT/(mode+'.log')).open('w') as stream:
        process = subprocess.Popen([str(PYTHON), '-B', 'predict_fast.py', mode], cwd=ROOT,
                                   env=env, stdout=stream, stderr=subprocess.STDOUT)
        print(json.dumps({'phase': mode, 'pid': process.pid, 'status': 'starting'}), flush=True)
        timeout = False
        try:
            code = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timeout = True
            process.kill()
            code = process.wait()
    state = {'mode': mode, 'pid': process.pid, 'exit_code': code, 'timed_out': timeout,
             'seconds': time.monotonic()-started}
    terminal.append(state)
    print(json.dumps({**state, 'tail': (ROOT/(mode+'.log')).read_text(errors='replace').splitlines()[-5:]}), flush=True)
    if code != 0:
        break
(ROOT/'terminal.json').write_text(json.dumps(terminal, indent=2))
for entry in manifest['files']:
    assert sha(ROOT/entry['path']) == entry['sha256']
archive = ROOT/'global_pu_outputs.tar.gz'
with tarfile.open(archive, 'w:gz') as target:
    for name in ('outputs','terminal.json','setup_receipt.json','run.log','replay.log'):
        if (ROOT/name).exists():
            target.add(ROOT/name, arcname=name)
receipt = {'archive': str(archive), 'bytes': archive.stat().st_size,
           'sha256': sha(archive), 'inputs_exact': True, 'terminal': terminal}
(ROOT/'transfer_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
