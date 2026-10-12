"""Run the reviewed immutable TabM package once, preserving all terminal states."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import zipfile

ROOT = Path('/workspace/poker_top5_20260919/tabm')
PACKAGE = Path('/workspace/poker_tabm_reviewed_20260919.zip')
EXPECTED = 'a55fdd1b46014f7d8a037b373cdb3e3e607edd2796b8209d97dd53b6fedb3398'
PYTHON = '/workspace/poker_top5_20260919/standard/.venv/bin/python'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert sha(PACKAGE) == EXPECTED
ROOT.mkdir(parents=True, exist_ok=True)
assert not (ROOT/'run').exists(), 'No repeat or rescue of pilot'
with zipfile.ZipFile(PACKAGE) as archive:
    for member in archive.infolist():
        assert (ROOT/member.filename).resolve().is_relative_to(ROOT.resolve())
    archive.extractall(ROOT)
env = os.environ.copy()
env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
with (ROOT/'bundle_check.log').open('w') as stream:
    result = subprocess.run([PYTHON, '-B', 'check_bundle.py'], cwd=ROOT, env=env,
                            stdout=stream, stderr=subprocess.STDOUT)
print(json.dumps({'phase': 'bundle_check', 'exit_code': result.returncode,
                  'log': (ROOT/'bundle_check.log').read_text()}), flush=True)
assert result.returncode == 0, 'No training attempted'
started = time.time()
with (ROOT/'terminal.log').open('w') as stream:
    process = subprocess.Popen([PYTHON, '-B', 'train.py', 'run'], cwd=ROOT, env=env,
                               stdout=stream, stderr=subprocess.STDOUT)
    identity = {'pid': process.pid, 'started_unix': started, 'bundle_sha256': EXPECTED}
    (ROOT/'process.json').write_text(json.dumps(identity, indent=2))
    print(json.dumps({'phase': 'training', **identity}), flush=True)
    timed_out = False
    try:
        code = process.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            code = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
terminal = {**identity, 'exit_code': code, 'timed_out': timed_out,
            'seconds': time.time()-started, 'success_receipt_exists': (ROOT/'run/receipt.json').exists()}
(ROOT/'terminal_status.json').write_text(json.dumps(terminal, indent=2))
print(json.dumps({'phase': 'terminal', **terminal,
                  'tail': (ROOT/'terminal.log').read_text(errors='replace').splitlines()[-8:]}), flush=True)
manifest = json.loads((ROOT/'MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert sha(ROOT/name) == expected
archive = ROOT/'tabm_outputs.tar.gz'
with tarfile.open(archive, 'w:gz') as output:
    if (ROOT/'run').exists():
        output.add(ROOT/'run', arcname='run')
    for name in ('bundle_check.log','terminal.log','terminal_status.json','process.json','MANIFEST.json'):
        output.add(ROOT/name, arcname=name)
receipt = {'archive': str(archive), 'bytes': archive.stat().st_size,
           'sha256': sha(archive), 'inputs_exact': True, 'terminal': terminal}
(ROOT/'transfer_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
