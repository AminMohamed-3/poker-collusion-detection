"""One-shot local setup, preflight, and frozen training queue."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
receipt = HERE / 'downloads/receipt.json'
started = time.monotonic()
while not receipt.exists():
    if time.monotonic() - started > 1800:
        raise TimeoutError('Official wheel download did not complete within 30 minutes')
    time.sleep(5)
r = json.loads(receipt.read_text())
assert r['verified'] and r['sha256'] == '7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c'
wheel = HERE / 'downloads/torch-2.11.0+cu128-cp312-cp312-win_amd64.whl'
digest = hashlib.sha256()
with wheel.open('rb') as f:
    while chunk := f.read(1024 * 1024):
        digest.update(chunk)
assert digest.hexdigest() == r['sha256']
commands = [
    [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)],
    [sys.executable, '-m', 'pip', 'check'],
    [sys.executable, '-u', str(HERE / 'test_model.py')],
    [sys.executable, '-u', str(HERE / 'runtime_check.py')],
    [sys.executable, '-u', str(HERE / 'run_local.py')],
]
for name in ['tests.json', 'tests.log']:
    previous = HERE / name
    archive = HERE / ('initial-cpu-' + name)
    if previous.exists() and not archive.exists():
        shutil.copy2(previous, archive)
for command in commands:
    print('Executing: ' + repr(command), flush=True)
    subprocess.run(command, check=True)
print('Local pipeline exited; see fold0/queue.json for completed fits.', flush=True)
