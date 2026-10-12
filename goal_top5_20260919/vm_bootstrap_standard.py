"""Prepare and run the fixed standard-model study on the user-provided VM."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tarfile
import time
import zipfile

ZIP = Path('/workspace/poker_standard_20260919.zip')
EXPECTED_ZIP = '1022464e5d259d60cfbcf52bd4dd332a5564240216246b44398b0ee73671d869'
ROOT = Path('/workspace/poker_top5_20260919/standard')
with ZIP.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == EXPECTED_ZIP
ROOT.mkdir(parents=True, exist_ok=True)
assert not (ROOT / 'outputs/predictions_fold0.npz').exists(), 'Preserve existing model predictions'
with zipfile.ZipFile(ZIP) as package:
    for member in package.infolist():
        assert (ROOT / member.filename).resolve().is_relative_to(ROOT.resolve())
    package.extractall(ROOT)
manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
for entry in manifest['files']:
    assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256']
environment = ROOT / '.venv'
subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(environment)], check=True)
# Reuse the matching preinstalled Torch without modifying the source environment.
libraries = environment / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
(libraries / 'poker_base_runtime.pth').write_text('\n'.join(site.getsitepackages()) + '\n')
python = str(environment / 'bin/python')
settings = os.environ.copy()
settings.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', TABPFN_NO_BROWSER='1',
                MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'), SKB_DATA_DIRECTORY=str(ROOT / 'cache/skrub'))

def execute(name, command):
    print(json.dumps({'phase': name, 'status': 'starting'}), flush=True)
    started = time.monotonic()
    log = ROOT / (name + '.log')
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=settings, stdout=stream, stderr=subprocess.STDOUT)
    lines = log.read_text(errors='replace').splitlines()
    print(json.dumps({'phase': name, 'exit_code': result.returncode,
                      'seconds': round(time.monotonic()-started, 1), 'tail': lines[-6:]}), flush=True)
    assert result.returncode == 0, f'{name} failed; inspect {log}'

execute('install', [python, '-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements.txt'])
execute('versions', [python, '-m', 'pip', 'freeze'])
execute('checkpoint', [python, 'download_checkpoint.py'])
execute('preflight', [python, 'preflight.py'])
execute('known', [python, 'predict_standard.py', 'known'])
execute('replay', [python, 'predict_standard.py', 'replay'])
for entry in manifest['files']:
    assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256']
archive = ROOT / 'standard_known_outputs.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(ROOT / 'outputs', arcname='outputs')
    for name in ['versions.log', 'preflight.log', 'known.log', 'replay.log']:
        tar.add(ROOT / name, arcname=name)
with archive.open('rb') as f:
    receipt = {'archive': str(archive), 'bytes': archive.stat().st_size,
               'sha256': hashlib.file_digest(f, 'sha256').hexdigest(), 'inputs_exact': True}
(ROOT / 'transfer_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
