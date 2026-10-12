"""Resume the partial official checkpoint, verify it, then run the frozen study."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request

ROOT = Path('/workspace/poker_top5_20260919/standard')
MODEL = ROOT / 'tabpfn-v3.5-20260909_multiclass.safetensors'
TOTAL = 876028348
SHA = '4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149'
URL = 'https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/' + MODEL.name
for attempt in range(1, 7):
    offset = MODEL.stat().st_size if MODEL.exists() else 0
    assert offset <= TOTAL
    if offset == TOTAL:
        break
    print(json.dumps({'phase': 'resume_checkpoint', 'attempt': attempt, 'offset': offset, 'total': TOTAL}), flush=True)
    request = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-', 'User-Agent': 'poker-research/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            if offset:
                assert response.status == 206, 'Server ignored resume range; existing partial preserved'
                assert response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
            last_report = offset
            with MODEL.open('ab' if offset else 'wb') as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    offset += len(chunk)
                    if offset-last_report >= 64*1024*1024:
                        target.flush()
                        print(json.dumps({'phase': 'checkpoint_progress', 'bytes': offset, 'total': TOTAL}), flush=True)
                        last_report = offset
    except (TimeoutError, OSError) as exc:
        print(json.dumps({'attempt': attempt, 'error_type': type(exc).__name__, 'bytes_saved': MODEL.stat().st_size}), flush=True)
assert MODEL.stat().st_size == TOTAL
with MODEL.open('rb') as source:
    assert hashlib.file_digest(source, 'sha256').hexdigest() == SHA
print(json.dumps({'phase': 'checkpoint', 'verified': True, 'sha256': SHA}), flush=True)
python = str(ROOT / '.venv/bin/python')
env = os.environ.copy()
env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
           HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
           TABPFN_NO_BROWSER='1', MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
           SKB_DATA_DIRECTORY=str(ROOT/'cache/skrub'))
for name, arguments in [('preflight', ['preflight.py']), ('known', ['predict_standard.py', 'known']), ('replay', ['predict_standard.py', 'replay'])]:
    print(json.dumps({'phase': name, 'status': 'starting'}), flush=True)
    started = time.monotonic()
    log = ROOT / (name + '.log')
    with log.open('w') as output:
        status = subprocess.run([python, *arguments], cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    print(json.dumps({'phase': name, 'exit_code': status.returncode, 'seconds': time.monotonic()-started,
                      'tail': log.read_text(errors='replace').splitlines()[-6:]}), flush=True)
    assert status.returncode == 0
manifest = json.loads((ROOT/'MANIFEST.json').read_text())
for entry in manifest['files']:
    assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest() == entry['sha256']
archive = ROOT/'standard_known_outputs.tar.gz'
with tarfile.open(archive, 'w:gz') as out:
    out.add(ROOT/'outputs', arcname='outputs')
    for name in ['versions.log','preflight.log','known.log','replay.log']:
        out.add(ROOT/name, arcname=name)
with archive.open('rb') as source:
    receipt = {'archive': str(archive), 'bytes': archive.stat().st_size,
               'sha256': hashlib.file_digest(source, 'sha256').hexdigest(), 'inputs_exact': True}
(ROOT/'transfer_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
