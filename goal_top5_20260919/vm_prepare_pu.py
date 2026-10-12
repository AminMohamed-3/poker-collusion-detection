"""Verify and stage Global-PU inputs; download the official model without GPU use."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path('/workspace/poker_top5_20260919/global_pu')
PACKAGE = Path('/workspace/poker_global_pu_20260919.zip')
EXPECTED_PACKAGE = '5522436d8408e54e6e0aa6f42ccb2cf91b2e0d4d24c7c7330d58eada37504555'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert sha(PACKAGE) == EXPECTED_PACKAGE
ROOT.mkdir(parents=True, exist_ok=True)
assert not (ROOT/'outputs').exists(), 'Preserve any existing run'
with zipfile.ZipFile(PACKAGE) as archive:
    for member in archive.infolist():
        assert (ROOT/member.filename).resolve().is_relative_to(ROOT.resolve())
    archive.extractall(ROOT)
manifest = json.loads((ROOT/'MANIFEST.json').read_text())
for entry in manifest['files']:
    assert sha(ROOT/entry['path']) == entry['sha256']
model = ROOT / manifest['checkpoint']['filename']
total = 334181708
url = 'https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/' + model.name
for attempt in range(1, 7):
    offset = model.stat().st_size if model.exists() else 0
    assert offset <= total
    if offset == total:
        break
    print(json.dumps({'phase': 'checkpoint', 'attempt': attempt, 'offset': offset}), flush=True)
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            if offset:
                assert response.status == 206
                assert response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
            last = offset
            with model.open('ab' if offset else 'wb') as target:
                while chunk := response.read(1024*1024):
                    target.write(chunk)
                    offset += len(chunk)
                    if offset-last >= 64*1024*1024:
                        target.flush()
                        print(json.dumps({'phase': 'checkpoint', 'bytes': offset}), flush=True)
                        last = offset
    except (TimeoutError, OSError) as exc:
        print(json.dumps({'error_type': type(exc).__name__, 'bytes_saved': model.stat().st_size}), flush=True)
assert model.stat().st_size == total
assert sha(model) == manifest['checkpoint']['sha256']
receipt = {'bundle_sha256': EXPECTED_PACKAGE, 'checkpoint_sha256': sha(model),
           'checkpoint_bytes': total, 'inputs_exact': True, 'gpu_used': False}
(ROOT/'setup_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
