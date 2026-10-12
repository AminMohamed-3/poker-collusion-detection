"""Resumable, bounded download of the official CUDA wheel; verify its pinned hash."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import subprocess

ROOT = Path(__file__).resolve().parent / 'downloads'
URL = 'https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl'
SIZE = 2753189216
SHA = '7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c'
CHUNK = 16 * 1024 * 1024

def fetch(i):
    start, end = i * CHUNK, min(SIZE, (i + 1) * CHUNK) - 1
    part = ROOT / f'part-{i:03}.bin'
    if part.exists() and part.stat().st_size == end - start + 1:
        return i
    for attempt in range(4):
        try:
            header = part.with_suffix('.headers')
            proc = subprocess.run(['curl.exe', '--silent', '--show-error', '--fail', '-L',
                '--range', f'{start}-{end}', '--max-time', '90', '-D', str(header),
                '--output', str(part.with_suffix('.tmp')), URL], capture_output=True, text=True)
            assert proc.returncode == 0, proc.stderr
            assert f'content-range: bytes {start}-{end}/{SIZE}' in header.read_text().lower()
            assert part.with_suffix('.tmp').stat().st_size == end - start + 1
            part.with_suffix('.tmp').replace(part)
            return i
        except Exception as exc:
            print(json.dumps({'part': i, 'attempt': attempt, 'error': str(exc)}), flush=True)
            if attempt == 3:
                raise
            time.sleep(1 + attempt)

def main():
    ROOT.mkdir(exist_ok=True)
    began = time.monotonic()
    count = (SIZE + CHUNK - 1) // CHUNK
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for done, result in enumerate(concurrent.futures.as_completed([pool.submit(fetch, i) for i in range(count)]), 1):
            result.result()
            print(json.dumps({'chunks': done, 'total': count, 'seconds': round(time.monotonic() - began, 1)}), flush=True)
    target = ROOT / 'torch-2.11.0+cu128-cp312-cp312-win_amd64.whl'
    digest = hashlib.sha256()
    with target.open('wb') as out:
        for i in range(count):
            with (ROOT / f'part-{i:03}.bin').open('rb') as part:
                while data := part.read(1024 * 1024):
                    out.write(data)
                    digest.update(data)
    assert target.stat().st_size == SIZE and digest.hexdigest() == SHA
    receipt = {'url': URL, 'size': SIZE, 'sha256': SHA, 'verified': True, 'seconds': time.monotonic() - began}
    (ROOT / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)

if __name__ == '__main__':
    main()
