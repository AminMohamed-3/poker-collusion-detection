"""Download only the public, Apache-2.0 structured PokerBench training data."""
import hashlib
import json
from pathlib import Path
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'external/pokerbench'


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'poker-research/1.0'}), timeout=120)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        for name, item in saved['files'].items():
            assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == item['sha256']
        print('Existing PokerBench download verified')
        return
    with get('https://huggingface.co/api/datasets/RZ412/PokerBench') as response:
        meta = json.load(response)
    revision = meta['sha']
    assert meta['cardData']['license'] == 'apache-2.0'
    names = ['README.md', 'preflop_60k_train_set_game_scenario_information.csv', 'postflop_500k_train_set_game_scenario_information.csv']
    files = {}
    for name in names:
        assert name in {item['rfilename'] for item in meta['siblings']}
        url = f'https://huggingface.co/datasets/RZ412/PokerBench/resolve/{revision}/{name}'
        path = OUT / name
        with get(url) as response, path.with_suffix(path.suffix + '.tmp').open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        path.with_suffix(path.suffix + '.tmp').replace(path)
        files[name] = {'url': url, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        print(json.dumps({'downloaded': name, 'bytes': path.stat().st_size}), flush=True)
    manifest.write_text(json.dumps({'retrieved_at': datetime.now(timezone.utc).isoformat(), 'revision': revision,
                                   'license': 'apache-2.0', 'files': files, 'used_in_submission': False,
                                   'purpose': 'Optional action-quality baseline. Solver choices are not collusion labels.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
