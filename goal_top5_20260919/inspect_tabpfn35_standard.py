"""Read official model metadata; optional verified streaming checkpoint download."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent / 'tabpfn35'
NAME = 'tabpfn-v3.5-20260909_multiclass.safetensors'
BASE = 'https://huggingface.co/Prior-Labs/tabpfn_3_5'

def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'poker-research/1.0'}), timeout=90)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    with request('https://huggingface.co/api/models/Prior-Labs/tabpfn_3_5/tree/main') as f:
        entries = json.load(f)
    model = next(x for x in entries if x['path'] == NAME)
    with request(BASE + '/raw/main/README.md') as f:
        card = f.read()
    (OUT / 'MODEL_CARD.md').write_bytes(card)
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'official_metadata': model,
               'model_card_sha256': hashlib.sha256(card).hexdigest(), 'downloaded': False,
               'poker_data_sent': False}
    if args.download:
        final = OUT / NAME
        if final.exists():
            digest = hashlib.file_digest(final.open('rb'), 'sha256').hexdigest()
        else:
            partial = OUT / (NAME + '.partial')
            digest_obj = hashlib.sha256()
            with request(BASE + '/resolve/main/' + NAME) as stream, partial.open('wb') as target:
                while chunk := stream.read(4 * 1024 * 1024):
                    digest_obj.update(chunk)
                    target.write(chunk)
            digest = digest_obj.hexdigest()
            assert partial.stat().st_size == model['size']
            assert digest == model['lfs']['oid']
            partial.rename(final)
        assert final.stat().st_size == model['size'] and digest == model['lfs']['oid']
        receipt.update(downloaded=True, model_sha256=digest, bytes=final.stat().st_size)
    (OUT / 'standard_multiclass_source.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
