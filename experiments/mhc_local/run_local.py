"""Serial local GPU queue for the frozen experiment; no adaptive selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', default='fold0')
    parser.add_argument('--variants', nargs='+')
    args = parser.parse_args()
    plan = json.loads((HERE / 'plan.json').read_text())
    variants = args.variants or list(plan['variants'])
    queue = HERE / args.job / 'queue.json'
    records = []
    for variant in variants:
        assert variant in plan['variants']
        for seed in plan['seeds']:
            dest = HERE / args.job / f'{variant}-{seed}'
            if dest.exists():
                r = json.loads((dest / 'receipt.json').read_text())
                assert r['plan_sha256'] == hashlib.sha256((HERE / 'plan.json').read_bytes()).hexdigest()
                assert hashlib.sha256((dest / 'model.pt').read_bytes()).hexdigest() == r['model_sha256']
                assert hashlib.sha256((dest / 'predictions.npz').read_bytes()).hexdigest() == r['predictions_sha256']
                print(f'Existing verified complete fit: {dest.name}', flush=True)
            else:
                command = [sys.executable, '-u', str(HERE / 'train.py'), '--job', args.job,
                           '--variant', variant, '--seed', str(seed), '--device', 'cuda']
                print(f'Starting {args.job}/{dest.name}', flush=True)
                with (HERE / args.job / f'{dest.name}.log').open('w') as log:
                    proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if proc.returncode:
                    raise RuntimeError(f'Fit failed ({proc.returncode}); inspect {dest.name}.log. No retry or outcome-dependent change.')
            records.append({'variant': variant, 'seed': seed, 'verified_at': datetime.now(timezone.utc).isoformat()})
            queue.write_text(json.dumps({'completed': records, 'total': len(variants) * len(plan['seeds'])}, indent=2))
            print(f'Completed {len(records)}/{len(variants) * len(plan["seeds"])}', flush=True)
            if (HERE / 'STOP_AFTER_CURRENT').exists():
                print('Requested machine migration: stopped after a complete fit.', flush=True)
                return

if __name__ == '__main__':
    main()
