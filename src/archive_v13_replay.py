"""Retain a private, bounded inventory for the completed inference replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/replay-v13'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    replay = json.loads((OUT / 'result.json').read_text())
    assert replay['identical_cells'] and replay['byte_identical']
    assert replay['pair_hands_replayed'] == 9651820
    destination = OUT / 'input-inventory.json'
    if destination.exists():
        manifest = json.loads(destination.read_text())
        for record in manifest['files']:
            path = ROOT / record['path']
            assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], record['path']
        print(json.dumps({'verified_files': len(manifest['files']), 'manifest_sha256': sha(destination)}))
        return

    source = OUT / 'source-snapshot'
    source.mkdir(exist_ok=True)
    for path in sorted((ROOT / 'src').glob('*.py')):
        target = source / path.name
        if target.exists():
            assert sha(target) == sha(path)
        else:
            shutil.copy2(path, target)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, capture_output=True, text=True).stdout
    (OUT / 'requirements-observed.txt').write_text(freeze, encoding='utf-8')
    paths = set(source.glob('*.py'))
    direct = [
        'data/sample_submission.csv', 'outputs_v2/pairs_evaluation.parquet',
        'outputs_v2/submission_advanced_robust.csv.provenance.json',
        'outputs_v3/rank_batches/manifest.json', 'outputs_v6/fit.json',
        'outputs_v11_conditional/fit.json', 'outputs_v11_conditional/inference-package/config.json',
        'outputs_v13_coverage/ready-audit.json', 'outputs_v13_coverage/submission_conditional_coverage.csv',
        'reports/replay-v13/plan.json', 'reports/replay-v13/result.json',
        'reports/replay-v13/requirements-observed.txt',
    ]
    plan = json.loads((OUT / 'plan.json').read_text())
    direct += [record['path'] for record in plan['models'].values()]
    direct += ['outputs_v11_conditional/inference-package/data/' + name for name in [
        'hands.parquet', 'seats.parquet', 'actions.parquet', 'evaluation_pairs.csv']]
    paths.update(ROOT / path for path in direct)
    patterns = {
        'outputs_v2/hands_evaluation': ['batch_*.parquet'],
        'outputs_v11_conditional/inference-package/results': ['batch_*-pairs.parquet', 'batch_*-evidence.parquet', 'batch_*.json'],
        'reports/evidence-public-calibration': ['batch_*-features.parquet'],
        'outputs_v13_coverage/batches': ['batch_*-features.parquet', 'batch_*.json'],
    }
    for directory, globs in patterns.items():
        for pattern in globs:
            matches = sorted((ROOT / directory).glob(pattern))
            assert len(matches) == 20, (directory, pattern, len(matches))
            paths.update(matches)
    records = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(paths)]
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'scope': 'Private snapshot and inventory for inference replay from frozen models and caches. No clean-environment installation, raw-feature rebuild, or model retraining is claimed.',
        'source_snapshot': 'All current src/*.py retained privately; not all are dependencies of this replay.',
        'files': records, 'total_bytes': sum(r['bytes'] for r in records),
    }
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'inventoried_files': len(records), 'bytes': report['total_bytes'], 'manifest_sha256': sha(destination)}))


if __name__ == '__main__':
    main()
