"""Package only private validation inputs for the user-provided compute instance."""
import hashlib
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/gpu-objective-experiment'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    jobs = {'fold0': 'reports/tabicl-screen'}
    jobs.update({n: 'reports/tabicl-confirmation/' + n for n in
                 ['fold1', 'fold2', 'omit_directed_transfer', 'omit_soft_play', 'omit_coordinated_isolation']})
    files = {}
    for job, directory in jobs.items():
        for filename in ['arrays.npz', 'features.json', 'held-pairs.csv', 'predictions.npz']:
            files[f'inputs/{job}/{filename}'] = ROOT / directory / filename
    files['run_gpu_objective_experiment.py'] = ROOT / 'src/run_gpu_objective_experiment.py'
    files['LICENSE-TabICL'] = ROOT / 'external/tabicl/LICENSE'
    plan = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Compare larger pretrained ensembles and binary versus family objectives on existing pool splits.',
        'jobs': list(jobs),
        'recipes': {'gpu_control': {'n_estimators': 2, 'binary': False},
                    'multiclass8': {'n_estimators': 8, 'binary': False},
                    'binary8': {'n_estimators': 8, 'binary': True}},
        'checkpoint_repo': 'jingang/TabICL',
        'checkpoint_revision': '4dcd344ece2c00be9e831fdd35bed57b5ad83e19',
        'checkpoint_file': 'tabicl-classifier-v2-20260212.ckpt',
        'checkpoint_sha256': 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0',
        'common_parameters': {'batch_size': 1, 'device': 'cuda', 'n_jobs': 4, 'random_state': 7329,
                              'kv_cache': True, 'use_amp': False, 'use_fa3': False},
        'selection': 'For each new recipe evaluate standalone risk and the fixed equal arithmetic blend with v5. Relative to the frozen CPU v10 blend, require no ordinary AP decline exceeding .003 on any known fold/history, mean shortened hypothetical 1% AP gain >.002, omitted sparse mean >=0 and minimum >=-.03. Select the passing option with the highest mean shortened sparse AP. No evaluation predictions or submission automatically follows.',
        'limitations': 'Reused overlapping splits; selection remains adaptive. One-percent prevalence is hypothetical. No tuning or calibration on evaluation data. GPU and CPU library versions may differ; record control differences.',
        'files': {name: sha(path) for name, path in files.items()},
    }
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    bundle = OUT / 'validation-bundle.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        archive.add(OUT / 'plan.json', arcname='plan.json')
        for name, path in files.items():
            archive.add(path, arcname=name)
    (OUT / 'bundle-receipt.json').write_text(json.dumps({'sha256': sha(bundle), 'bytes': bundle.stat().st_size}, indent=2) + '\n')
    print(json.dumps({'bundle': str(bundle), 'bytes': bundle.stat().st_size, 'jobs': list(jobs)}))


if __name__ == '__main__':
    main()
