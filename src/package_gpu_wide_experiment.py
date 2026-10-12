"""Freeze the validation-only experiment plan before any GPU outcomes."""
from pathlib import Path
import hashlib
import json
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/gpu-wide-experiment'


def main():
    files = {p.relative_to(OUT).as_posix(): p for p in (OUT / 'inputs').rglob('*') if p.is_file()}
    files['run_gpu_wide_experiment.py'] = ROOT / 'src/run_gpu_wide_experiment.py'
    files['preparation-plan.json'] = OUT / 'preparation-plan.json'
    files['LICENSE-TabICL'] = ROOT / 'external/tabicl/LICENSE'
    checkpoint = json.loads((ROOT / 'reports/gpu-objective-experiment/plan.json').read_text())
    plan = {k: v for k, v in checkpoint.items() if k.startswith('checkpoint_')}
    plan.update({'created_at': datetime.now(timezone.utc).isoformat(), 'jobs': checkpoint['jobs'],
        'inference_parameters': {'n_estimators': 2, 'batch_size': 1, 'device': 'cuda', 'n_jobs': 4,
            'random_state': 7329, 'kv_cache': True, 'use_amp': False, 'use_fa3': False},
        'finetune_parameters': {'epochs': 15, 'learning_rate': 1e-5, 'amp': False,
            'n_estimators_finetune': 2, 'n_estimators_validation': 2, 'n_estimators_inference': 2,
            'max_data_size': 512, 'early_stopping': True, 'patience': 4, 'time_limit': 180,
            'save_interval': 5, 'device': 'cuda', 'random_state': 7329, 'verbose': True,
            'eval_metric': 'log_loss', 'extra_classifier_kwargs': {'batch_size': 1, 'kv_cache': True,
                'use_amp': False, 'use_fa3': False, 'n_jobs': 4}},
        'selection': checkpoint['selection'],
        'scope': 'Two wider-feature recipes plus one inner-pool fine-tuning recipe and matching unfitted-weight control. Only validation inputs. No automatic submission or evaluation predictions. Inner feature selection excludes inner validation labels. Reused outer splits remain adaptive.',
        'files': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items()}})
    plan_path = OUT / 'plan.json'
    assert not plan_path.exists()
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    archive = OUT / 'wide-bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        t.add(plan_path, arcname='plan.json')
        for name, file in files.items():
            t.add(file, arcname=name)
    receipt = {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size}
    (OUT / 'bundle-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
