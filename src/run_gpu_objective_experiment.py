"""Run an immutable, resumable validation-only ablation on the provided GPU."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '4'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from huggingface_hub import hf_hub_download
from tabicl import TabICLClassifier


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, risk):
    positive = y > 0
    assert np.isfinite(risk).all() and np.all((risk >= 0) & (risk <= 1))
    weight = np.where(positive, .01 / positive.mean(), .99 / (1 - positive.mean()))
    return {'pair_ap': float(average_precision_score(positive, risk)),
            'sparse_ap': float(average_precision_score(positive, risk, sample_weight=weight)),
            'unique_scores': len(np.unique(risk))}


def main():
    root = Path(__file__).resolve().parent
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    assert all(sha(root / name) == digest for name, digest in plan['files'].items())
    out = root / 'results'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    assert torch.cuda.is_available()
    checkpoint = Path(hf_hub_download(plan['checkpoint_repo'], plan['checkpoint_file'], revision=plan['checkpoint_revision']))
    assert sha(checkpoint) == plan['checkpoint_sha256']
    environment = {'plan_sha256': sha(plan_path), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
                   'packages': {p: importlib.metadata.version(p) for p in ['tabicl', 'numpy', 'scikit-learn']}}
    env_path = out / 'environment.json'
    if env_path.exists():
        assert json.loads(env_path.read_text()) == environment
    else:
        env_path.write_text(json.dumps(environment, indent=2) + '\n')
    print(json.dumps(environment), flush=True)
    rows = []
    differences = []
    for job in plan['jobs']:
        arrays = np.load(root / 'inputs' / job / 'arrays.npz', allow_pickle=False)
        cpu = np.load(root / 'inputs' / job / 'predictions.npz', allow_pickle=False)
        y = arrays['y_held']
        for history in ['full', 'shortened']:
            baseline = .5 * (arrays['control_' + history] + cpu[history])
            rows.append({'job': job, 'history': history, 'candidate': 'cpu_v10', **metrics(y, baseline)})
        for recipe, params in plan['recipes'].items():
            name = job + '__' + recipe
            receipt = out / (name + '.json')
            predictions_file = out / (name + '.npz')
            if receipt.exists():
                result = json.loads(receipt.read_text())
                assert result['plan_sha256'] == sha(plan_path) and result['predictions_sha256'] == sha(predictions_file)
                rows.extend(result['results'])
                differences.extend(result['cpu_comparison'])
                continue
            start = time.perf_counter()
            model = TabICLClassifier(n_estimators=params['n_estimators'], model_path=checkpoint,
                allow_auto_download=False, **plan['common_parameters'])
            target = arrays['y_train'] > 0 if params['binary'] else arrays['y_train']
            model.fit(arrays['x_train'], target)
            predictions, result_rows, delta_rows = {}, [], []
            for history in ['full', 'shortened']:
                probability = model.predict_proba(arrays['x_' + history])
                assert model.classes_[0] == 0 and np.isfinite(probability).all()
                np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-5)
                risk = 1 - probability[:, 0]
                predictions[history + '_probability'] = probability
                predictions['classes'] = model.classes_
                for blend in [False, True]:
                    value = .5 * (risk + arrays['control_' + history]) if blend else risk
                    result_rows.append({'job': job, 'history': history,
                        'candidate': recipe + ('_blend' if blend else ''), **metrics(y, value)})
                if recipe == 'gpu_control':
                    delta_rows.append({'job': job, 'history': history,
                        'cpu_max_abs_risk_difference': float(np.max(np.abs(risk - cpu[history]))),
                        'cpu_pair_ap_difference': metrics(y, risk)['pair_ap'] - metrics(y, cpu[history])['pair_ap']})
            np.savez_compressed(predictions_file, **predictions)
            result = {'plan_sha256': sha(plan_path), 'predictions_sha256': sha(predictions_file),
                      'seconds': time.perf_counter() - start, 'results': result_rows, 'cpu_comparison': delta_rows,
                      'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30}
            receipt.write_text(json.dumps(result, indent=2) + '\n')
            rows.extend(result_rows)
            differences.extend(delta_rows)
            print(json.dumps({'job': job, 'recipe': recipe, 'seconds': result['seconds'], 'results': result_rows}), flush=True)
            del model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    lookup = {(r['job'], r['history'], r['candidate']): r for r in rows}
    comparisons = {}
    passing = []
    for candidate in ['multiclass8', 'multiclass8_blend', 'binary8', 'binary8_blend']:
        known = [lookup[j, h, candidate]['pair_ap'] - lookup[j, h, 'cpu_v10']['pair_ap']
                 for j in ['fold0', 'fold1', 'fold2'] for h in ['full', 'shortened']]
        sparse = [lookup[j, 'shortened', candidate]['sparse_ap'] - lookup[j, 'shortened', 'cpu_v10']['sparse_ap']
                  for j in ['fold0', 'fold1', 'fold2']]
        omitted = [lookup[j, h, candidate]['sparse_ap'] - lookup[j, h, 'cpu_v10']['sparse_ap']
                   for j in plan['jobs'] if j.startswith('omit_') for h in ['full', 'shortened']]
        passed = min(known) >= -.003 and np.mean(sparse) > .002 and np.mean(omitted) >= 0 and min(omitted) >= -.03
        comparisons[candidate] = {'known_ap_deltas': known, 'short_sparse_deltas': sparse,
                                  'omitted_sparse_deltas': omitted, 'passed': bool(passed)}
        if passed:
            passing.append(candidate)
    selected = max(passing, key=lambda c: np.mean(comparisons[c]['short_sparse_deltas'])) if passing else None
    (out / 'summary.json').write_text(json.dumps({'results': rows, 'comparisons': comparisons,
        'cpu_gpu_comparisons': differences, 'selected_for_further_review': selected}, indent=2) + '\n')
    print(json.dumps({'selected_for_further_review': selected, 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
