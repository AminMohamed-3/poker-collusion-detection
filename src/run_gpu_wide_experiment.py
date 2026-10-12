"""Fixed wider-feature and inner-pool fine-tuning tests, with separate receipts."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '4'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tabicl import TabICLClassifier
from sklearn.metrics import average_precision_score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(y, risk):
    positive = y > 0
    assert np.isfinite(risk).all() and np.all((risk >= 0) & (risk <= 1))
    weights = np.where(positive, .01 / positive.mean(), .99 / (1 - positive.mean()))
    return {'pair_ap': float(average_precision_score(positive, risk)),
            'sparse_ap': float(average_precision_score(positive, risk, sample_weight=weights))}


def main():
    root = Path(__file__).resolve().parent
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    assert all(sha(root / n) == h for n, h in plan['files'].items())
    phase = sys.argv[1] if len(sys.argv) > 1 else 'wide'
    assert phase in ['wide', 'finetune']
    out = root / ('results-' + phase)
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    checkpoint = Path(hf_hub_download(plan['checkpoint_repo'], plan['checkpoint_file'], revision=plan['checkpoint_revision']))
    assert sha(checkpoint) == plan['checkpoint_sha256']
    environment = {'plan_sha256': sha(plan_path), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'packages': {p: importlib.metadata.version(p) for p in ['tabicl', 'numpy', 'scikit-learn']}}
    envpath = out / 'environment.json'
    if envpath.exists():
        assert json.loads(envpath.read_text()) == environment
    else:
        envpath.write_text(json.dumps(environment, indent=2) + '\n')
    recipes = ['features256', 'features765'] if phase == 'wide' else ['inner_control', 'finetuned']
    rows = []
    for job in plan['jobs']:
        arrays = np.load(root / 'inputs' / job / 'arrays.npz', allow_pickle=False)
        y = arrays['y_held']
        for history in ['full', 'shortened']:
            base = .5 * (arrays['cpu_' + history] + arrays['control_' + history])
            rows.append({'job': job, 'history': history, 'candidate': 'cpu_v10', **metric(y, base)})
        for recipe in recipes:
            name = job + '__' + recipe
            receipt = out / (name + '.json')
            predictions_file = out / (name + '.npz')
            if receipt.exists():
                result = json.loads(receipt.read_text())
                assert result['plan_sha256'] == sha(plan_path) and result['predictions_sha256'] == sha(predictions_file)
                rows.extend(result['results'])
                continue
            start = time.perf_counter()
            if phase == 'wide':
                count = int(recipe.replace('features', ''))
                model = TabICLClassifier(model_path=checkpoint, allow_auto_download=False, **plan['inference_parameters'])
                model.fit(arrays['x_train'][:, :count], arrays['y_train'])
            elif recipe == 'inner_control':
                model = TabICLClassifier(model_path=checkpoint, allow_auto_download=False, **plan['inference_parameters'])
                model.fit(arrays['ft_x_train'], arrays['ft_y_train'])
            else:
                from tabicl import FinetunedTabICLClassifier
                model = FinetunedTabICLClassifier(model_path=checkpoint, allow_auto_download=False, **plan['finetune_parameters'])
                model.fit(arrays['ft_x_train'], arrays['ft_y_train'], X_val=arrays['ft_x_val'],
                          y_val=arrays['ft_y_val'], output_dir=out / (name + '_checkpoints'))
            result_rows, predictions = [], {}
            for history in ['full', 'shortened']:
                X = arrays['x_' + history][:, :count] if phase == 'wide' else arrays['ft_x_' + history]
                probability = model.predict_proba(X)
                assert model.classes_[0] == 0 and np.isfinite(probability).all()
                np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-5)
                risk = 1 - probability[:, 0]
                predictions[history + '_probability'] = probability
                predictions['classes'] = model.classes_
                for blend in [False, True]:
                    values = .5 * (risk + arrays['control_' + history]) if blend else risk
                    result_rows.append({'job': job, 'history': history, 'candidate': recipe + ('_blend' if blend else ''), **metric(y, values)})
            np.savez_compressed(predictions_file, **predictions)
            result = {'plan_sha256': sha(plan_path), 'predictions_sha256': sha(predictions_file),
                'seconds': time.perf_counter() - start, 'results': result_rows,
                'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30}
            receipt.write_text(json.dumps(result, indent=2) + '\n')
            rows.extend(result_rows)
            print(json.dumps({'job': job, 'recipe': recipe, 'seconds': result['seconds'], 'results': result_rows}), flush=True)
            del model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    lookup = {(r['job'], r['history'], r['candidate']): r for r in rows}
    candidates = [r + suffix for r in recipes for suffix in ['', '_blend'] if r != 'inner_control']
    comparisons, passing = {}, []
    for candidate in candidates:
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
        'selected_for_further_review': selected}, indent=2) + '\n')
    print(json.dumps({'selected_for_further_review': selected, 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
