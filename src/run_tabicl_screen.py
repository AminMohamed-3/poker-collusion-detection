"""Run the frozen screen in an isolated Python environment; CPU or optional GPU."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
from tabicl import TabICLClassifier


def ap(truth, risk, weight=None):
    truth = np.asarray(truth) > 0
    weight = np.ones(len(truth)) if weight is None else weight
    order = np.argsort(-risk, kind='stable')
    positive = truth[order] * weight[order]
    return float(np.sum(np.cumsum(positive) / np.cumsum(weight[order]) * positive) / positive.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'reports/tabicl-screen'
    if (out / 'metrics.json').exists() or (out / 'run-start.json').exists():
        raise FileExistsError('Preserve previous run; inspect rather than silently restart')
    arrays_path = out / 'arrays.npz'
    metadata = json.loads((out / 'features.json').read_text())
    assert hashlib.sha256(arrays_path.read_bytes()).hexdigest() == metadata['array_sha256']
    plan = json.loads((out / 'plan.json').read_text())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    package_versions = {name: importlib.metadata.version(name) for name in ['tabicl', 'torch', 'numpy', 'scikit-learn']}
    (out / 'run-start.json').write_text(json.dumps({'device': args.device, 'packages': package_versions,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    arrays = np.load(arrays_path, allow_pickle=False)
    start = time.perf_counter()
    model = TabICLClassifier(n_estimators=plan['n_estimators'], batch_size=plan['batch_size'],
        checkpoint_version=plan['checkpoint'], device=args.device, n_jobs=2, random_state=7329,
        kv_cache=True, use_amp=False, verbose=True)
    model.fit(arrays['x_train'], arrays['y_train'])
    print(json.dumps({'fit_seconds': time.perf_counter() - start}), flush=True)
    y = arrays['y_held']
    positive = y > 0
    weights = np.where(positive, .01 / positive.mean(), .99 / (1 - positive.mean()))
    results = []
    predictions = {}
    for history in ['full', 'shortened']:
        t0 = time.perf_counter()
        probabilities = model.predict_proba(arrays['x_' + history])
        assert np.array_equal(model.classes_, np.arange(4))
        assert probabilities.shape == (len(y), 4) and np.isfinite(probabilities).all()
        risk = 1 - probabilities[:, 0]
        predictions[history] = risk
        control = arrays['control_' + history]
        for name, values in [('v5_control', control), ('tabicl', risk), ('equal_blend', .5 * (control + risk))]:
            result = {'history': history, 'candidate': name, 'pair_ap': ap(y, values),
                      'hypothetical_1pct_ap': ap(y, values, weights), 'seconds': time.perf_counter() - t0}
            results.append(result)
            print(json.dumps(result), flush=True)
    np.savez_compressed(out / 'predictions.npz', **predictions)
    lookup = {(r['history'], r['candidate']): r for r in results}
    passing = [name for name in ['tabicl', 'equal_blend'] if
        lookup['shortened', name]['hypothetical_1pct_ap'] > lookup['shortened', 'v5_control']['hypothetical_1pct_ap'] and
        min(lookup[h, name]['pair_ap'] - lookup[h, 'v5_control']['pair_ap'] for h in ['full', 'shortened']) >= -.003]
    (out / 'metrics.json').write_text(json.dumps({'results': results, 'advance': passing,
        'seconds': time.perf_counter() - start, 'packages': package_versions,
        'limitations': 'Single reused pool split; external pretrained model does not add real collusion labels.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
