"""Evaluate five fixed pretrained-model jobs, recording results without tuning."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tabicl import TabICLClassifier
from run_tabicl_screen import ap


def main():
    root = Path(__file__).resolve().parents[1]
    report = root / 'reports/tabicl-confirmation'
    plan = json.loads((report / 'plan.json').read_text())
    checkpoint = json.loads((root / 'reports/tabicl-screen/checkpoint.json').read_text())
    assert hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest() == checkpoint['sha256']
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    allresults = []
    for job in plan['jobs']:
        out = report / job
        if (out / 'metrics.json').exists():
            allresults.extend(json.loads((out / 'metrics.json').read_text())['results'])
            continue
        if (out / 'run-start.json').exists():
            raise FileExistsError('Inspect incomplete run before any restart')
        arrays_path = out / 'arrays.npz'
        assert hashlib.sha256(arrays_path.read_bytes()).hexdigest() == json.loads((out / 'features.json').read_text())['array_sha256']
        (out / 'run-start.json').write_text(json.dumps({'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'checkpoint_sha256': checkpoint['sha256'], 'device': 'cpu'}, indent=2) + '\n')
        arrays = np.load(arrays_path, allow_pickle=False)
        model = TabICLClassifier(n_estimators=2, batch_size=1, model_path=checkpoint['path'], allow_auto_download=False,
            device='cpu', n_jobs=2, random_state=7329, kv_cache=True, use_amp=False)
        start = time.perf_counter()
        model.fit(arrays['x_train'], arrays['y_train'])
        y = arrays['y_held']
        positive = y > 0
        weights = np.where(positive, .01 / positive.mean(), .99 / (1 - positive.mean()))
        results, predictions = [], {}
        for history in ['full', 'shortened']:
            probabilities = model.predict_proba(arrays['x_' + history])
            assert model.classes_[0] == 0 and np.isfinite(probabilities).all()
            risk = 1 - probabilities[:, 0]
            predictions[history] = risk
            control = arrays['control_' + history]
            for name, values in [('v5_control', control), ('tabicl', risk), ('equal_blend', .5 * (control + risk))]:
                result = {'job': job, 'history': history, 'candidate': name,
                          'pair_ap': ap(y, values), 'hypothetical_1pct_ap': ap(y, values, weights)}
                results.append(result)
        np.savez_compressed(out / 'predictions.npz', **predictions)
        (out / 'metrics.json').write_text(json.dumps({'results': results, 'seconds': time.perf_counter() - start}, indent=2) + '\n')
        allresults.extend(results)
        print(json.dumps({'job': job, 'seconds': time.perf_counter() - start, 'results': results}), flush=True)
        del model
    first = json.loads((root / 'reports/tabicl-screen/metrics.json').read_text())['results']
    allresults.extend([{**r, 'job': 'fold0'} for r in first])
    lookup = {(r['job'], r['history'], r['candidate']): r for r in allresults}
    comparisons = {}
    passing = []
    for name in ['tabicl', 'equal_blend']:
        known = [lookup[j, h, name]['pair_ap'] - lookup[j, h, 'v5_control']['pair_ap']
                 for j in ['fold0', 'fold1', 'fold2'] for h in ['full', 'shortened']]
        sparse = [lookup[j, 'shortened', name]['hypothetical_1pct_ap'] - lookup[j, 'shortened', 'v5_control']['hypothetical_1pct_ap'] for j in ['fold0', 'fold1', 'fold2']]
        omitted = [lookup[j, h, name]['hypothetical_1pct_ap'] - lookup[j, h, 'v5_control']['hypothetical_1pct_ap']
                   for j in plan['jobs'] if j.startswith('omit_') for h in ['full', 'shortened']]
        passed = min(known) >= -.003 and np.mean(sparse) > 0 and np.mean(omitted) >= 0 and min(omitted) >= -.03
        comparisons[name] = {'known_ap_deltas': known, 'known_short_sparse_deltas': sparse,
                             'omitted_sparse_deltas': omitted, 'gate_passed': bool(passed)}
        if passed:
            passing.append(name)
    selected = max(passing, key=lambda n: np.mean(comparisons[n]['known_short_sparse_deltas'])) if passing else None
    (report / 'metrics.json').write_text(json.dumps({'results': allresults, 'comparisons': comparisons,
        'selected': selected, 'limitations': 'Reused overlapping known folds and proxy-unseen families; not a test-set score.'}, indent=2) + '\n')
    print(json.dumps({'selected': selected, 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
