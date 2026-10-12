"""Fit a confirmed TabICL recipe and produce a risk-only candidate in resumable chunks."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tabicl import TabICLClassifier


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'outputs_v10_tabicl'
    plan_path = out / 'plan.json'
    plan = json.loads(plan_path.read_text())
    assert plan['selected'] in ['tabicl', 'equal_blend']
    assert all(sha(root / p) == value for p, value in plan['inputs'].items())
    assert sha(out / 'arrays.npz') == plan['array_sha256']
    checkpoint = json.loads((root / 'reports/tabicl-screen/checkpoint.json').read_text())
    assert sha(checkpoint['path']) == checkpoint['sha256']
    arrays = np.load(out / 'arrays.npz', allow_pickle=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model_path = out / 'tabicl.pkl'
    fit_path = out / 'fit.json'
    start = time.perf_counter()
    if fit_path.exists():
        fitted = json.loads(fit_path.read_text())
        assert sha(model_path) == fitted['model_sha256'] and sha(plan_path) == fitted['plan_sha256']
        model = TabICLClassifier.load(model_path, device='cpu')
    else:
        assert not model_path.exists()
        model = TabICLClassifier(**plan['parameters'], model_path=checkpoint['path'], allow_auto_download=False)
        model.fit(arrays['x_train'], arrays['y_train'])
        probe = model.predict_proba(arrays['x_evaluation'][:64])
        model.save(model_path, save_model_weights=True, save_training_data=True, save_kv_cache=True)
        reloaded = TabICLClassifier.load(model_path, device='cpu')
        np.testing.assert_array_equal(probe, reloaded.predict_proba(arrays['x_evaluation'][:64]))
        del reloaded
        fitted = {'plan_sha256': sha(plan_path), 'model_sha256': sha(model_path), 'checkpoint_sha256': checkpoint['sha256'],
                  'reload_predictions_identical': True, 'fit_seconds': time.perf_counter() - start}
        fit_path.write_text(json.dumps(fitted, indent=2) + '\n')
        print(json.dumps(fitted), flush=True)
    cache = out / 'prediction_chunks'
    cache.mkdir(exist_ok=True)
    predictions = []
    X = arrays['x_evaluation']
    chunk = plan['inference_chunk_rows']
    for offset in range(0, len(X), chunk):
        file = cache / f'{offset:06d}.npy'
        receipt = cache / f'{offset:06d}.json'
        if receipt.exists():
            meta = json.loads(receipt.read_text())
            assert meta['model_sha256'] == fitted['model_sha256'] and meta['sha256'] == sha(file)
            risk = np.load(file, allow_pickle=False)
        else:
            t0 = time.perf_counter()
            probability = model.predict_proba(X[offset:offset + chunk])
            assert np.array_equal(model.classes_, np.arange(4))
            risk = 1 - probability[:, 0]
            assert np.isfinite(risk).all() and np.all((risk >= 0) & (risk <= 1))
            np.save(file, risk)
            receipt.write_text(json.dumps({'model_sha256': fitted['model_sha256'], 'sha256': sha(file),
                'offset': offset, 'rows': len(risk), 'seconds': time.perf_counter() - t0}, indent=2) + '\n')
            print(json.dumps({'predicted': min(offset + chunk, len(X)), 'total': len(X), 'chunk_seconds': time.perf_counter() - t0}), flush=True)
        predictions.append(risk)
    risk = np.concatenate(predictions)
    with (out / 'evaluation-order.csv').open(newline='') as handle:
        order = [r['pair_id'] for r in csv.DictReader(handle)]
    assert len(order) == len(set(order)) == len(risk) == 112540
    prediction = dict(zip(order, risk))
    source = root / plan['source']
    with source.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert {r['pair_id'] for r in rows} == set(prediction)
    for row in rows:
        value = float(prediction[row['pair_id']])
        if plan['selected'] == 'equal_blend':
            value = .5 * (value + float(row['risk_score']))
        row['risk_score'] = repr(value)
    destination = out / 'submission_tabicl.csv'
    if destination.exists():
        with destination.open(newline='') as handle:
            assert list(csv.DictReader(handle)) == rows
    else:
        with destination.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    with source.open(newline='') as handle:
        originals = list(csv.DictReader(handle))
    assert all(all(a[c] == b[c] for c in fieldnames if c != 'risk_score') for a, b in zip(originals, rows))
    assert all(sha(root / p) == value for p, value in plan['inputs'].items())
    (out / 'provenance.json').write_text(json.dumps({'submission_sha256': sha(destination),
        'plan_sha256': sha(plan_path), 'rows': len(rows), 'changed_column': 'risk_score',
        'other_cells_preserved_exactly': True, 'membership_validation_required': True,
        'selected': plan['selected'], 'seconds': time.perf_counter() - start}, indent=2) + '\n')
    print(f'Wrote {destination}; validation required.', flush=True)


if __name__ == '__main__':
    main()
