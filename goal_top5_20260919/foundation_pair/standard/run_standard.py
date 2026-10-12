from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tabpfn35'))
import runtime35
import argparse
import gc
import hashlib
import json
import time
import numpy as np
import torch
from tabpfn import TabPFNClassifier

OUT = Path(__file__).resolve().parent
SCREEN = OUT.parent / 'screen'
RT = ROOT / 'tabpfn35'
CK = RT / 'tabpfn-v3.5-20260909_multiclass.safetensors'
source = json.loads((RT / 'standard_multiclass_source.json').read_text())
with CK.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == source['model_sha256']
gate = json.loads((RT / 'standard_preflight.json').read_text())
assert len(gate) == 3 and all(x['passed'] for x in gate), 'Incomplete or failed local hardware gate'
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(.70)

def fixed4(p, classes):
    out = np.zeros((len(p), 4), dtype=np.float64)
    out[:, classes.astype(int)] = p
    assert np.isfinite(out).all() and np.allclose(out.sum(1), 1, atol=1e-5)
    return out

def predict(model, x):
    return fixed4(np.concatenate([model.predict_proba(x[i:i+64]) for i in range(0, len(x), 64)]), model.classes_)

def run(fold, omitted=None, replay=False):
    name = f'fold{fold}' + (f'_omit{omitted}' if omitted else '')
    context = SCREEN / ('context_' + name + '.npz')
    z = np.load(context, allow_pickle=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = TabPFNClassifier(model_path=str(CK), device='cuda', n_estimators=4,
                            auto_scale_n_estimators=False, fit_mode='low_memory',
                            memory_saving_mode=True, random_state=35935,
                            n_preprocessing_jobs=1)
    model.fit(z['X_train'], z['y_train'])
    predictions = predict(model, z['X_held'])
    dest = OUT / (('replay_' if replay else 'predictions_') + name + '.npz')
    assert not dest.exists(), f'Preserve existing {dest}'
    np.savez_compressed(dest, probabilities=predictions, pair_id=z['pair_id'], classes=model.classes_)
    if omitted is None and not replay:
        pre = np.load(SCREEN / f'context_fold{fold}_prefix60.npz', allow_pickle=False)
        np.savez_compressed(OUT / f'predictions_fold{fold}_prefix60.npz',
                            probabilities=predict(model, pre['X_held']), pair_id=pre['pair_id'])
    receipt = {'context': name, 'seconds': time.monotonic()-started,
               'context_sha256': hashlib.sha256(context.read_bytes()).hexdigest(),
               'checkpoint_sha256': source['model_sha256'],
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'classes': model.classes_.tolist(), 'seed': 35935, 'members': 4, 'query_batch': 64}
    (OUT / (('replay_' if replay else 'runtime_') + name + '.json')).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    assert receipt['peak_reserved_bytes'] < 3 * 1024**3
    if replay:
        original = np.load(OUT / f'predictions_fold{fold}.npz')
        assert np.array_equal(original['pair_id'], z['pair_id'])
        exact = np.array_equal(original['probabilities'], predictions)
        (OUT / 'replay_check.json').write_text(json.dumps({'exact': bool(exact), 'max_abs': float(np.max(np.abs(original['probabilities']-predictions)))}, indent=2))
        assert exact
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['known', 'replay', 'lofo'])
    args = parser.parse_args()
    if args.mode == 'known':
        for f in (0, 1):
            run(f)
    elif args.mode == 'replay':
        run(0, replay=True)
    else:
        verdict = json.loads((OUT / 'known_results.json').read_text())
        assert verdict['advance_to_lofo'], 'Frozen advance condition failed'
        for f in (0, 1):
            for family in (1, 2, 3):
                run(f, omitted=family)
