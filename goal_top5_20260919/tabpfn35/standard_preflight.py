"""Synthetic multiclass hardware gate, no poker labels or predictions."""
import runtime35
from runtime35 import HERE
import gc
import hashlib
import json
import time
import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.errors import TabPFNCUDAOutOfMemoryError

path = HERE / 'tabpfn-v3.5-20260909_multiclass.safetensors'
source = json.loads((HERE / 'standard_multiclass_source.json').read_text())
with path.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == source['model_sha256']
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(.70)
results = []
for rows, members in [(128, 1), (1024, 4), (1860, 4)]:
    rng = np.random.default_rng(35935)
    x = rng.normal(size=(rows, 128)).astype(np.float32)
    y = np.argmax(x[:, :4], axis=1)
    query = rng.normal(size=(64, 128)).astype(np.float32)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    try:
        model = TabPFNClassifier(model_path=str(path), device='cuda', n_estimators=members,
                                 auto_scale_n_estimators=False, fit_mode='low_memory',
                                 memory_saving_mode=True, random_state=35935,
                                 n_preprocessing_jobs=1)
        model.fit(x, y)
        p = model.predict_proba(query)
        assert p.shape == (64, 4) and np.isfinite(p).all()
        result = {'rows': rows, 'features': 128, 'query': 64, 'members': members,
                  'seconds': time.monotonic()-start,
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'passed': True}
        del model, p
        gc.collect()
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, TabPFNCUDAOutOfMemoryError) as exc:
        result = {'rows': rows, 'features': 128, 'query': 64, 'members': members,
                  'seconds': time.monotonic()-start, 'passed': False, 'reason': 'GPU memory cap',
                  'error': str(exc)}
    results.append(result)
    (HERE / 'standard_preflight.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        break
