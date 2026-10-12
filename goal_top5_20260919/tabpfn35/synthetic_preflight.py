"""Offline, label-free hardware gate for the verified TabPFN-3.5-Fast checkpoint."""
import runtime35
from runtime35 import HERE
import gc
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from tabpfn import TabPFNClassifier

model_path = HERE / "tabpfn-v3.5-fast-20260909.safetensors"
assert hashlib.sha256(model_path.read_bytes()).hexdigest() == "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550"
assert torch.cuda.is_available()
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.70)
results = []
parser = argparse.ArgumentParser()
parser.add_argument("--large", action="store_true")
args = parser.parse_args()
shapes = [(2048, 4), (4096, 4), (8192, 4), (10438, 4)] if args.large else [(64, 1), (1024, 1), (1024, 4)]
output_name = "large_context_preflight.json" if args.large else "synthetic_preflight.json"
for rows, estimators in shapes:
    rng = np.random.default_rng(2773)
    X = rng.normal(size=(rows, 64)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 0.3 > 0).astype(np.int64)
    query = rng.normal(size=(256, 64)).astype(np.float32)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = TabPFNClassifier(model_path=str(model_path), device="cuda", n_estimators=estimators,
                            auto_scale_n_estimators=False, fit_mode="low_memory",
                            memory_saving_mode=True, random_state=2773,
                            n_preprocessing_jobs=1)
    model.fit(X, y)
    pred = model.predict_proba(query)
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_reserved()
    assert pred.shape == (256, 2) and np.isfinite(pred).all()
    passed = elapsed < 120 and peak < 3 * 1024 ** 3
    result = {"rows": rows, "features": 64, "query_rows": 256, "estimators": estimators,
              "seconds": elapsed, "peak_reserved_bytes": peak, "passed": passed}
    results.append(result)
    (HERE / output_name).write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    del model, pred
    gc.collect()
    torch.cuda.empty_cache()
    assert passed, result
