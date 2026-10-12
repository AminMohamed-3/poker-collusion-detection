"""Label-free hardware gate at the two frozen training-context sizes."""
import gc, hashlib, os, time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
from tabpfn import TabPFNClassifier

HERE = Path(__file__).resolve().parent
MODEL = HERE / "tabpfn-v3.5-20260909_multiclass.safetensors"
EXPECTED = "4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149"
with MODEL.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == EXPECTED
assert torch.cuda.is_available()
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(.70)

for rows in (1024, 1860):
    rng = np.random.default_rng(35935 + rows)
    x = rng.normal(size=(rows, 128)).astype(np.float32)
    y = np.arange(rows, dtype=np.int64) % 4
    query = rng.normal(size=(64, 128)).astype(np.float32)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = TabPFNClassifier(
        model_path=str(MODEL), device="cuda", n_estimators=4,
        auto_scale_n_estimators=False, fit_mode="low_memory",
        memory_saving_mode=True, random_state=35935, n_preprocessing_jobs=1,
    )
    model.fit(x, y)
    probability = model.predict_proba(query)
    receipt = {"rows": rows, "features": 128, "query": 64, "members": 4,
               "seconds": time.monotonic() - started,
               "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
               "finite": bool(np.isfinite(probability).all())}
    print(receipt, flush=True)
    assert receipt["finite"] and probability.shape == (64, 4)
    del model, probability
    gc.collect()
    torch.cuda.empty_cache()
