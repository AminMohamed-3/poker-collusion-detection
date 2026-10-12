from __future__ import annotations
import os
for key, value in {
    'TABPFN_DISABLE_TELEMETRY': '1',
    'TABPFN_NO_BROWSER': '1',
    'HF_HUB_OFFLINE': '1',
    'POLARS_MAX_THREADS': '2',
    'OMP_NUM_THREADS': '2',
    'OPENBLAS_NUM_THREADS': '2',
    'MKL_NUM_THREADS': '2',
}.items():
    os.environ[key] = value

import gc, hashlib, importlib.metadata as md, json, socket, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNClassifier

HERE = Path(__file__).resolve().parent
CODEX = HERE.parents[1]
CHECKPOINT = HERE / 'checkpoint' / 'tabpfn-v3-classifier-v3_default.ckpt'
RUNTIME_RECEIPT = HERE / 'runtime-receipt.json'
DESIGN = CODEX / 'experiments' / 'stage2_scaling_20260912' / 'freeze.json'
OUTPUT = HERE / 'benchmark.json'
EXPECTED_CHECKPOINT = 'd0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988'
EXPECTED_DESIGN = '958476a54f1dca9b06695dcad657ee510e075de7b3e29a96843356870d654416'
TRAIN_ROWS, QUERY_ROWS, WIDTH = 1160, 300, 453
CHUNK = 256
SEED = 7329
CEILING = int(3.6 * 1024**3)
PARAMS = dict(
    n_estimators=8,
    auto_scale_n_estimators=False,
    softmax_temperature=0.9,
    balance_probabilities=False,
    average_before_softmax=False,
    device='cuda:0',
    ignore_pretraining_limits=False,
    inference_precision='auto',
    fit_mode='low_memory',
    memory_saving_mode=True,
    random_state=SEED,
    n_preprocessing_jobs=1,
    show_progress_bar=False,
)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def network_guard(event, args):
    if event == 'socket.connect':
        raise RuntimeError('Synthetic TabPFN benchmark forbids outbound network connections')

def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f'write-once benchmark receipt already exists: {OUTPUT}')
    started = time.perf_counter()
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'passed': False,
        'synthetic_only': True,
        'poker_outcomes_read': False,
        'shape': {'kind':'independent_maxima_envelope','source_family':'directed_transfer','training_source_fold':2,'query_source_fold':0,'training_rows':TRAIN_ROWS,'query_rows':QUERY_ROWS,'features':WIDTH},
        'chunks': [256, 44],
        'seed': SEED,
        'parameters': PARAMS,
        'memory_ceiling_bytes': CEILING,
        'sources': {str(Path(__file__).resolve()): sha(Path(__file__).resolve()), str(DESIGN.resolve()): sha(DESIGN), str(RUNTIME_RECEIPT.resolve()): sha(RUNTIME_RECEIPT), str(CHECKPOINT.resolve()): sha(CHECKPOINT)},
    }
    model = None
    try:
        assert md.version('tabpfn') == '8.2.0'
        assert md.version('torch') == '2.11.0+cu128'
        assert sha(CHECKPOINT) == EXPECTED_CHECKPOINT
        assert sha(DESIGN) == EXPECTED_DESIGN
        design = json.loads(DESIGN.read_text(encoding='utf-8'))
        rows = [x for x in design['design'] if x['arm']=='incumbent']
        assert (max(x['train_rows'] for x in rows), max(x['held_rows'] for x in rows), max(len(x['features']) for x in rows)) == (TRAIN_ROWS, QUERY_ROWS, WIDTH)
        assert torch.cuda.is_available()
        device = torch.cuda.get_device_properties(0)
        result['runtime'] = {
            'python': sys.version,
            'tabpfn': md.version('tabpfn'),
            'torch': torch.__version__,
            'torch_cuda_build': torch.version.cuda,
            'cuda_device': device.name,
            'cuda_total_memory_bytes': device.total_memory,
            'cuda_free_before_bytes': torch.cuda.mem_get_info(0)[0],
        }
        sys.addaudithook(network_guard)
        rng = np.random.default_rng(SEED)
        xtrain = rng.normal(size=(TRAIN_ROWS, WIDTH))
        xtest = rng.normal(size=(QUERY_ROWS, WIDTH))
        ytrain = np.tile(np.array([0, 1], dtype=np.int64), TRAIN_ROWS // 2)
        rng.shuffle(ytrain)
        torch.set_num_threads(2)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        model = TabPFNClassifier(model_path=str(CHECKPOINT.resolve()), **PARAMS)
        model.fit(xtrain, ytrain)
        assert model.n_estimators == 8
        assert model.n_estimators_ == 8
        assert len(model.ensemble_configs_) == 8
        assert tuple(model.executor_.X_train.shape) == tuple(xtrain.shape)
        positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
        torch.cuda.synchronize(0)
        inference_started = time.perf_counter()
        parts = []
        for start in range(0, QUERY_ROWS, CHUNK):
            part = model.predict_proba(xtest[start:start+CHUNK])[:, positive_index]
            assert np.isfinite(part).all() and np.all((part >= 0) & (part <= 1))
            parts.append(part)
        torch.cuda.synchronize(0)
        probabilities = np.concatenate(parts)
        allocated = int(torch.cuda.max_memory_allocated(0))
        reserved = int(torch.cuda.max_memory_reserved(0))
        result['runtime'].update({
            'effective_estimators_attribute': int(model.n_estimators_),
            'ensemble_configs_count': len(model.ensemble_configs_),
            'executor_class': type(model.executor_).__name__,
            'retained_context_shape': list(model.executor_.X_train.shape),
            'inference_seconds': time.perf_counter() - inference_started,
            'peak_allocated_bytes': allocated,
            'peak_reserved_bytes': reserved,
            'cuda_free_after_bytes': torch.cuda.mem_get_info(0)[0],
        })
        result['probabilities'] = {
            'count': len(probabilities),
            'minimum': float(probabilities.min()),
            'maximum': float(probabilities.max()),
            'mean': float(probabilities.mean()),
            'sha256': hashlib.sha256(probabilities.tobytes(order='C')).hexdigest(),
        }
        result['checks'] = {
            'checkpoint_digest': True,
            'design_digest': True,
            'exact_shape': True,
            'balanced_synthetic_labels': int(ytrain.sum()) == TRAIN_ROWS // 2,
            'effective_estimators_exactly_8': model.n_estimators_ == 8 and len(model.ensemble_configs_) == 8,
            'all_context_rows_retained': tuple(model.executor_.X_train.shape) == tuple(xtrain.shape),
            'fixed_chunks': [len(x) for x in parts] == [256, 44],
            'finite_probabilities': bool(np.isfinite(probabilities).all()),
            'memory_ceiling': reserved <= CEILING,
        }
        result['passed'] = all(result['checks'].values())
        assert result['passed'], result['checks']
    except BaseException as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        result['seconds'] = time.perf_counter() - started
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if not result['passed']:
        raise RuntimeError(f"Synthetic benchmark failed; see {OUTPUT}")
    print(json.dumps({'passed':True,'seconds':result['seconds'],'runtime':result['runtime'],'checks':result['checks']}, indent=2))

if __name__ == '__main__':
    main()


