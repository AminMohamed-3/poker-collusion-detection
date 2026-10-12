"""Bounded, reproducible TabM citation-pilot training runner.

The only supported invocation is ``python -B train.py run``.  Inputs and their
freeze manifest are prepared by the parent workflow; this program never fits a
transformer or reads audit labels.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path

# Must be set before importing torch for deterministic CUDA GEMMs.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, _CUBLAS_WORKSPACE_CONFIG):
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG was already set incompatibly")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_env] = "2"

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
import tabm  # noqa: E402


RUN_NAME = "run"
MAX_RESERVED_BYTES = 2 * 1024**3


class TimeoutExceeded(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(repr(tuple(a.shape)).encode("ascii"))
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_array(name: str, a: np.ndarray) -> None:
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")


def require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


class PlainMLP(nn.Module):
    """The single conventional MLP control with the shared output interface."""

    def __init__(self, n_features: int, n_blocks: int, d_block: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            *sum(
                (
                    [nn.Linear(n_features if i == 0 else d_block, d_block), nn.ReLU(), nn.Dropout(dropout)]
                    for i in range(n_blocks)
                ),
                [],
            ),
            nn.Linear(d_block, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).unsqueeze(1)


def build_model(arm: str, n_features: int, cfg: dict) -> nn.Module:
    if arm == "plain_mlp":
        return PlainMLP(n_features, cfg["n_blocks"], cfg["d_block"], cfg["dropout"])
    if arm == "tabm":
        # num_embeddings=None intentionally: numerical columns go directly to TabM.
        return tabm.TabM.make(
            n_num_features=n_features,
            d_out=1,
            num_embeddings=None,
            n_blocks=cfg["n_blocks"],
            d_block=cfg["d_block"],
            dropout=cfg["dropout"],
            k=cfg["k"],
            arch_type="tabm",
        )
    raise ValueError(f"Unknown arm: {arm}")


def member_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    require(tuple(logits.shape[1:]) == (int(getattr(model, "k", logits.shape[1])), 1),
            f"model returned {tuple(logits.shape)}, expected (batch, k, 1)")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite logits")
    return logits.squeeze(-1)


def weighted_member_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # BCE is first averaged over members for each pair, then pair weights are applied.
    per_pair = F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits), reduction="none").mean(1)
    loss = (per_pair * w).sum() / w.sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    return loss


def as_tensors(x: np.ndarray, y: np.ndarray | None, w: np.ndarray | None):
    tx = torch.from_numpy(np.ascontiguousarray(x))
    ty = None if y is None else torch.from_numpy(np.ascontiguousarray(y))
    tw = None if w is None else torch.from_numpy(np.ascontiguousarray(w))
    return tx, ty, tw


def batches(n: int, batch_size: int, shuffle: bool, generator: torch.Generator | None):
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def check_deadline(started: float, max_seconds: float) -> None:
    if time.monotonic() - started > max_seconds:
        raise TimeoutExceeded(f"max_total_seconds={max_seconds} exceeded")


def check_gpu_budget() -> int:
    reserved = int(torch.cuda.max_memory_reserved())
    if reserved >= MAX_RESERVED_BYTES:
        raise MemoryError(f"GPU reserved memory {reserved} violates < {MAX_RESERVED_BYTES} byte limit")
    return reserved


@torch.inference_mode()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, cfg: dict, device: torch.device, started: float) -> float:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    for ids in batches(len(x), cfg["batch_size"], False, None):
        check_deadline(started, cfg["max_total_seconds"])
        xb, yb, wb = x[ids].to(device), y[ids].to(device), w[ids].to(device)
        logits = member_logits(model, xb)
        # Early stopping targets the arithmetic ensemble probability, not individual members.
        per_pair = F.binary_cross_entropy(torch.sigmoid(logits).mean(1), yb, reduction="none")
        numerator += float((per_pair * wb).sum().item())
        denominator += float(wb.sum().item())
    value = numerator / denominator
    if not np.isfinite(value):
        raise FloatingPointError("non-finite validation loss")
    return value


@torch.inference_mode()
def predict_members(model: nn.Module, x: torch.Tensor, cfg: dict, device: torch.device, started: float) -> np.ndarray:
    model.eval()
    values = []
    for ids in batches(len(x), cfg["batch_size"], False, None):
        check_deadline(started, cfg["max_total_seconds"])
        values.append(torch.sigmoid(member_logits(model, x[ids].to(device))).cpu())
    result = torch.cat(values).numpy().astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite predictions")
    return result


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, cfg: dict, device: torch.device, started: float, epoch_seed: int) -> tuple[float, int]:
    model.train()
    numerator = 0.0
    denominator = 0.0
    updates = 0
    generator = torch.Generator(device="cpu").manual_seed(epoch_seed)
    for ids in batches(len(x), cfg["batch_size"], True, generator):
        check_deadline(started, cfg["max_total_seconds"])
        xb, yb, wb = x[ids].to(device), y[ids].to(device), w[ids].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_member_bce(member_logits(model, xb), yb, wb)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite gradient")
        optimizer.step()
        updates += 1
        numerator += float(loss.detach().item()) * float(wb.sum().item())
        denominator += float(wb.sum().item())
    return numerator / denominator, updates


def env_receipt() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tabm": getattr(tabm, "__version__", "unknown"),
        "device_name": torch.cuda.get_device_name(0),
        "gpu_total_memory": int(torch.cuda.get_device_properties(0).total_memory),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def write_progress(run_dir: Path, arm: str, phase: str, epoch: int, selected_epoch: int | None, trace: list, updates: int, started: float) -> None:
    save_json(run_dir / "progress.json", {
        "arm": arm, "phase": phase, "epoch": epoch, "selected_epoch": selected_epoch,
        "optimizer_updates": updates, "elapsed_seconds": time.monotonic() - started, "trace": trace,
    })


def run_arm(arm: str, arrays: dict, cfg: dict, run_dir: Path, device: torch.device, started: float) -> dict:
    arm_dir = run_dir / arm
    arm_dir.mkdir(exist_ok=False)
    x_fit, y_fit, w_fit = as_tensors(arrays["X_fit"], arrays["y_fit"], arrays["w_fit"])
    x_val, y_val, w_val = as_tensors(arrays["X_val"], arrays["y_val"], arrays["w_val"])
    x_train, y_train, w_train = as_tensors(arrays["X_train"], arrays["y_train"], arrays["w_train"])
    x_audit, _, _ = as_tensors(arrays["X_audit"], None, None)

    seed_everything(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    model = build_model(arm, x_fit.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    trace: list[dict] = []
    best_loss, best_epoch, stale, updates = float("inf"), 0, 0, 0
    best_state = None
    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss, epoch_updates = train_one_epoch(model, optimizer, x_fit, y_fit, w_fit, cfg, device, started, cfg["seed"] + epoch)
        updates += epoch_updates
        val_loss = evaluate(model, x_val, y_val, w_val, cfg, device, started)
        reserved = check_gpu_budget()
        row = {"epoch": epoch, "train_weighted_member_bce": train_loss, "val_weighted_ensemble_bce": val_loss, "gpu_peak_reserved_bytes": reserved}
        trace.append(row)
        if val_loss < best_loss - cfg["min_delta"]:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        write_progress(run_dir, arm, "inner_fit", epoch, best_epoch, trace, updates, started)
        if stale >= cfg["patience"]:
            break
    if best_state is None or best_epoch == 0:
        raise RuntimeError("inner fit selected no epoch")
    torch.save(best_state, arm_dir / "inner_best_state_dict.pt")

    # Rebuild from the same seed and use every training pair for exactly best_epoch epochs.
    seed_everything(cfg["seed"])
    final_model = build_model(arm, x_train.shape[1], cfg).to(device)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    final_trace: list[dict] = []
    final_updates = 0
    for epoch in range(1, best_epoch + 1):
        train_loss, epoch_updates = train_one_epoch(final_model, final_optimizer, x_train, y_train, w_train, cfg, device, started, cfg["seed"] + epoch)
        final_updates += epoch_updates
        reserved = check_gpu_budget()
        final_trace.append({"epoch": epoch, "train_weighted_member_bce": train_loss, "gpu_peak_reserved_bytes": reserved})
        write_progress(run_dir, arm, "final_refit", epoch, best_epoch, final_trace, updates + final_updates, started)

    state_path = arm_dir / "final_state_dict.pt"
    final_state = {key: value.detach().cpu().clone() for key, value in final_model.state_dict().items()}
    torch.save(final_state, state_path)
    members = predict_members(final_model, x_audit, cfg, device, started)
    prediction = members.mean(axis=1, dtype=np.float32)
    prediction_path = arm_dir / "audit_probabilities.npy"
    np.save(prediction_path, prediction)

    # Replay with a fresh instance loaded only from the state file, and demand bitwise equality.
    seed_everything(cfg["seed"])
    replay_model = build_model(arm, x_train.shape[1], cfg).to(device)
    replay_model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    replay_members = predict_members(replay_model, x_audit, cfg, device, started)
    replay_exact = bool(np.array_equal(members, replay_members))
    if not replay_exact:
        raise RuntimeError("fresh-model prediction replay was not exact")

    receipt = {
        "arm": arm,
        "inner_best_epoch": best_epoch,
        "inner_best_val_weighted_ensemble_bce": best_loss,
        "inner_trace": trace,
        "final_refit_trace": final_trace,
        "n_parameters": sum(p.numel() for p in final_model.parameters()),
        "optimizer_updates": {"inner_fit": updates, "final_refit": final_updates, "total": updates + final_updates},
        "audit_prediction_file": prediction_path.name,
        "audit_prediction_sha256": sha256_file(prediction_path),
        "member_probability_dispersion": {
            "mean_std": float(members.std(axis=1).mean()),
            "median_std": float(np.median(members.std(axis=1))),
            "max_std": float(members.std(axis=1).max()),
        },
        "exact_saved_fresh_prediction_replay": replay_exact,
        "state_dicts": {"inner_best": "inner_best_state_dict.pt", "final": state_path.name},
        "gpu_peak_reserved_bytes": check_gpu_budget(),
        "elapsed_seconds": time.monotonic() - started,
    }
    save_json(arm_dir / "arm-receipt.json", receipt)
    return receipt


def load_and_validate() -> tuple[dict, dict, dict]:
    config_path, context_path, freeze_path = HERE / "config.json", HERE / "context.npz", HERE / "run-freeze.json"
    for path in (config_path, context_path, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path.name}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    actual = {"config_sha256": sha256_file(config_path), "context_sha256": sha256_file(context_path), "train_script_sha256": sha256_file(Path(__file__).resolve())}
    require(all(freeze.get(k) == v for k, v in actual.items()), "run-freeze.json hash validation failed")
    context_receipt = json.loads((HERE / "context-receipt.json").read_text(encoding="utf-8"))
    require(context_receipt["context_sha256"] == actual["context_sha256"], "context receipt does not bind this context")
    require(context_receipt["audit_keys_sha256"] == sha256_file(HERE / "audit-keys.parquet"), "audit keys are not bound to this context")
    for name, expected_hash in freeze["vendor"].items():
        require(sha256_file(HERE / "vendor" / name) == expected_hash, "vendored source hash mismatch")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {"n_blocks": 3, "d_block": 256, "dropout": 0.1, "lr": 0.002, "weight_decay": 0.0003, "batch_size": 256, "max_epochs": 100, "patience": 12, "min_delta": 1e-6, "seed": 19143, "k": 32, "max_total_seconds": 900, "device": "cuda", "gpu_reserved_limit_bytes": MAX_RESERVED_BYTES, "arms": ["plain_mlp", "tabm"]}
    require(all(cfg.get(k) == v for k, v in expected.items()), "config.json does not match the locked pilot configuration")
    with np.load(context_path, allow_pickle=False) as z:
        expected_arrays = ("X_fit", "y_fit", "w_fit", "X_val", "y_val", "w_val", "X_train", "y_train", "w_train", "X_audit")
        require(set(z.files) == set(expected_arrays), "context.npz arrays do not match required schema")
        arrays = {key: z[key] for key in expected_arrays}
    for key in ("X_fit", "X_val", "X_train", "X_audit"):
        require(arrays[key].dtype == np.float32 and arrays[key].ndim == 2, f"{key} must be float32 2D")
        finite_array(key, arrays[key])
    require(arrays["X_fit"].shape[1] == arrays["X_val"].shape[1], "X_fit and X_val feature widths differ")
    require(arrays["X_train"].shape[1] == arrays["X_audit"].shape[1], "X_train and X_audit feature widths differ")
    for split in ("fit", "val", "train"):
        x, y, w = arrays[f"X_{split}"], arrays[f"y_{split}"], arrays[f"w_{split}"]
        require(y.dtype == np.float32 and w.dtype == np.float32 and y.ndim == 1 and w.ndim == 1 and len(x) == len(y) == len(w), f"{split} label/weight shape or dtype invalid")
        require(np.isin(y, [0.0, 1.0]).all() and (w > 0).all(), f"{split} labels must be binary and weights positive")
        require(abs(float(w.mean()) - 1.0) <= 1e-5, f"{split} weights are not mean-normalized")
        finite_array(f"y_{split}", y); finite_array(f"w_{split}", w)
    return cfg, arrays, actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run"])
    args = parser.parse_args()
    if args.mode != "run":
        raise AssertionError("unreachable")
    run_dir = HERE / RUN_NAME
    # Any prior final receipt is a completed/attempted immutable run, never overwrite it.
    if (run_dir / "receipt.json").exists() or (run_dir / "failure.json").exists():
        raise FileExistsError("refusing to overwrite final outputs in run/")
    run_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        cfg, arrays, input_hashes = load_and_validate()
        require(torch.cuda.is_available(), "locked configuration requires CUDA")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.95, (MAX_RESERVED_BYTES - 64 * 1024**2) / total), 0)
        seed_everything(cfg["seed"])
        device = torch.device("cuda")
        array_hashes = {key: sha256_array(value) for key, value in arrays.items()}
        save_json(run_dir / "input-receipt.json", {"input_file_hashes": input_hashes, "array_hashes": array_hashes, "environment": env_receipt()})
        arms = [run_arm(arm, arrays, cfg, run_dir, device, started) for arm in cfg["arms"]]
        save_json(run_dir / "receipt.json", {"status": "complete", "input_file_hashes": input_hashes, "array_hashes": array_hashes, "arms": arms, "environment": env_receipt(), "total_elapsed_seconds": time.monotonic() - started})
    except Exception as exc:
        save_json(run_dir / "failure.json", {"status": "failed", "exception_type": type(exc).__name__, "message": str(exc), "elapsed_seconds": time.monotonic() - started, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
