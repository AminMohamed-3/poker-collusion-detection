# Revision 3: explicit 3/3/8 ensemble counts; original 2/2/8 setup failure preserved.
"""One-epoch synthetic-only TabPFN-3 fine-tuning hardware preflight.

This program deliberately has no data-loading, scoring, credential, download, retry,
or out-of-memory recovery path.  It is a single, reproducible memory gate for a
future real-label study; do not use its output as model-quality evidence.
"""
from __future__ import annotations

# Set before importing TabPFN/Hugging Face code.  The audit hook installed below is
# the second, runtime-level guard against accidental network access.
import os

for _name in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ.update(
    {
        "TABPFN_DISABLE_TELEMETRY": "1",
        "TABPFN_NO_BROWSER": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
)

import argparse
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import socket
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

EXPECTED_TABPFN_VERSION = "8.2.0"
EXPECTED_WHEEL_SHA256 = "c9432bd301038764db210723e1a49569c4285310a2ecbecd1972241c9d6df290"
EXPECTED_CHECKPOINT_SHA256 = "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"
CHECKPOINT_NAME = "tabpfn-v3-classifier-v3_default.ckpt"
WHEEL_NAME = "tabpfn-8.2.0-py3-none-any.whl"
SYNTHETIC_SEED = 26091382
TRAIN_SHAPE = (1160, 453)
VALIDATION_SHAPE = (300, 453)
MEMORY_GUARD_BYTES = int(13.5 * 1024**3)
MINIMUM_GPU_BYTES = 15 * 1024**3  # A 16 GB card reports slightly less than 16 GiB.
SAMPLE_VALUES_PER_PARAMETER = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"write-once receipt already exists: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_value(value: Any) -> Any:
    """Keep observational fields JSON-safe without silently discarding them."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return repr(value)


def disable_network() -> None:
    """Reject every later socket connection, including a missing-model fallback."""
    def audit(event: str, _arguments: tuple[Any, ...]) -> None:
        if event == "socket.connect":
            raise RuntimeError("Network access is disabled for this synthetic preflight")

    sys.addaudithook(audit)


def require_fresh_absolute_output(output: Path) -> Path:
    if not output.is_absolute():
        raise ValueError("--output must be an absolute, previously nonexistent directory")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"--output must be write-once and nonexistent: {output}")
    output.mkdir(parents=True)
    return output


def resolve_assets(root: Path, checkpoint_arg: Path | None) -> dict[str, Path]:
    if not root.is_absolute():
        raise ValueError("--root must be an absolute runtime directory")
    root = root.resolve()
    wheel = root / WHEEL_NAME
    checkpoint = checkpoint_arg.resolve() if checkpoint_arg is not None else root / "checkpoint" / CHECKPOINT_NAME
    if not root.is_dir():
        raise FileNotFoundError(f"runtime root does not exist: {root}")
    if not wheel.is_file():
        raise FileNotFoundError(f"pinned local wheel not found: {wheel}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"pinned v3 checkpoint not found: {checkpoint}")
    return {"root": root, "wheel": wheel, "checkpoint": checkpoint}


def verify_wheel_sources(wheel: Path) -> dict[str, str]:
    expected_entries = (
        "tabpfn/finetuning/finetuned_base.py",
        "tabpfn/finetuning/finetuned_classifier.py",
        "tabpfn/constants.py",
    )
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = set(expected_entries) - names
        if missing:
            raise RuntimeError(f"pinned wheel lacks expected source entries: {sorted(missing)}")
        return {entry: hashlib.sha256(archive.read(entry)).hexdigest() for entry in expected_entries}


def installed_source_sha256(path: str | Path | None) -> str:
    if path is None:
        raise RuntimeError("could not locate installed TabPFN source")
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"installed TabPFN source missing: {source}")
    return sha256_file(source)


def observed_estimators(estimator: Any) -> dict[str, Any]:
    """Read public/runtime attributes; this is evidence, not a constructor assertion."""
    result: dict[str, Any] = {"estimator_type": type(estimator).__name__}
    for name in ("n_estimators", "n_estimators_", "fit_mode"):
        if hasattr(estimator, name):
            result[name] = json_value(getattr(estimator, name))
    if hasattr(estimator, "ensemble_configs_"):
        configs = getattr(estimator, "ensemble_configs_")
        try:
            result["ensemble_configs_length"] = len(configs)
        except TypeError:
            result["ensemble_configs_length"] = None
    return result


def make_synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SYNTHETIC_SEED)
    X_train = np.ascontiguousarray(rng.standard_normal(TRAIN_SHAPE, dtype=np.float32))
    X_val = np.ascontiguousarray(rng.standard_normal(VALIDATION_SHAPE, dtype=np.float32))
    # Balanced, deterministic binary labels independent of feature values.
    y_train = np.arange(TRAIN_SHAPE[0], dtype=np.int64) % 2
    y_val = np.arange(VALIDATION_SHAPE[0], dtype=np.int64) % 2
    rng.shuffle(y_train)
    rng.shuffle(y_val)
    assert X_train.dtype == np.float32 and X_val.dtype == np.float32
    assert np.isfinite(X_train).all() and np.isfinite(X_val).all()
    assert np.array_equal(np.unique(y_train), [0, 1]) and np.array_equal(np.unique(y_val), [0, 1])
    return X_train, y_train, X_val, y_val


def make_gradient_hook(torch: Any, trace: dict[str, Any]) -> Callable[[Any], Any]:
    """Wrap the actual optimizer step selected by TabPFN's pinned training loop."""
    def decorate(optimizer: Any) -> Any:
        original_step = optimizer.step

        def traced_step(*args: Any, **kwargs: Any) -> Any:
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            if not gradients:
                raise RuntimeError("actual optimizer.step reached with no gradients")
            squared_norm = torch.zeros((), dtype=torch.float64, device=gradients[0].device)
            gradients_finite = True
            parameter_samples: list[Any] = []
            for parameter in parameters:
                if parameter.grad is not None:
                    gradient = parameter.grad.detach()
                    gradients_finite = gradients_finite and bool(torch.isfinite(gradient).all().item())
                    squared_norm += gradient.detach().to(torch.float64).square().sum()
                parameter_samples.append(
                    parameter.detach().reshape(-1)[:SAMPLE_VALUES_PER_PARAMETER].to("cpu", dtype=torch.float32).clone()
                )
            trace["optimizer_step_calls"] = int(trace.get("optimizer_step_calls", 0)) + 1
            trace["gradient_tensor_count"] = len(gradients)
            trace["gradient_finite_before_step"] = gradients_finite
            trace["gradient_l2_norm_before_step"] = float(torch.sqrt(squared_norm).item())
            result = original_step(*args, **kwargs)
            finite_parameters = True
            changed_parameter_tensors = 0
            for parameter, before in zip(parameters, parameter_samples, strict=True):
                after = parameter.detach()
                finite_parameters = finite_parameters and bool(torch.isfinite(after).all().item())
                sampled_after = after.reshape(-1)[:SAMPLE_VALUES_PER_PARAMETER].to("cpu", dtype=torch.float32)
                if not torch.equal(before, sampled_after):
                    changed_parameter_tensors += 1
            trace["parameters_finite_after_step"] = finite_parameters
            trace["changed_parameter_tensors_by_sample"] = changed_parameter_tensors
            trace["parameter_tensors_examined"] = len(parameters)
            if not gradients_finite or not finite_parameters or changed_parameter_tensors == 0:
                raise RuntimeError("optimizer-step gradient/parameter integrity check failed")
            return result

        optimizer.step = traced_step
        return optimizer

    return decorate


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = require_fresh_absolute_output(args.output)
    started = time.time()
    receipt: dict[str, Any] = {
        "purpose": "synthetic-only one-epoch TabPFN-3 fine-tuning hardware preflight",
        "status": "started",
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "synthetic_only": True,
        "synthetic_seed": SYNTHETIC_SEED,
        "train_shape": list(TRAIN_SHAPE),
        "validation_shape": list(VALIDATION_SHAPE),
        "memory_guard_bytes": MEMORY_GUARD_BYTES,
        "memory_guard_gib": 13.5,
        "network_disabled": True,
        "mode": "metadata-only-dry-run" if args.dry_run else "one-epoch-finetune",
    }
    try:
        assets = resolve_assets(args.root, args.checkpoint)
        receipt.update(
            runtime_root=str(assets["root"]),
            local_wheel=str(assets["wheel"]),
            local_wheel_sha256=sha256_file(assets["wheel"]),
            checkpoint_path=str(assets["checkpoint"]),
            original_checkpoint_sha256_before=sha256_file(assets["checkpoint"]),
            expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            wheel_source_sha256=verify_wheel_sources(assets["wheel"]),
        )
        if receipt["original_checkpoint_sha256_before"] != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("checkpoint SHA-256 does not match the pinned v3 checkpoint")

        if receipt["local_wheel_sha256"] != EXPECTED_WHEEL_SHA256:
            raise RuntimeError("Wheel is not the verified PyPI8.2.0 artifact")
        disable_network()
        import torch
        import tabpfn
        import tabpfn.finetuning.finetuned_base as finetuned_base_module
        import tabpfn.finetuning.finetuned_classifier as finetuned_classifier_module
        from tabpfn.constants import ModelVersion
        from tabpfn.finetuning import FinetunedTabPFNClassifier

        package_versions = {name: metadata.version(name) for name in ("tabpfn", "torch", "numpy", "scikit-learn")}
        expected_versions = json.loads((assets["root"] / "runtime-receipt.json").read_text())["required_versions"]
        all_versions = {k: metadata.version(k) for k in expected_versions}
        if all_versions != expected_versions:
            raise RuntimeError(f"Pinned runtime version drift: {all_versions}")
        receipt["all_pinned_versions"] = all_versions
        receipt["package_versions"] = package_versions
        receipt["python"] = sys.version
        receipt["platform"] = platform.platform()
        receipt["tabpfn_module_path"] = str(Path(tabpfn.__file__).resolve())
        if package_versions["tabpfn"] != EXPECTED_TABPFN_VERSION:
            raise RuntimeError(f"tabpfn must be {EXPECTED_TABPFN_VERSION}, got {package_versions['tabpfn']}")
        installed_sources = {
            "tabpfn/finetuning/finetuned_base.py": installed_source_sha256(inspect.getsourcefile(finetuned_base_module)),
            "tabpfn/finetuning/finetuned_classifier.py": installed_source_sha256(inspect.getsourcefile(finetuned_classifier_module)),
            "tabpfn/constants.py": installed_source_sha256(inspect.getsourcefile(sys.modules["tabpfn.constants"])),
        }
        receipt["installed_source_sha256"] = installed_sources
        for entry, digest in installed_sources.items():
            if digest != receipt["wheel_source_sha256"][entry]:
                raise RuntimeError(f"installed runtime source does not match the supplied pinned wheel: {entry}")

        classifier_kwargs = {
            "model_path": str(assets["checkpoint"]),
            "show_progress_bar": False,
        }
        constructor_kwargs = {
            "device": "cuda:0",
            "epochs": 1,
            "random_state": SYNTHETIC_SEED,
            "learning_rate": 1e-5,
            "validation_split_ratio": 0.0,
            "early_stopping": False,
            "use_lr_scheduler": False,
            "n_estimators_finetune": 3,
            "n_estimators_validation": 3,
            "n_estimators_final_inference": 8,
            "use_activation_checkpointing": True,
            "use_fixed_preprocessing_seed": False,
            "save_checkpoint_interval": None,
            "eval_metric": "log_loss",
            "model_version": ModelVersion.V3,
            "extra_classifier_kwargs": classifier_kwargs,
        }
        receipt["requested_configuration"] = {
            **{name: json_value(value) for name, value in constructor_kwargs.items() if name not in {"model_version", "extra_classifier_kwargs"}},
            "model_version": str(ModelVersion.V3),
            "extra_classifier_kwargs": classifier_kwargs,
            "explicit_validation_required": True,
            "real_data_loaded": False,
            "output_dir_passed_to_fit": None,
        }
        classifier = FinetunedTabPFNClassifier(**constructor_kwargs)
        receipt["constructor_only_checks"] = {
            "constructed_without_loading_weights": True,
            "model_path_exists": Path(classifier.extra_classifier_kwargs["model_path"]).is_file(),
            "model_version_observed": str(classifier.finetune_model_version),
            "constructor_params": json_value(classifier.get_params(deep=False)),
        }
        if classifier.finetune_model_version != ModelVersion.V3:
            raise RuntimeError("constructor did not retain ModelVersion.V3")
        if args.dry_run:
            receipt["original_checkpoint_sha256_after"] = sha256_file(assets["checkpoint"])
            if receipt["original_checkpoint_sha256_after"] != receipt["original_checkpoint_sha256_before"]:
                raise RuntimeError("checkpoint changed during metadata-only dry run")
            receipt["status"] = "dry-run-passed"
            return receipt

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; synthetic preflight will not fall back to CPU")
        properties = torch.cuda.get_device_properties(0)
        if "4060 Ti" not in properties.name or properties.total_memory < MINIMUM_GPU_BYTES:
            raise RuntimeError(f"requires the 16 GB RTX 4060 Ti target, observed {properties.name} with {properties.total_memory} bytes")
        free, total = torch.cuda.mem_get_info(0)
        if free < 14 * 1024**3:
            raise RuntimeError("GPU is occupied; do not run concurrently")
        torch.set_num_threads(2)
        torch.manual_seed(SYNTHETIC_SEED)
        torch.cuda.manual_seed_all(SYNTHETIC_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        receipt["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "cuda_build": torch.version.cuda,
            "capability": list(torch.cuda.get_device_capability(0)),
            "deterministic_seeded": True,
        }

        X_train, y_train, X_val, y_val = make_synthetic_data()
        receipt["synthetic_data_checks"] = {
            "float32_features": True,
            "finite_features": True,
            "train_binary_class_counts": np.bincount(y_train, minlength=2).tolist(),
            "validation_binary_class_counts": np.bincount(y_val, minlength=2).tolist(),
            "explicit_validation_distinct_object": X_train is not X_val and y_train is not y_val,
        }

        gradient_trace: dict[str, Any] = {}
        loss_trace: dict[str, Any] = {"loss_values_finite": True, "loss_calls": 0}
        original_optimizer_factory = finetuned_base_module.get_and_init_optimizer
        original_clone = finetuned_classifier_module.clone_model_for_evaluation
        original_forward = classifier._forward_with_loss
        clone_observations: list[dict[str, Any]] = []
        receipt["gradient_path"] = gradient_trace
        receipt["loss_check"] = loss_trace
        receipt["clone_observations"] = clone_observations

        def optimizer_factory(*factory_args: Any, **factory_kwargs: Any) -> Any:
            return make_gradient_hook(torch, gradient_trace)(original_optimizer_factory(*factory_args, **factory_kwargs))

        def clone_hook(*clone_args: Any, **clone_kwargs: Any) -> Any:
            clone = original_clone(*clone_args, **clone_kwargs)
            expected = clone_args[1].get("n_estimators") if len(clone_args) > 1 and isinstance(clone_args[1], dict) else None
            original_fit = clone.fit

            def traced_clone_fit(*fit_args: Any, **fit_kwargs: Any) -> Any:
                value = original_fit(*fit_args, **fit_kwargs)
                clone_observations.append({"requested_n_estimators": expected, "observed_after_fit": observed_estimators(clone)})
                return value

            clone.fit = traced_clone_fit
            return clone

        def forward_hook(batch: Any) -> Any:
            loss = original_forward(batch)
            finite = bool(torch.isfinite(loss.detach()).all().item())
            loss_trace["loss_calls"] += 1
            loss_trace.setdefault("values", []).append(float(loss.detach().item()))
            loss_trace["loss_values_finite"] = bool(loss_trace["loss_values_finite"] and finite)
            return loss

        # These narrow temporary hooks observe the library's real optimizer and clones.
        # They are restored even when TabPFN raises, and never alter the update itself.
        finetuned_base_module.get_and_init_optimizer = optimizer_factory
        finetuned_classifier_module.clone_model_for_evaluation = clone_hook
        classifier._forward_with_loss = forward_hook
        try:
            classifier.fit(X_train, y_train, X_val=X_val, y_val=y_val, output_dir=None)
            probabilities = classifier.predict_proba(X_val)
        finally:
            finetuned_base_module.get_and_init_optimizer = original_optimizer_factory
            finetuned_classifier_module.clone_model_for_evaluation = original_clone
            classifier._forward_with_loss = original_forward

        torch.cuda.synchronize(0)
        probabilities_finite = bool(np.isfinite(probabilities).all())
        probabilities_in_range = bool(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        probabilities_rows_sum_to_one = bool(np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6))
        receipt["gradient_path"] = gradient_trace
        receipt["loss_check"] = loss_trace
        receipt["probability_check"] = {
            "shape": list(probabilities.shape),
            "finite": probabilities_finite,
            "in_closed_unit_interval": probabilities_in_range,
            "rows_sum_to_one": probabilities_rows_sum_to_one,
        }
        receipt["effective_estimators_observed"] = {
            "training_estimator_after_fit": observed_estimators(classifier.finetuned_estimator_),
            "validation_and_final_clone_fits": clone_observations,
            "final_inference_estimator_after_fit": observed_estimators(classifier.finetuned_inference_classifier_),
        }
        for observation in clone_observations:
            requested = observation["requested_n_estimators"]
            actual = observation["observed_after_fit"]
            if requested not in (3,8) or actual.get("n_estimators_") != requested or actual.get("ensemble_configs_length") != requested:
                raise RuntimeError(f"Unexpected clone estimator count: {observation}")
        if not {3,8}.issubset({o["requested_n_estimators"] for o in clone_observations}):
            raise RuntimeError("Both validation and final inference clone fits must be observed")
        if classifier.finetuned_estimator_.n_estimators != 3:
            raise RuntimeError("Training estimator count differs from three")
        receipt["peak_vram_bytes"] = {
            "allocated": torch.cuda.max_memory_allocated(0),
            "reserved": torch.cuda.max_memory_reserved(0),
            "measurement_scope": "after optimizer creation/step, explicit-validation clones, and final inference predict_proba",
        }
        receipt["memory_guard_pass"] = receipt["peak_vram_bytes"]["reserved"] <= MEMORY_GUARD_BYTES
        receipt["original_checkpoint_sha256_after"] = sha256_file(assets["checkpoint"])
        if receipt["original_checkpoint_sha256_after"] != receipt["original_checkpoint_sha256_before"]:
            raise RuntimeError("original checkpoint hash changed during preflight")
        if not all((
            loss_trace["loss_calls"] > 0,
            loss_trace["loss_values_finite"],
            gradient_trace.get("optimizer_step_calls", 0) > 0,
            gradient_trace.get("gradient_finite_before_step", False),
            gradient_trace.get("parameters_finite_after_step", False),
            gradient_trace.get("changed_parameter_tensors_by_sample", 0) > 0,
            probabilities_finite,
            probabilities_in_range,
            probabilities_rows_sum_to_one,
            receipt["memory_guard_pass"],
        )):
            raise RuntimeError("synthetic preflight integrity or 13.5 GiB memory guard failed")
        receipt["status"] = "passed"
        return receipt
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        receipt["traceback"] = traceback.format_exc()
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            receipt["final_peak_vram_bytes"] = {"allocated":torch.cuda.max_memory_allocated(0),"reserved":torch.cuda.max_memory_reserved(0)}
        if "assets" in locals() and assets["checkpoint"].is_file():
            receipt["original_checkpoint_sha256_finally"] = sha256_file(assets["checkpoint"])
        receipt["elapsed_seconds"] = time.time() - started
        # Preserve the only receipt on every path, including an OOM.  There is no retry.
        if output.exists() and not (output / "summary.json").exists():
            write_once(output / "summary.json", receipt)
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="absolute local runtime root containing the pinned wheel and checkpoint/")
    parser.add_argument("--checkpoint", type=Path, help="optional absolute checkpoint path; defaults to --root/checkpoint/" + CHECKPOINT_NAME)
    parser.add_argument("--output", type=Path, required=True, help="absolute, fresh directory for the single write-once summary receipt")
    parser.add_argument("--dry-run", action="store_true", help="verify wheel/runtime/constructor paths without loading checkpoint weights or using CUDA")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    try:
        result = run(parsed)
    except BaseException:
        raise
    else:
        print(json.dumps({"status": result["status"], "summary": str(parsed.output.resolve() / "summary.json")}), flush=True)
