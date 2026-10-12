"""Portable, offline-after-checkpoint predictor for the frozen standard 3.5 study."""
import argparse, gc, hashlib, json, os, time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["TABPFN_NO_BROWSER"] = "1"

import numpy as np
import torch
from tabpfn import TabPFNClassifier

HERE = Path(__file__).resolve().parent
CONTEXTS = HERE / "contexts"
OUTPUTS = HERE / "outputs"
MODEL = HERE / "tabpfn-v3.5-20260909_multiclass.safetensors"
EXPECTED = "4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149"
SEED = 35935


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fixed4(probability, classes):
    out = np.zeros((len(probability), 4), dtype=np.float64)
    out[:, np.asarray(classes, dtype=int)] = probability
    assert np.isfinite(out).all() and np.allclose(out.sum(1), 1.0, atol=1e-5)
    return out


def predict(model, features):
    chunks = [model.predict_proba(features[i:i + 64]) for i in range(0, len(features), 64)]
    return fixed4(np.concatenate(chunks), model.classes_)


def fit_and_score(fold, omitted=None, replay=False):
    name = f"fold{fold}" + (f"_omit{omitted}" if omitted else "")
    context = CONTEXTS / f"context_{name}.npz"
    data = np.load(context, allow_pickle=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = TabPFNClassifier(
        model_path=str(MODEL), device="cuda", n_estimators=4,
        auto_scale_n_estimators=False, fit_mode="low_memory",
        memory_saving_mode=True, random_state=SEED, n_preprocessing_jobs=1,
    )
    model.fit(data["X_train"], data["y_train"])
    probability = predict(model, data["X_held"])
    filename = f"replay_{name}.npz" if replay else f"predictions_{name}.npz"
    destination = OUTPUTS / filename
    assert not destination.exists(), f"Refusing to overwrite {destination}"
    np.savez_compressed(destination, probabilities=probability,
                        pair_id=data["pair_id"], classes=np.asarray(model.classes_, dtype=np.int8))
    if omitted is None and not replay:
        prefix = np.load(CONTEXTS / f"context_fold{fold}_prefix60.npz", allow_pickle=False)
        prefix_destination = OUTPUTS / f"predictions_fold{fold}_prefix60.npz"
        assert not prefix_destination.exists()
        np.savez_compressed(prefix_destination, probabilities=predict(model, prefix["X_held"]),
                            pair_id=prefix["pair_id"], classes=np.asarray(model.classes_, dtype=np.int8))
    receipt = {
        "context": name, "replay": replay, "seconds": time.monotonic() - started,
        "context_sha256": sha(context), "checkpoint_sha256": sha(MODEL),
        "prediction_sha256": sha(destination),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "classes": np.asarray(model.classes_, dtype=int).tolist(),
        "seed": SEED, "members": 4, "query_batch": 64,
        "fit_mode": "low_memory", "memory_saving_mode": True,
    }
    (OUTPUTS / (f"replay_runtime_{name}.json" if replay else f"runtime_{name}.json")).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("known", "replay", "lofo"))
    args = parser.parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    assert torch.cuda.is_available()
    assert sha(MODEL) == EXPECTED
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70)
    if args.mode == "known":
        for fold in (0, 1):
            fit_and_score(fold)
    elif args.mode == "replay":
        replay = fit_and_score(0, replay=True)
        original = np.load(OUTPUTS / "predictions_fold0.npz", allow_pickle=False)
        repeated = np.load(replay, allow_pickle=False)
        assert np.array_equal(original["pair_id"], repeated["pair_id"])
        exact = np.array_equal(original["probabilities"], repeated["probabilities"])
        check = {"exact": bool(exact), "max_abs": float(np.max(np.abs(
            original["probabilities"] - repeated["probabilities"]))) }
        (OUTPUTS / "replay_check.json").write_text(json.dumps(check, indent=2) + "\n")
        assert exact, check
    else:
        for fold in (0, 1):
            for omitted in (1, 2, 3):
                fit_and_score(fold, omitted=omitted)


if __name__ == "__main__":
    main()
