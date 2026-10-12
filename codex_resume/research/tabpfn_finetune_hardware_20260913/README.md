# Synthetic TabPFN fine-tuning hardware preflight

`tabpfn_finetune_hardware.py` is intentionally unexecuted.  It performs exactly one deterministic synthetic epoch only when someone runs it later.  It never loads project data, reads credentials, scores a task, downloads dependencies or weights, changes the checkpoint, retries after an OOM, or changes the requested dimensions.

The required runtime root must contain the local `tabpfn-8.2.0-py3-none-any.whl` and `checkpoint/tabpfn-v3-classifier-v3_default.ckpt`.  The runner verifies the wheel source against the installed `tabpfn==8.2.0` source, verifies the checkpoint digest before and after, blocks socket connections, requires a 16 GB RTX 4060 Ti, and writes exactly one `summary.json` into a fresh absolute output directory.

Use the metadata-only control first; it constructs the configured fine-tuner but does not load checkpoint weights or use CUDA:

```powershell
python tabpfn_finetune_hardware.py --root <absolute-runtime-root> --output <absolute-new-output-directory> --dry-run
```

The normal invocation is identical without `--dry-run`.  It uses fixed seed `26091382`, synthetic `float32` arrays of 1160x453 plus a distinct explicit 300x453 validation set, balanced binary labels, `ModelVersion.V3`, learning rate `1e-5`, one epoch, two fine-tuning/validation estimators, eight final estimators, activation checkpointing, and `use_fixed_preprocessing_seed=False`.  Early stopping and scheduler changes are disabled; validation still runs so its clone and final inference are included in the recorded VRAM peak.

`summary.json` retains either success or failure, including CUDA peak allocated/reserved bytes, the 13.5 GiB reserved-memory guard, finite loss/gradient/probability checks, sampled evidence that the actual optimizer step changed parameters, observed estimator attributes after each runtime clone fit, source/package hashes and versions.  Values are only runtime integrity evidence; it emits no real-data metric and should not be used for model selection.
