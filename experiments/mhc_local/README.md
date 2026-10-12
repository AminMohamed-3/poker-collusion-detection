# Tiny local V4.1-inspired mHC experiment

This experiment trains new weights on our poker development data. It tests whether mHC helps a small classifier; it does not reproduce DeepSeek's language capabilities or constitute a Kaggle submission. **All 30 predeclared fits are complete and backed up locally. Neither mHC variant is admitted:** both passed the initial blend screen, but their worst omitted-family sparse AP changes (−.2044 and −.2161) breached the −.05 floor. See [initial comparison](RESULTS.md) and [omitted-family confirmation](OMITTED.md). The separate data-augmentation experiment does not change this outcome.

## Research and implementation scope

DeepSeek announced V4.1 Flash on September 10, 2026. Its official implementation and technical report were retrieved at revision `dba1be0a40aa45a94ad051997016db3960a90277`; the downloaded references and MIT notice are in `references/`, with hashes in `plan.json`.

The official architecture combines a causal encoder and decoder, shared decoder global KV, compressed sparse attention, MoE, and Single-Pass mHC. Its 40 layers comprise 20 encoder and 20 decoder layers. Single-Pass changes the dependency between successive residual sublayers: the current function reads the previous sublayer's input-mixing coefficients, while current residual/output coefficients apply to the current update. This permits more efficient fused implementation. The tiny model preserves this dependency; it does not implement the production fused kernel. [Official repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277), [release log](https://api-docs.deepseek.com/updates/).

mHC maintains several residual streams. A learned input mixture feeds each attention/FFN function; another mixture writes its output back. Sinkhorn normalization makes the residual mixing matrix approximately nonnegative with unit row and column sums. An exact doubly stochastic residual matrix preserves stream means and cannot increase the residual-only Euclidean norm. This does **not** bound the complete nonlinear network's Jacobian or guarantee better accuracy. The original paper's language-model evidence is at a vastly larger scale than this test. [mHC paper](https://arxiv.org/html/2512.24880v2).

| Component | Local implementation |
|---|---|
| Causal encoder/decoder | Two encoder layers, two decoder layers; width 48, four heads |
| Decoder global memory | Project once from final causal encoder states; reuse in the second decoder layer |
| Attention | Dense causal global memory plus an eight-token local window |
| MoE | One shared expert, four routed experts, top two selected; dense evaluation of tiny experts |
| mHC | Four streams, 20 Sinkhorn iterations, epsilon 1e-6; official matrix orientation and coefficient transforms |
| Single-Pass | Previous sublayer's pre-mix; first pre-mix selects stream zero |
| Inputs | 897 existing gameplay features, fixed groups into 24 tokens plus a readout token |
| Outputs | Four poker behavior classes; suspicious risk is one minus probability of none |

This adaptation omits CSA2 compression/indexing, the Full/Reindex/Reuse schedule, FP4 caches, Engram, vision, DSpark, rotary positions, and custom kernels. Learned feature positions are used instead. Feature tokens are not chronological hands or language tokens. Normalization, initialization, dropout, classifier head, and the tiny balancing schedule are our experimental choices, not a reconstruction of DeepSeek's training recipe. The residual prior favors the identity, with diagonal logits 4 and off-diagonal logits 0. The standard-mHC variant tests the mixing mechanism with this same adapter and final collapse, not a complete V4 architecture.

## Frozen comparison

`plan.json` was written before poker model fitting. It hashes the model, training, preparation, evaluation, and unit-test sources. The trainer verifies those hashes and the data manifest at launch.

| Variant | Parameters | Purpose |
|---|---:|---|
| `residual` | 239,524 | Same backbone with ordinary residual additions |
| `residual_budget` | 276,964 | Wider FFNs to match mHC's total parameter budget within 1% |
| `mhc` | 276,604 | Current-sublayer input mixture |
| `single_pass` | 276,604 | V4.1-style delayed input mixture |

Each variant uses seeds 7329, 39117, and 55193, 20 epochs, batch size 64, FP32, AdamW, and the same fixed schedule. The final checkpoint averages trainable weights over the last four epochs. There is no best-seed or validation-selected checkpoint. Matching seeds match common backbone weights between ordinary residual, mHC, and Single-Pass; the wider baseline necessarily has different FFN tensors. Architecture-specific random operations mean dropout draws need not match across variants.

Training contains 1,403 distinct pairs from 297 player pools (4,209 full/70%/40% views). The screen contains 457 distinct pairs from 100 separate pools. The existing feature pipeline is audited for pool/player exclusion, and its hashes are rechecked. Scaling uses training features only. The trainer's archive contains no held labels, and it never opens held metadata.

The primary architecture comparison is Single-Pass against the parameter-matched residual baseline: average paired log-loss improvement at least .005, with Pair AP delta at least -.002 across three seeds and two histories. All twelve outcomes are retained. These development pools have been reused; this is exploratory evidence, not an untouched generalization estimate. Seeds and history views are not independent samples.

The separate Kaggle screen evaluates a predeclared 50/50 risk blend with the incumbent, retaining its behavior/evidence. Only passing mHC variants may advance to omitted-family checks. Production requires further known-pool, omitted-family, history, and audit confirmation. Existing submission files are protected by hashes. A successful architecture comparison alone is insufficient for admission.

## Reproduction

The frozen fold0 comparison completed on the user-supplied RTX 3080 20 GB VM, using PyTorch 2.11.0+cu128. Three seeds ran concurrently. The entire twelve-fit comparison was restarted on that same environment after migration; earlier local fits are preserved and excluded from scoring. `remote-results.tar.gz` and its member manifest back up all twelve remote models and receipts locally. The following commands also support local reproduction with the isolated CUDA environment `work/poker/.venv_mhc`.

```powershell
.\work\poker\.venv_mhc\Scripts\python.exe work/poker/experiments/mhc_local/test_model.py
.\work\poker\.venv_mhc\Scripts\python.exe -u work/poker/experiments/mhc_local/run_local.py
.\work\poker\.venv\Scripts\python.exe work/poker/experiments/mhc_local/evaluate.py --job fold0
```

Completed fits contain portable `model.pt` checkpoints with normalization statistics, probabilities, and receipts recording runtime, hashes, exact reload verification, and stability diagnostics. The queue skips only complete hash-verified fits. Evaluation is write-once; reruns should use a separate experiment directory rather than overwrite evidence. The remote training packages contain training labels and unlabelled held features; held labels remain on the local machine. No data or weights were published.
