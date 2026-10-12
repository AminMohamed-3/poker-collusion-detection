# Pretrained tabular-model risk blend (private)

V10 is fully validated and scored: **0.84351**, up **0.03522** over v8 equity at 0.80829. Official submission **56130776** is COMPLETE. It is the current public best, with no prize earned. All 112,540 pairs and 562,700 evidence memberships pass validation. All five daily submission slots were used after this upload.

The frozen TabICL recipe uses 128 numeric gameplay features selected by a LightGBM fitted on the appropriate training partition, two transformed-context ensemble members, seed 7329, float32 CPU inference, and the versioned TabICLv2 checkpoint. Only explicitly labeled pairs are context examples. At final fitting, all 1,860 labeled pairs and the full-data v5 selector are used. No evaluation labels or unknown-pair negative labels are introduced.

The selected candidate is the equal arithmetic mean of TabICL risk and v5 risk. V5 risk is also the risk in v8. Every v8 behavior and evidence cell is retained exactly. The evidence gate remains based on the original v6 risk ranking, not the new blended scores.

| Reused pool fold, shortened histories | V5 pair AP | Blend pair AP | V5 hypothetical 1% AP | Blend hypothetical 1% AP |
|---|---:|---:|---:|---:|
| 0 | .976957 | .982091 | .891427 | .940343 |
| 1 | .975744 | .980305 | .936459 | .952734 |
| 2 | .977005 | .983606 | .850594 | .884408 |

Both standalone TabICL and the blend pass the predeclared known-family and omitted-family criteria. The blend wins the specified selection rule based on mean shortened-history sparse-prior performance. Omitted transfer and isolation benefit; soft-play results are mixed. The 1% prevalence is hypothetical. These reused splits and proxy-unseen families do not establish accuracy on the private test set.

Recipes and immutable results:

- `src/prepare_tabicl_screen.py`, `src/run_tabicl_screen.py`, `reports/tabicl-screen`.
- `src/prepare_tabicl_confirmation.py`, `src/run_tabicl_confirmation.py`, `reports/tabicl-confirmation`.
- `src/prepare_tabicl_candidate.py`, `src/run_tabicl_candidate.py`, `outputs_v10_tabicl`.
- CPU environment package lock: `reports/tabicl-screen/requirements-cpu-lock.txt`.

The official checkpoint is `jingang/TabICL`, revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`, file `tabicl-classifier-v2-20260212.ckpt`, SHA-256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`. The library and checkpoint use BSD-3-Clause; retain the copyright/license notice in any released solution. Sources: [official repository](https://github.com/soda-inria/tabicl), [official checkpoint](https://huggingface.co/jingang/TabICL).

On Windows, prepare inputs with the original `.venv` and run TabICL in the isolated `.venv_tabicl`. The runner stores a fitted model including its checkpoint and context, checks reload predictions exactly, and predicts in fixed 4,096-row chunks. Each chunk receives a model/hash receipt and can be resumed only against the same model and inputs. Full submission validation must follow assembly. Cross-device numerical reproducibility has not been established; do not mix CPU and GPU chunks silently.

The inherited five v8 case reviews remain valid as evidence memberships because this candidate only changes risk. `reports/methodology-v10-draft.md` now documents the pretrained model and its license in 1,243 words; its companion validation JSON verifies all five evidence memberships and the draft/submission hashes. The latest public browser observation places v10 at rank 23 of 129. Keep all competition data and work private until the required post-competition release.
