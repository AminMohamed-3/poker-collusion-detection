# Private novelty-risk alternative

This is a controlled fourth candidate. It retains the submitted v3 evidence and behavior labels and changes only pair risk. V3 scored **0.79021**, rank **33/104** at observation. The user submitted the novelty candidate; it scored **0.78569**, a decrease of 0.00452. Retain v3 as the current best. This public test does not support adopting the novelty floor.

An IsolationForest with 200 trees, max_samples=256, seed=7329, and two threads learns from all **1,488 explicitly confirmed non-target pairs**, using the original 653 v2 pair features. No unlisted pair is labeled negative. The empirical percentile of each evaluation anomaly score is measured against training-negative scores. The fixed recipe is:

`new_risk = max(robust_risk, 0.5 * clip((novelty_percentile - 0.95) / 0.05, 0, 1))`.

This raises **7,347 of 112,540 risks** and preserves all **1,035 original risks at least 0.5**. Percentiles and resulting scores are ranking heuristics, not calibrated coordination probabilities. The three fixed development splits showed gains when transfer or soft play was omitted from supervised training, but losses for known-family ranking and omitted isolation. These repeated tradeoffs justify an exploratory alternative, not a claim that it is better. The actual fourth hidden mechanism and its prevalence are unknown.

Use the existing v2 environment and completed v3 entry. From this directory:

```powershell
& .\.venv\Scripts\python.exe -X utf8 src\prepare_novelty_candidate.py
& .\.venv\Scripts\python.exe -X utf8 src\assemble_novelty_entry.py
& .\.venv\Scripts\python.exe -X utf8 src\validation.py outputs_novelty\submission_novelty_floor.csv --data-dir data --report outputs_novelty\submission_novelty_floor.validation.json
```

The scripts preserve completed original artifacts and may refuse to overwrite existing candidate outputs. Keep them for review instead of deleting them to bypass a provenance mismatch.

The source-entry, candidate, and assembly hashes are in `outputs_novelty/submission_novelty_floor.provenance.json`; the model and negative-only fit record are in `outputs_novelty/`. Candidate CSV SHA-256: `023a82d384a4089df05e3c33388ec981f2921639b224866baef911623e81c6ed`. Full validation passed for all 112,540 rows and 562,700 evidence references. No prize or income has been earned.
