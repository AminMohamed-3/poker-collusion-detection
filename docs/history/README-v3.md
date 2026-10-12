# Private evidence ranking model

This candidate keeps every `pair_id`, `risk_score`, and `predicted_behavior` cell from `outputs_v2/submission_advanced_robust.csv` exactly, replacing only the five ranked evidence hands. That source entry scored **0.77586**, rank **35/104** at observation. The user submitted this third entry; Kaggle displayed Success with **0.79021**, rank **33/104**, an improvement of 0.01435. Because the other scoring components are unchanged, the displayed difference implies approximately 0.07175 improvement in public evidence MAP@5 (scores are rounded). Its SHA-256 is `ad92ddcbef59cf3a8d67cdd98cd191da9e77b6b4e7fcaca92f92c47b6d3178a3`.

The hand ranker uses 163 gameplay features, 56 sample-standardized event features relative to the same pair's observable history, and three behavior-family indicator features. It uses the same board-prefix/card/action features as v2. Context denominators have a 0.001 floor and z-scores are clipped to [-20,20]. Identifier values are not predictors; independent seeded uniform values resolve score ties.

Training includes all **372 explicitly confirmed target pairs**, **45,129 shared hands**, and **1,817 listed evidence hands**. LambdaRank learns a preference for listed evidence over unannotated hands within that same target pair. The 43,312 unannotated hands have relevance zero for this noisy ranking preference; they are not asserted to be normal or non-coordinated. Unlisted development pairs are excluded. The ranker has 238 boosting rounds, 15 leaves, learning rate 0.03, minimum child samples 20, column fraction 0.8, L2 penalty 5, and seed 7329. It uses two LightGBM threads and at most four Polars threads.

Complete the baseline and v2 steps first, as described in their READMEs. Use the existing environment pinned by `requirements-v2.lock.txt`. Then run from this directory in PowerShell:

```powershell
$env:POLARS_MAX_THREADS='4'
& .\.venv\Scripts\python.exe -X utf8 src\rank_evidence.py train
& .\.venv\Scripts\python.exe -X utf8 src\rank_evidence.py predict
& .\.venv\Scripts\python.exe -X utf8 src\validation.py outputs_v3\submission_evidence_ranker.csv --data-dir data --report outputs_v3\submission_evidence_ranker.validation.json
```

All new outputs live in `outputs_v3/`. The code verifies source/model/input hashes, refuses stale or untracked caches, checks complete pair histories, preserves the source risk/class cells as strings, and checks them again after writing the CSV. A valid matching fit or prediction can be resumed without replacing completed artifacts. Full membership validation is still required before submitting the resulting file.

`outputs_v3/fit.json` contains the final all-target fit provenance. Model SHA-256: `6d496a022cc2a050d2d786203f9ec988aa54db0c77216e108b077a395f58d55d`.

The fixed evidence classifier/ranker comparisons on three pool holdouts gave true-family MAP@5 of **0.380969/0.468359**, **0.324694/0.455417**, and **0.370266/0.459479**. Each disclosed family improved in the two additional splits. Fold0 predicted-family MAP@5 improved from **0.374337 to 0.468359**, using saved held-out pair predictions. The final all-target fit has no independent validation score. These overlapping development holdouts were reused for model selection and do not guarantee leaderboard gains. The fourth undisclosed coordination family remains a limitation.

Experiments, independent checks, and limitations are in `reports/evidence-rank-experiment/` and `reports/evidence-rank-review.md`. Keep source data, detailed methods, and this draft private during competition. Update the final selected model's writeup and five case reviews before the required post-competition publication deadline. No income or prize has been earned.
