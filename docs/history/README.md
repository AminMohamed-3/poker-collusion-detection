# Synthetic poker coordination detection

September 10 closeout: [five new evidence submissions](outputs_v14_evidence_portfolio/README.md) are fully validated and ready for the next daily allowance; none is uploaded. [Internal/public comparison](reports/internal-public-20260910/REPORT.md) is descriptive, with no public-score-fitted weights. Work and the goal are paused at the user's request. VM stop remains unconfirmed because account API/SSH access failed; Vast sign-in/manual stop is pending. See the current entry in `../../STATUS.md` and `reports/diverse-next-day/vast-stop-status.json`.

Private working directory for the Kaggle competition:
https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker

This is a synthetic-data classification and evidence-retrieval task. No real players or gambling are involved.

Latest verified public result, September 10 UTC: **v12/v13 conditional models 0.86005**, observed public rank **22/129**, fifteen scored entries, no prize earned. Five controlled submissions improved v10 by .01654; risk and evidence improved separately, behavior added .00072, and wider evidence coverage tied at displayed precision. The leader scores .93720. All full membership validations and immutable-file hashes pass. A complete inference replay reproduces v13 byte-for-byte from frozen models and feature caches. The user-provided GPU instance completed 49 GPU control/research fits plus normal-action reference fits; workers have exited. Pool/family checks, coherent histories and a less-reused 169-pool audit support the changes, but private performance remains unknown. Current notes: `README-v13.md`, `reports/conditional-public-results-20260910.md`, `reports/replay-v13/README.md` and `reports/GPU-RESEARCH-STATE.md`. The baseline material below is historical.

## Approach

1. Inspect the eight provided files and validate joins, labels, and chronological phases.
2. Derive features from gameplay: contributions in big blinds, player-specific aggression, responses to another player's bets, folding/checking behavior, hand strength, and unusual repeated patterns.
3. Hold out entire player pools/tables for validation. Fit using development-period gameplay and explicitly labeled pairs only. Unlisted development pairs are unknown, never assumed negative.
4. Predict pair risk and behavior; rank shared evaluation-period hands as evidence.
5. Validate the exact submission schema, uniqueness, finite scores, legal behavior labels, and phase/player membership of every evidence hand.
6. Save model settings, dependencies, validation results, and reproducible commands. Publish the required code/writeup/case reviews after submissions close if seeking a prize.

Identifiers are used only for joins, grouping, leakage prevention, and output. ID formats, row ordering, and simulator artifacts are excluded from predictive features.

## Current status

Competition enrollment and the eight-file data download are complete. The baseline is trained, and `outputs/submission_baseline.csv` contains predictions for all 112,540 evaluation pairs. It has been submitted to Kaggle: the latest reported public score is **0.61221**, ranked **56 of 104** at that check. Ranking can change. No prize has been awarded and no money has been earned.

Development training uses 1,860 explicitly labeled pairs: 372 confirmed targets and 1,488 confirmed non-targets. The caches contain 223,749 development pair-hands and 9,651,820 evaluation pair-hands. Every computed evaluation shared-hand count matches the supplied count. Each player belongs to exactly one table, so the table holdouts also separate players.

## Local validation

Results in `outputs/validation_metrics.json`:

| Table holdout | Pair average precision | Behavior mean average precision |
| --- | ---: | ---: |
| 0 | 0.9544 | 0.9177 |
| 1 | 0.9548 | 0.9367 |
| 2 | 0.9631 | 0.9051 |

The hand classifier's evidence-versus-confirmed-negative average precision is 0.5325. Evidence MAP@5 is 0.3017 over 98 held-out target pairs **when given the true behavior family**. This is a diagnostic, not the official combined competition score. End-to-end evidence retrieval also depends on predicting the behavior correctly.

Local pair/behavior AP uses scikit-learn's treatment of score ties. The official metric sorts by pair ID and uses stable ordering for tied scores, so the local AP figures do not exactly reproduce its scorer. The local-versus-public gap needs further investigation; these development diagnostics should not be compared directly with the 0.61221 public combined score.

The pair model uses 240 trees; the final hand model uses 175. LightGBM's random seed is 7329, table splits use seed 2901, and training uses four threads with deterministic column-wise execution. Dependencies are recorded in `requirements.lock.txt`. Holdouts are also used for early stopping, so these are development estimates rather than an untouched final test. The model learns the three positive behavior families present in the labels; it does not learn `other_coordination`. No validation result guarantees leaderboard performance or a prize.

## Reproduce the baseline

Run these PowerShell commands from this private working directory. The existing `.venv` has the pinned dependencies installed; no paid compute is used.

```powershell
Set-Location -LiteralPath 'C:\Data\Projects\Money Making\work\poker'
& .\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
& .\.venv\Scripts\python.exe src\features.py development --batch-size 20
& .\.venv\Scripts\python.exe src\features.py evaluation --batch-size 20
& .\.venv\Scripts\python.exe src\features.py development --batch-size 20 --check-cache
& .\.venv\Scripts\python.exe src\features.py evaluation --batch-size 20 --check-cache
& .\.venv\Scripts\python.exe src\train.py train
& .\.venv\Scripts\python.exe src\train.py predict
& .\.venv\Scripts\python.exe src\validation.py outputs\submission_baseline.csv --data-dir data --report outputs\submission_baseline.validation.json
& .\.venv\Scripts\python.exe -m pytest tests\test_cache_protocol.py tests\test_validation.py -q
```

Training and prediction overwrite their local model/prediction outputs. Run both cache checks immediately before using those commands: `train.py` itself does not check manifests. Keep the input files and code unchanged while a build or training run is in progress. A fresh build may produce different row ordering or small numerical differences across environments; the recorded local scores are not a promise of byte-identical retraining.

## Cache compatibility and the one-time adoption

Each `outputs/hands_<phase>/manifest.json` records the SHA-256 hashes of `src/features.py`, `hands.parquet`, `seats.parquet`, `actions.parquet`, and that phase's pair source, together with the Polars version, batch size, exact table allocation, and SHA-256 hashes of the cached batches. New caches record each completed batch atomically. Aggregation reads only the expected batches. `--check-cache` verifies a complete cache and leaves feature and summary files unchanged.

Changed fingerprints, altered/missing recorded batches, extra Parquet files, and untracked legacy batches stop reuse. Restore the matching inputs/code, or move the affected cache directory aside and rebuild it with `features.py`; then retrain and regenerate predictions. A failed or partial build can resume when its recorded batches remain intact. A file written just before an interrupted manifest update is untracked and intentionally requires investigation.

The original caches predate manifests. Their feature arithmetic and source data were independently reviewed: no duplicate pair-hands, correct pair/table coverage, exact evaluation shared-hand counts, and correct previous-aggressor attribution on 5,000 hands. The manifest patch leaves feature arithmetic unchanged. The following explicit one-time commands adopt those verified caches without rebuilding features or summaries; their timestamp, verification note, structural checks, and hashes are retained in each manifest:

```powershell
$cacheReview = 'Independent review confirmed feature arithmetic, pair/table coverage, no duplicate pair-hands, exact evaluation shared-hand counts, and previous-aggressor attribution on 5000 hands; the manifest patch preserves feature arithmetic.'
& .\.venv\Scripts\python.exe src\features.py development --batch-size 20 --adopt-cache --adoption-note $cacheReview
& .\.venv\Scripts\python.exe src\features.py evaluation --batch-size 20 --adopt-cache --adoption-note $cacheReview
```

Adoption checks the full expected batch layout, pair coverage, duplicate pair-hands, and evaluation shared-hand counts. It requires a verification note because those structural checks cannot prove that old feature values were computed by the current formula. Adoption cannot replace an existing manifest or bypass a fingerprint mismatch; do not repeat it for the already adopted caches.

Keep `data/`, models, and detailed solutions private during the competition. The dataset may only be used under its competition rules.
