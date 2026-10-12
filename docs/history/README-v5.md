# Repeated-event pair ranking candidate (private)

This entry changes only v3's risk scores. The behavior labels and all five evidence cells remain identical strings. The user submitted it successfully: public score **0.80252**, rank **30 of 104** at observation, improving the previous best **0.79021** by **0.01231**. It is the current best entry; earnings remain **$0** and the score remains below prize contention. See `outputs_v5/submission_receipt.json`.

The existing pair summaries use mean, standard deviation, maximum, and 95th percentile. V5 additionally measures the mean of the three largest hand values and the fraction of nonzero hands for each of the 56 v2 event features. These 112 summaries distinguish repeated unusual events from a single extreme observation. They use observed gameplay, never identifier structure or unknown pair labels.

The model retains the prescribed LightGBM settings (226 trees, 15 leaves, learning rate .03, regularization 5) and original/70%/40% history augmentation. Each source pair has total training weight one. Final fitting uses all 1,860 explicitly labeled pairs, 5,580 weighted views, and 765 features. Thinning does not recompute the older per-player baseline features.

The fixed three pool splits separate all players and keep augmented views together. Pair AP improved on each split for both full and shortened histories:

| Split | Full current → candidate | Short current → candidate | Short hypothetical 1% prevalence AP |
|---|---|---|---|
| 0 | .985198 → .985674 | .974207 → .976957 | .839336 → .891427 |
| 1 | .982605 → .985120 | .971828 → .975744 | .885495 → .936459 |
| 2 | .984782 → .986649 | .975834 → .977005 | .838619 → .850594 |

These are reused, overlapping exploratory validation splits, not three independent final tests. The 1% calculation is sensitivity analysis; evaluation prevalence is unknown. The experiment's own behavior predictions were mixed on shortened histories, so the prepared entry retains v3 behavior and evidence. Independent recomputation with each fold's existing control classes gives risk-only hybrid behavior mAP improvements on all shortened splits: .954240→.957370, .942414→.947283, .961492→.964379. Full-history changes are mixed: .970329→.970209, .967710→.970432, .968359→.970945.

Artifacts:

- `outputs_v5/pair_model_event_tails.joblib`: SHA-256 `5e24d4f9252377366a8336c808fe36225e47872e46114d59a2aa0837a4dd1d11`.
- `outputs_v5/submission_event_tails.csv`: SHA-256 `e2ed0f5c25be000437ccb2ae753649b5643cf328e84c545ed9590b57e84d0e76`.
- `outputs_v5/fit.json`: parameters, versions, input hashes, class counts, and reload verification.
- `outputs_v5/submission_event_tails.provenance.json`: exact preservation of all non-risk strings and all 20 evaluation batches.
- `outputs_v5/submission_event_tails.validation.json`: 112,540 pair rows and 562,700 valid evidence memberships; no errors.
- `reports/tail-pair-experiment/`: paired validation predictions and metrics.

The initial prediction guard caught inconsistent Windows path separators before any predictions were written. Relative paths now use POSIX separators. The first fit is preserved in `outputs_v5/initial_fit_path_guard_failure`; the refitted model is byte-identical. Existing v2/v3/v4 artifacts remain preserved.

To reproduce after the v2 features and v3 submission exist, run from this directory in the recorded environment:

```powershell
& .\.venv\Scripts\python.exe -X utf8 src/tail_pair_experiment.py
& .\.venv\Scripts\python.exe -X utf8 src/tail_pair_candidate.py fit
& .\.venv\Scripts\python.exe -X utf8 src/tail_pair_candidate.py predict
& .\.venv\Scripts\python.exe -X utf8 src/validation.py outputs_v5/submission_event_tails.csv --data-dir data --report outputs_v5/submission_event_tails.validation.json
```

The scripts refuse to overwrite completed results or fitted/submitted artifacts. Use a fresh private reproduction directory with the same inputs. Nothing in this folder has been publicly published; competition data restrictions and the eventual prize writeup requirements still apply.
