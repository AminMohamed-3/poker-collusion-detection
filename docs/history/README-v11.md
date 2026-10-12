Current update, September 10 UTC: all five controlled entries are COMPLETE; best v12/v13 is .86005, rank 22/129 at observation. See reports/conditional-public-results-20260910.md. The preparation notes below are historical.

# Conditional-action candidates — ready, unsubmitted

All three files in `outputs_v11_conditional` pass full validation: **112,540 pairs and 562,700 evidence references each**. Every v5 risk and v8 evidence control reproduces exactly across the 20 evaluation batches. Current public best remains v10, **.84351**; these files do not yet have public scores.

| Candidate | Changes from v10 | SHA-256 |
|---|---|---|
| `submission_conditional_risk.csv` | All pair risks; behavior and evidence identical | `c00b5481be05896fd074c569263934ea0c2af8648548b32d69cf7c1b75ca8c8b` |
| `submission_conditional_evidence.csv` | Evidence for the original 10,000-pair v8 selection; all risks and behaviors identical | `331d04222cf4b3ba670822009714cab524bd073c6bb8dcf01cb014885873b296` |
| `submission_conditional_combined.csv` | Both changes; all behaviors identical | `4058c05c6009ad3075258f38964282dc141c5d9f14818afb3bcc8aa0094febc9` |

The selected risk model has 897 features, including cross-fitted normal-action residual summaries. The evidence ranker has 382 features, adding residuals and pair-relative contexts. Normal references use own cards and public pre-action state; partner cards and the observed action are responses, never predictors. Reference training is limited to confirmed non-target pairs and excludes every pool it scores for training/validation. Unknown pairs and unlisted target hands are never treated as confirmed negatives.

The pure risk tree and equal pretrained hybrid passed the frozen tests; the prespecified rule selected the pure tree. It improves all six known full/short pair-AP comparisons against v10 and all six omitted-family sparse-prior checks. The evidence ranker improves all three known splits and three omitted-family checks; pooled unique-pair MAP gain is **+.039204**, with pool-bootstrap 95% interval **[+.017065, +.061430]**. These reused, overlapping research splits do not establish private-set gains.

The additional coherent-history stress test also **passed**. Whole hands were removed at 65% and 40% observation levels across all 228 validation pools, and all affected player baselines/style rates were recomputed using only retained hands. Pair AP improved on every fold/history comparison against v10. Pooled evidence gains were +.03551 and +.02825 respectively. The original frozen models were retained, and the three GPU TabICL controls matched full-history CPU AP before use. These checks address the earlier retained-baseline limitation without claiming new independent labels or guaranteed private performance. See `reports/conditional-coherent-history/REPORT.md`.

The CPU workers on the supplied Vast container completed and exited. One complete batch was duplicated locally: its pair summaries, evidence-hand features and resulting pair predictions are exactly equal across hosts. All returned artifacts and all twenty unique batches are verified locally. The final fitted models, references and predictions are local; the instance has not been stopped or destroyed.

A separate six-fit pointwise evidence experiment failed every comparison and is excluded. See `reports/conditional-evidence-classifier/REPORT.md`.

## Submission procedure

At the browser check near 22:12 UTC September 9, Kaggle showed **0 submissions remaining**, resetting in about two hours. Do not treat Cairo midnight as the reset. Recheck once the window opens. The user has authorized official CLI/SDK submissions without assistance. Submit the genuine risk-only and evidence-only candidates first; inspect official scored receipts before deciding on the combined file. Do not retry an uncertain upload until the official submission list has been checked.

```powershell
.\.venv\Scripts\kaggle.exe competitions submit -c detect-suspicious-value-transfers-in-poker -f outputs_v11_conditional/submission_conditional_risk.csv -m 'Conditional-action risk: cross-fitted normal-action and partner-strength residuals added to gameplay trees. Passed three pool splits and three omitted-family checks. All v10 behavior labels and evidence retained exactly.'
.\.venv\Scripts\python.exe -X utf8 src/capture_calibration_receipts.py
```

The evidence-only description should state that the same original 10,000-pair selection is reranked and every v10 risk/behavior cell remains fixed. Candidate-specific receipts are stored beside the CSVs. Changing risk affects both PairAP and BehaviorMAP, so do not attribute a risk-only composite delta entirely to PairAP.

## Receipts and reproduction

- `outputs_v11_conditional/ready-audit.json`: full validation, protected assets, all controls, draft and remote-import checks.
- `reports/conditional-action-confirmation/REPORT.md` and `results/summary.json`: frozen tests, complete results and limitations.
- `outputs_v11_conditional/policy`: final cross-fitting plan, references, training features and receipts.
- `fit-plan.json`, `fit.json`, `assembly-plan.json`, `assembly.json`: exact model and candidate provenance.
- `remote-import.json`: archive and cross-host parity verification.
- `reports/methodology-v11-risk-draft.md`: private 1,150-word risk-only draft with five verified retained evidence hands; companion validation JSON. This is not a methodology for the evidence/combined variant.
- Source scripts: `fit_conditional_final_policy.py`, `fit_conditional_final_models.py`, `package_conditional_inference.py`, `conditional_inference_worker.py`, `import_conditional_results.py`, `assemble_conditional_candidates.py`, `audit_conditional_candidates.py` under `src`.

The active user goal is first place with validation discipline. It is not complete. No Kaggle prize has been earned.
