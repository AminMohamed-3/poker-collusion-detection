# Action-sequence evidence candidate (private)

V6 was submitted and scored **0.80687** (submission 56122936). It changes only the evidence hands, preserving every v5 pair ID, risk score and behavior label exactly. The newer public best is v10's pretrained risk blend at **0.84351** (September 9 UTC CLI receipt). V6 remains an immutable control. No Kaggle prize has been earned.

The existing 222 evidence features gain 23 symmetric gameplay sequence measurements and their 23 within-pair z-scores. These describe outsider responses to partner raises, prior investment, partner folds, mutual checks and resulting chip outcomes. The LightGBM LambdaRank parameters remain fixed. The final model fits 372 labeled target pairs and 45,129 hands with 268 features. Unlisted pairs are never made negative training examples.

| Local evaluation | Existing evidence MAP@5 | Sequence evidence MAP@5 |
|---|---:|---:|
| Reused screening pools | 0.468359 | 0.505125 |
| 100 less-reused confirmation pools | 0.460529 | 0.482867 |
| Same confirmation pools, coherent 65% histories, retained annotations | 0.555908 | 0.561828 |

These gains are exploratory. Omitted-family screening worsened by an average 0.01064. The confirmation gain's paired pool-bootstrap 95% interval is **[-0.010367, +0.055500]**. Both strict gates failed; `reports/sequence-evidence-experiment/confirmation-plan.md` explicitly records the separate adaptive decision to prepare one public entry. No feature, parameter or blend-weight tuning was done against the confirmation pools. Some reviewed first-ranked evidence hands are weak or ordinary play. Preserve v5 if this candidate scores worse.

Two prior experiments were rejected: richer pair-relative ranks/z-scores and direct MAP optimization with XGBoost. Their fixed controls and failed comparisons remain in `reports/relative-evidence-experiment/` and `reports/map-evidence-experiment/`.

## Verification and artifacts

- All 20 evaluation batches finished, covering 9,651,820 shared pair-hands.
- `outputs_v6/submission_sequence_evidence.csv`: 112,540 pairs, 562,700 valid evidence references, 384,403 distinct evidence hands; full validator reports no errors. SHA-256 **bc309294d0e2feecc2cded72ed0330498485ccea55dfa433c8c6a5ebf05457f7**.
- `outputs_v6/sequence_ranker.joblib`: SHA-256 **23785e38e3d5c17f75513cf5a314f14e572db9ab30c09987d1ee105ee5740d21**. Reload predictions match exactly.
- `outputs_v6/fit.json`, `submission_sequence_evidence.provenance.json` and `rank_batches/manifest.json`: parameters, input/source hashes, coverage and exact preserved cells. The provenance's `membership_validation_required` flag records the assembly stage; the separate completed validation report supplies the subsequent result.
- Six scalar sequence tests cover substantive gameplay cases. Vectorized extraction exactly matches **1,037,967** scalar development cells, with pair/input-row reversal checks; see `reports/sequence-evidence-experiment/fast-verification.json`.
- `reports/methodology-v6-draft.md`: private methodology and five honest case reviews. All 25 underlying evidence records are frozen in `reports/selected-v6-evidence-cases.json`.

## Reproduction

Use a fresh private copy with the recorded Python environment and original competition data. First reproduce the previous feature and model dependencies in README-v2/v3/v5; retain v5's frozen submission. Scripts refuse to overwrite completed experiments or fitted outputs.

```powershell
& .\.venv\Scripts\python.exe -X utf8 src/sequence_evidence_experiment.py
& .\.venv\Scripts\python.exe -X utf8 src/sequence_evidence_confirmation.py
& .\.venv\Scripts\python.exe -X utf8 src/verify_sequence_fast.py
& .\.venv\Scripts\python.exe -X utf8 src/sequence_evidence_candidate.py train
& .\.venv\Scripts\python.exe -X utf8 src/sequence_evidence_candidate.py predict
& .\.venv\Scripts\python.exe -X utf8 src/validation.py outputs_v6/submission_sequence_evidence.csv --data-dir data --report outputs_v6/submission_sequence_evidence.validation.json
& .\.venv\Scripts\python.exe -X utf8 src/extract_v6_cases.py
```

The confirmation script retains the preparation failures and fixed membership described in its plan. Inspect its expected input/output paths when reproducing in a fresh directory. Do not rerun confirmation to tune the candidate. Data, code and writeups remain private until the competition's publication requirements apply.
