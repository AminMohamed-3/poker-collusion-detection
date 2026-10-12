# Private v13 inference replay

Completed September 9, 2026 at 23:46:57 UTC. The replay produced 112,540 rows and recomputed sequence features and rankings over all 9,651,820 evaluation pair-hands. Every cell and the CSV bytes match the frozen v13 candidate. SHA-256: `418084a433e9f251f01b9e7d69028586fb2be7d818ff15bc0a6a856121a21581`.

The replay reads no predecessor submission to supply predictions. It reconstructs old family assignments and risk selection from the saved v2/v5 models, new risk and behavior from the conditional model, and the union of 16,060 selected pairs. It recomputes sequence features from supplied evaluation gameplay. For selected pairs it reuses previously verified equity and conditional-action caches and applies the frozen 382-feature ranker. For the remaining pairs it reapplies the sequence ranker with its original family assignments. The original candidate is read only after constructing the output, for the final comparison.

This is an inference replay. It does not retrain models, rebuild every raw feature from scratch, or establish that a fresh environment installation reproduces all results. Source feature extraction, model fitting, held-pool validation and host parity have separate receipts in their original directories. The full membership validator has already checked every reference in the original candidate; byte identity establishes that this replay has the same memberships.

From `C:\Data\Projects\Money Making\work\poker`:

```powershell
.\.venv\Scripts\python.exe -X utf8 src/archive_v13_replay.py
.\.venv\Scripts\python.exe -X utf8 src/replay_v13_models.py
```

The first command verifies the 255-file, 1,470,069,250-byte private inventory, including a snapshot of current Python sources and observed dependencies. It never installs packages. The second command verifies and reuses completed replay batches if they exist; the recorded first run constructed every batch. For a fresh inference run, use a separate copy of the project with the inventoried inputs and no `reports/replay-v13/batches` directory. Preserve the recorded artifacts in this workspace.

Key records: `plan.json`, `result.json`, `input-inventory.json`, `requirements-observed.txt`, `source-snapshot`, and twenty pairs of pair/ranking batch receipts. The archive is private and contains competition-derived data references; no public distribution has occurred. The five unsubmitted candidates and upload controls are listed in `../conditional-submission-plan.md`.
