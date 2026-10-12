# Exploratory other-coordination submission

Submit `submission_other_conf050_risk050.csv` after your Kaggle quota resets.

Base: current best .89457. The sole change is three behavior labels: current risk >= .5 and conditional confidence in every known family < .5 become `other_coordination`. All risks, evidence, other rows and row order remain identical. Full submission validation passed for all112,540 pairs.

This is one user-authorized exploratory test, not a locally proven improvement or verified fourth-family discovery. The threshold is a simple no-majority rule, not a calibrated unknown probability. It could help, hurt, or produce no visible change. A score change cannot identify which individual cases are correct. No raw-hand judgments, manually assigned test labels, or case-specific overrides were used.

Suggested description: Exploratory known-family abstention: risk >=0.50 and conditional family max <0.50; three behavior changes only; risks and evidence identical to best .89457.

Provenance and exact differences: protocol.json, build-receipt.json, validation.json. This file has not been submitted by Codex.
