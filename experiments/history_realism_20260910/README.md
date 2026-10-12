# Mini-goal: remove artifacts from complete simulated histories

**Completed:** persistent balances and session seating repair the two dominant defects. Full pair source AUC improves from .9228 to .6682, then .6726 on fresh seeds with identical code and parameters. Whole-player temporal source AUC improves from 1.0000 to .6886, then .6718. Both final batches pass all history-realism checks. The fresh-seed batch fails the separate positive-coverage gate (15/16 cells), so the complete expansion confirmation remains unpassed. See [results](RESULTS.md) and [machine-readable decisions](summary.json).

All 192,000 final-recipe hands are preserved, including excluded targets and failed coverage. No threshold changed, production file changed, original-held trial ran, VM restarted, or RL training started. The existing earning goal remains separate from this completed mini-goal.

Resumed at the user's request on September 10, 2026. This is a new, separate research batch; frozen prior trials and their stopping decisions remain unchanged. Work stays local initially; no VM restart, RL training, detector-utility trial or Kaggle submission is part of this diagnostic stage.

The final earlier simulator passed hand-level checks but benign-pair source AUC remained .9228 versus a .8 ceiling. The hypothesis to test is a mismatch in how player stack ranges, table context and attendance combine over a full history. Begin with source-feature ablations and original-training-only comparisons of serial and within-hand stack dependence, player ranges and pool-level structure. Do not remove revealing features from the admission test to make it pass; ablations diagnose causes only.

Calibration is restricted to the existing 297 training pools. Original outer-held and evaluation observations stay out of all diagnostics and generator fitting. Existing source audits have been reused and are exploratory. A repair must be grounded in a concrete observed mechanism, then checked using fresh generated seeds with the full representation and legal/information/coverage audits. Passing source discrimination alone never establishes detector usefulness or real-world realism.
