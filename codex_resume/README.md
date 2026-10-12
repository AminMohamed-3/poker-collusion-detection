# Codex continuation, September 12, 2026

The user resumed research after two days with Claude and reported public score .89447, rank 10. Immediate target: improve toward .91 without weakening validation. Claude's complete notes and artifacts are read-only inputs; all new code and experiments go here. The previous Codex and Claude files remain preserved.

Read `../Claude Version/OBSERVATIONS.md` (especially 17 and 18), `../Claude Version/HANDOFF_CLAUDE_TO_CODEX.md`, and `../Claude Version/REPRODUCE.md`. Some historical conclusions and the reproduction document contain stale score attribution: original v13 scored .86005; chronology_frac scored .87625. The exact file behind .89055 is now verified as the S blend in the official submission history. Risk changes affect both PairAP and risk-gated BehaviorMAP, so their composite delta must not automatically be divided by .7. Public differences are exact for the fixed public data but not estimates of private differences with known uncertainty.

Current best reported file: `../Claude Version/outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv`. Claude's prespecified simpler reference uses S order with identical evidence and scored .89336. Preserve both. The evidence reference is the full chain: chronology stage-1 + r382 stage-2 for predicted transfer/soft play + structural stage-2 for predicted isolation. A gain over stage-1 alone or over structural stage-2 alone is not enough.

No new evaluation labels, no label probes, no VM restart/spending, no subagents. Use bounded two-thread local CPU experiments. All new selection is on the 162-pair audit outside the union of outer folds; outer folds have been reused four times and are not pristine. A promising audit requires a frozen, grouped confirmation against the complete incumbent, all variants preserved, and a negative control. Do not label unknown pairs benign from their scores. Eligibility for pair-level stress tests is at least 38 shared hands. Read individual plans before invoking old scripts: many are write-once or have side effects.

First study: `experiments/symmetry_censor_20260912`. Test two untried mechanisms: unordered-player invariance in the structural features, and less weight on unlisted late hands whose annotation status may be censored by a five-hand cap. Censor weights are training-only, do not declare those hands positive, and do not enter inference features. This is an approximation to missing-label supervision, not an unbiased estimator or a recovered annotation rule.

One important audit limitation: the cheap two-stage cross-validation uses common stage-1 OOF scores; it is exploratory stacking selection, not strictly nested evaluation. Any accepted candidate must subsequently use stage-1 references entirely inside each outer-training population. Audit stage-2 routing is by known family as in Claude's existing harness; confirmation must use fixed predicted family.

## Latest instructions and status

The historical startup restrictions above were superseded by the user authorizing the supplied VM and Sol/Terra review agents. The active target is .95 public with disciplined incremental feedback. Read `STATUS.md` for current score, completed studies, compute and next work; the earlier paragraph naming the best file is historical.
