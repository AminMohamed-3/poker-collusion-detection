# Conditional evidence coverage

Official entry 56133064 is COMPLETE at **0.86005**, tied with v12 and up .01654 from v10. The latest observed public rank is **22/129**; the leader scores .93720. All five controlled entries have scored. No Kaggle prize has been earned. See reports/conditional-public-results-20260910.md for the component comparisons.

The fifth candidate extends the already validated conditional ranker to the union of the old and new top-10,000 risk lists: 16,060 pairs total. The added set contains 6,060 pairs; 163 of the conditional model's top 1,000 were previously outside the evidence selection. There is no new model fit, threshold search, risk change or behavior change.

The file outputs_v13_coverage/submission_conditional_coverage.csv validates all 112,540 pairs and 562,700 evidence references. It changes 5,928 evidence lists. Every other v12 cell is unchanged. SHA-256: `418084a433e9f251f01b9e7d69028586fb2be7d818ff15bc0a6a856121a21581`.

Twenty bounded inference batches recomputed full sequence, equity and conditional histories for the added pairs. Every old v6 ranking reproduces exactly. One original selected pair per batch also reproduces its previous sequence/equity/conditional features and v12 ranking exactly. The plan, cached features, rankings, batch receipts, validation and ready audit are in outputs_v13_coverage. Run src/extend_conditional_evidence_coverage.py to verify and reproduce this artifact.

The fixed models passed the earlier pool/family and coherent-history checks and the locked 169-pool audit documented in reports/conditional-reserved-audit/REPORT.md. That audit is less-reused validation, not untouched data. The full-history composite rises .894866 to .903852; the pair-thinned composite rises .904439 to .917149. These are local results, not a new leaderboard score.

A separate complete inference replay at 23:46:57 UTC rebuilt predictions from the frozen models and feature caches without predecessor CSV predictions. It recomputed sequence features and ranking across all 9,651,820 pair-hands and produced byte-identical v13 output. This does not claim raw-feature rebuilding or model retraining. See reports/replay-v13/README.md and result.json.

The original submission order and controls remain in reports/conditional-submission-plan.md as a preparation record. The current scored private writeup is reports/methodology-v13-scored-draft.md, 1,403 words with five verified cases. Preserve earlier submissions; do not infer hidden labels from score feedback or alter this fixed coverage rule after viewing its public result. No manual final-entry selection has been made, and private performance remains unknown.
