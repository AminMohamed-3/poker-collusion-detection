Current update, September 10 UTC: all five controlled entries are COMPLETE; best v12/v13 is .86005, rank 22/129 at observation. See reports/conditional-public-results-20260910.md. The preparation notes below are historical.

# Conditional behavior candidate

This separate candidate uses the validated v11 conditional risk and evidence models, with supported-family predictions from the conditional pair classifier. It passed three full-history pool splits, three omitted-family evidence checks and six whole-hand history reductions. Details: reports/conditional-behavior-experiment/REPORT.md.

The file outputs_v12_behavior/submission_conditional_behavior.csv fully validates 112,540 pairs and 562,700 evidence references. All v11 combined risks and all evidence outside the original 10,000-pair selection are unchanged. The assembly reproduces old predictions exactly before applying new family indicators. The ready-audit.json, assembly-plan.json and twenty scored-batch receipts preserve provenance. The v11 files are immutable.

Status: unsubmitted. It should follow the controlled v11 risk/evidence comparisons and be compared against the v11 combined entry. No public improvement or prize is claimed. Local score gains do not guarantee private-leaderboard gains. The unobserved fourth behavior family remains unsupported.

A further locked confirmation on 169 less-reused pools passes all risk, evidence and behavior gates. The composite improves from .894866 to .903852 on full histories and .904439 to .917149 on pair-thinned histories. These pools were used in some earlier validation; this is not an untouched holdout. See reports/conditional-reserved-audit/REPORT.md. The private methodology draft and five retained evidence cases are recorded in reports/methodology-v12-validation.json.

Build and validation: .venv/Scripts/python.exe -X utf8 src/assemble_conditional_behavior.py. Reruns verify existing batch identities and refuse inconsistent artifacts.
