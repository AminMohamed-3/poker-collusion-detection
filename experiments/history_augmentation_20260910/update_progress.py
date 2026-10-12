from pathlib import Path
H=Path(__file__).resolve().parent
p=H.parents[3]/'STATUS.md';s=p.read_text(encoding='utf-8')
tag='## Current work\n\n'
entry='''**September 10, augmentation retest running after the user restarted the VM:** reconnected to instance 50489879 (RTX 3080 20 GB). A frozen sensitivity batch generated 192,000 hands across 64 pools and four balance/session regimes. Cards/accounting, transition ledgers and independent replay pass; deliberate nuisance variation fails the unchanged source ceilings (pair AUC .8908, temporal AUC .9809), so it is retained only for synthetic stress testing. Actual continuation spans 94.26-98.07%; the observed 96.8% is not a universal constant. The admitted retest compares both calibrated history-repair batches against matched original-data resampling: 4,875 rows per arm, 666 extra views from 222 pairs, exact class/step matching, original-only normalization. Four tree fits are complete and 18 fixed neural fits are running; original-held labels have not been scored. Upload-review concerns were resolved by audited training boundaries and proof that original arrays were already resident on the user-provided VM. No new submission or production promotion. See `work/poker/experiments/history_augmentation_20260910/study-plan.json` and `trial/plan.json`. This running status supersedes older paused statements below.

'''
assert entry not in s;p.write_text(s.replace(tag,tag+entry,1),encoding='utf-8')
(H/'README.md').write_text('''# History augmentation retest - in progress

The user asked whether observed 96.8% balance continuation could shift, and requested a new augmentation test on the restarted VM. It can shift; no hidden-test stability is assumed.

Two usable training arms (matched real-data resampling and calibrated repaired simulation), three neural architectures with three seeds each, and two tree feature sets. All use 4,875 rows, identical label counts and original-only normalization. Four tree fits are complete. Eighteen neural fits run under `poker_history_trial` on Vast instance 50489879. No held-label scoring until every fit is complete.

A 192,000-hand, four-regime variation batch passed engine, ledger and replay checks but failed pair (.8908) and temporal (.9809) source-AUC ceilings. It was excluded from training under the prespecified rule. All fixed models will also receive this dataset as an explicitly synthetic stress test, reporting every regime without tuning or promotion from those scores.

Original held pools have been used by previous research. Results are exploratory, not untouched or private-test estimates. The prior 15/16 coverage confirmation remains failed. Current Kaggle candidates remain protected.

Files: `study-plan.json`, `trial/plan.json`, `regime-diagnostics.json`, `stress-plan.json`, `varied/*-audit.json`. Final results and backup receipts will be added after completion.
''',encoding='utf-8')
