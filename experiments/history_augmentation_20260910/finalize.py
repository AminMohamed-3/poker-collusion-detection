"""Assemble the frozen experiment's results without changing admission rules."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'trial'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(O/'plan.json');ev=read(O/'evaluation.json');stress=read(O/'stress-evaluation.json');unc=read(O/'uncertainty.json')
assert ev['plan_sha256']==stress['plan_sha256']==sha(O/'plan.json')
assert all(sha(R/n)==v for n,v in p['protected'].items())
assert read(H/'raw-backup-receipt.json')['all_generator_member_hashes_match']
assert read(H/'cross-batch-identity.json')['no_identifier_or_full_payload_overlap']
assert read(H/'varied/replay-counted-audit.json')['pools']==64
decisions=ev['decisions']['calibrated vs resampled_control'];passed=[n for n,d in decisions.items() if d['passed_data_screen']]
labels={'residual_budget':'Residual','mhc':'mHC','single_pass':'SinglePass','tree897':'Tree (897)','tree1217':'Tree (1,217)'}
summary={'completed_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(O/'plan.json'),'neural_fits':18,'tree_fits':4,
         'passed_data_endpoints':passed,'production_admitted':False,'decisions':decisions,
         'varied_source_auc':{'pair':read(H/'varied/pair-domain-audit.json')['auc'],'temporal':read(H/'varied/temporal-audit.json')['gates']['all']['auc']},
         'varied_data_excluded_from_training':True,'stress_deltas':stress['deltas'],'uncertainty':unc['rows'],
         'all_protected_submissions_unchanged':True,'kaggle_uploads':0,'new_prize_income':0,
         'training_rows_per_arm':4875,'extra_pair_views':666,'extra_pairs':222,
         'not_an_untouched_holdout':True,'inference':'These results apply to this generator, data dose and frozen recipes. They do not prove all augmentation or mHC ineffective and do not estimate hidden-test gains.'}
(H/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
names=list(decisions);loss=[decisions[n]['mean_deltas']['logloss_improvement'] for n in names];ap=[decisions[n]['mean_deltas']['pair_ap'] for n in names]
interval={r['variant']:r['pool_bootstrap_95_interval'] for r in unc['rows']}
fig,axs=plt.subplots(1,2,figsize=(11,4.6));plt.rcParams.update({'font.family':'DejaVu Sans'})
for ax,values,title in zip(axs,[loss,ap],['Held-data log-loss reduction','Held-data pair AP change']):
    colors=['#24765f' if v>=0 else '#ac4c3b' for v in values]
    ax.barh([labels[n] for n in names],values,color=colors);ax.invert_yaxis();ax.axvline(0,color='#444',lw=.8)
    ax.set_title(title,loc='left',fontsize=12);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    if ax is axs[0]:
        for i,n in enumerate(names):ax.plot(interval[n],[i,i],color='#333',lw=1.4)
    ax.set_xlabel('Positive favors calibrated augmentation')
fig.suptitle('Repaired synthetic histories versus matched real-data resampling',x=.08,ha='left',fontsize=14)
fig.text(.08,.015,'All seeds and both histories included. Loss intervals resample whole pools; reused validation data.',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.045,1,.93]);fig.savefig(H/'augmentation-results.png',dpi=180);plt.close(fig)
headline=(f"{len(passed)} of five endpoints passed the fixed data screen: "+', '.join(labels[n] for n in passed)+'.') if passed else 'None of the five endpoints passed the fixed data-improvement screen.'
lines=['# Repaired-history augmentation retest','','## Outcome','',headline+' No production model or submission was promoted. This is a test of this synthetic-data recipe and dose, not a general rejection of data augmentation or mHC.','',
'## What was tested','','The user restarted Vast instance 50489879. The GPU check succeeded on an RTX 3080 with 20 GB memory. The trial compared matched resampling of real training pairs with calibrated synthetic histories from both repaired batches. Each arm has 4,875 rows: 4,209 original full/70%/40% views plus 666 extra views from 222 pairs. Extras contain 200 benign, eight directed-transfer, six soft-play and eight coordinated-isolation pairs. All class counts, optimizer steps, seeds and normalization are matched. The selected generated pairs cover all 64 calibrated pools; no pool contributes more than 2.3%, and each related ordinary-policy family contributes about 25%.','',
'Eighteen neural fits cover residual, mHC and SinglePass architectures, each with three fixed seeds and 20 epochs in both arms. Four LightGBM fits cover the 897- and 1,217-feature recipes. Original held labels stayed out of the workers and were scored only after all fits completed. The scaler uses original training rows only. Both full and shortened held histories are included. These 100 original held pools have been used in previous research; the results are exploratory and not an untouched private-test estimate.','',
'## Original-held utility','','Positive deltas favor augmentation. Neural results below average paired metrics across all seeds and both histories; tree results average both histories. The AP and loss gates are distinct, so a favorable value alone does not imply admission.','',
'| Model | Log-loss reduction | Pair AP change | Weighted pair/behavior change | Passes data screen |','|---|---:|---:|---:|:---:|']
for n,d in decisions.items():
    m=d['mean_deltas'];lines.append(f"| {labels[n]} | {m['logloss_improvement']:+.6f} | {m['pair_ap']:+.6f} | {m['pair_behavior']:+.6f} | {'Yes' if d['passed_data_screen'] else 'No'} |")
lines+=['','![Held-data effects](augmentation-results.png)','','The descriptive loss intervals resample whole pools jointly across both histories and all seeds. They are conditional on these fitted models and do not correct for repeated research or multiple endpoints. No seed, blend weight, checkpoint or cutoff was chosen using these results.','',
'## Could the 96.8% continuity change?','','Yes. The observed rate describes the available original training histories. Legal poker accounting requires settled chips to be carried forward unless there is an explicit buy-in, cash-out, reset or a change in what histories were observed; it does not require the exact fitted reset probabilities, balance bounds or session process to remain fixed in hidden data. We did not add a feature that declares deviations from 96.8% suspicious.','',
'A separate 192,000-hand batch crosses four balance/session regimes with all policy and behavior conditions. Every reset remains explicit and all chip accounting is preserved. The observed continuation rates span 94.26-98.07%. Cards, ledgers and independent replay pass. However, this deliberate variation is readily separable from the observed original distribution: pair-source AUC .8908 and temporal-source AUC .9809 exceed the unchanged .8 ceiling. The fixed study therefore excludes it from training and retains it as an inference-only stress test. Source separability alone does not prove that such variation could never help; that would require a separately predeclared study.','',
'| Stress regime | Exact balance continuation |','|---|---:|']
for row in read(H/'regime-diagnostics.json')['varied_regimes']:lines.append(f"| {row['regime'].replace('_',' ')} | {100*row['exact_continuation']:.3f}% |")
lines+=['','## Synthetic shift results','','These are fixed-model predictions on generated data excluded from training. They are not Kaggle scores, private-test estimates, or evidence for changing the promotion gates. The per-regime results have few positives and are reported in full in `trial/stress-evaluation.json`.','',
'| Model | Aggregate stress Pair AP change | Aggregate stress log-loss reduction |','|---|---:|---:|']
for n in names:
    r=next(r for r in stress['deltas'] if r['variant']==n and r['regime']=='all');lines.append(f"| {labels[n]} | {r['pair_ap_delta']:+.6f} | {r['logloss_improvement']:+.6f} |")
lines+=['','## Remaining mismatch and limitations','','A new training-only diagnostic separates generated from original examples within each behavior class. Source AUC is .7644 for benign pairs, .8273 for directed transfer, .9880 for soft play and .7973 for isolation. The positive samples are small (13, 10 and 14 pairs), so this is a lead, not proof of bad labels. Soft-play differences center on calling behavior and conditional action/card residuals. Repairing stack and seating history does not by itself repair the coordination-policy distribution.','',
'The two calibrated batches still share four related ordinary-action models and one original data source. There are 1,732 distinct sampled training-player prototypes, which is not 1,732 independent learned policies. The earlier fresh-seed 15/16 positive-coverage failure remains recorded; the pooled dataset does not retroactively pass that confirmation. All current and earlier failures are retained.','',
'## Verification and artifacts','','All 384,000 full payloads across calibrated and shifted batches are unique, with 128 pools and 3,840 simulated player IDs. The new 192,000-hand raw archive has 576 verified generated files. Independent replay covers 640 hands and 5,950 actions across all 64 shifted pools. A reused receipt had a hard-coded `pools=32` field; a separate corrected replay receipt reports 64 and preserves the original audit and all validity assertions. Protected production submissions remain byte-identical.','',
'[Study plan](study-plan.json) · [Frozen training plan](trial/plan.json) · [Full held metrics](trial/evaluation.json) · [All stress metrics](trial/stress-evaluation.json) · [Pool intervals](trial/uncertainty.json) · [Class-source diagnostic](class-source-diagnostic.json) · [Raw backup receipt](raw-backup-receipt.json) · [Machine-readable summary](summary.json)','']
(H/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
(H/'README.md').write_text('# Completed augmentation retest\n\n'+headline+'\n\n[Full report](RESULTS.md) · [Summary](summary.json) · [Frozen plan](trial/plan.json)\n',encoding='utf-8')
print(json.dumps({'passed_data_endpoints':passed,'production_admitted':False,'protected_unchanged':True},indent=2))
