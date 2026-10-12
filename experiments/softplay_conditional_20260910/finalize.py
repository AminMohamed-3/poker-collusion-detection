"""Report every paired controller and both doses, retaining failed gates."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parents[1];W=R.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());diag=json.loads((H/'diagnostics.json').read_text());ev=json.loads((H/'trial/evaluation.json').read_text());freeze=json.loads((H/'trial/freeze.json').read_text());fit=json.loads((H/'controller-fit.json').read_text());gen=json.loads((H/'generation-time.json').read_text())
    audits={m:json.loads((H/m/'pilot-audit.json').read_text()) for m in plan['generator_arms']};replays={m:json.loads((H/m/'intervention-replay-audit.json').read_text()) for m in plan['generator_arms']};receipts=[json.loads((H/'trial'/a/'receipt.json').read_text()) for a in plan['detector_arms']];seconds=sum(r['seconds'] for r in receipts);counts={k:len(v) for k,v in freeze['cohorts'].items()};comparisons=ev['comparisons_to_matched_dose']
    passed=[a for a,c in comparisons.items() if c['passes_utility_screen']];ready=[a for a in passed if diag['results'][a.split('_')[0]]['realism_gate_passed']]
    original=json.loads((H.parent/'softplay_fast_20260910/action-diagnosis.json').read_text())['results']['soft_play'];mismatch=json.loads((H.parent/'softplay_fast_20260910/remaining-mismatch.json').read_text())['results']['real_soft']
    label={'baseline':'Previous rule','strength':'Strength-aware rule','learned':'Learned conditional policy'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');fig.patch.set_facecolor('#fbfaf6')
    for mode,color,offset in [('strength','#bf7e32',-.1),('learned','#087f8c',.1)]:
        vals=[];lo=[];hi=[]
        for dose in ['low','high']:
            c=comparisons[mode+'_'+dose];v=c['mean_deltas']['logloss_improvement'];a,b=c['pool_bootstrap_logloss_improvement_95pct_interval'];vals.append(v);lo.append(v-a);hi.append(b-v)
        axes[0].errorbar(np.array([0,1])+offset,vals,yerr=np.array([lo,hi]),fmt='o-',capsize=4,color=color,label=label[mode])
    axes[0].axhline(0,color='#74818c',lw=1);axes[0].set_xticks([0,1],[f"Small: {counts['low']} unique pairs",f"Large: {counts['high']} unique pairs"]);axes[0].set_ylabel('Loss reduction vs equal-sized real resampling ↑');axes[0].set_title('Does adding unique synthetic data help?');axes[0].legend(frameon=False,fontsize=9);axes[0].grid(axis='y',alpha=.15)
    modes=plan['generator_arms'];v=[diag['results'][m]['source_auc'] for m in modes];bars=axes[1].bar(range(3),v,color=['#adb6c0','#bf7e32','#087f8c']);axes[1].bar_label(bars,fmt='%.3f',padding=3);axes[1].set_xticks(range(3),['Previous\nrule','Strength-aware\nrule','Learned\npolicy']);axes[1].set_ylim(.5,1.03);axes[1].axhline(.85,color='#933c43',ls='--',label='Preset ceiling: .85');axes[1].set_ylabel('Grouped source-classifier AUC ↓');axes[1].set_title('Can real and synthetic pairs be distinguished?');axes[1].legend(frameon=True,facecolor='white',loc='lower left',fontsize=9)
    fig.suptitle('Conditional soft play · paired generators and data size',fontsize=15,weight='bold');fig.savefig(H/'dose-screen.png',dpi=160);plt.close(fig)
    mechanism=[]
    for mode in modes:
        r=diag['results'][mode];a=r['action_summary'];mechanism.append(f"| {label[mode]} | {r['supported_pairs']} | {a['preflop']['equal_pair_mean_fold_check_call_raise'][2]:.2%} | {a['postflop']['equal_pair_mean_fold_check_call_raise'][1]:.2%} | {r['conditional']['preflop_calls']['own_strength']:.3f} | {r['source_auc']:.4f} |")
    table=[]
    for dose in ['low','high']:
        for mode in ['resampled','strength','learned']:
            arm=mode+'_'+dose;rows=[next(r for r in ev['metrics'] if r['arm']==arm and r['scope']=='all' and r['history']==t) for t in ['full','shortened']];c=comparisons.get(arm);d='—' if c is None else f"{c['mean_deltas']['logloss_improvement']:+.6f}";gate='Control' if c is None else 'Pass' if c['passes_utility_screen'] else 'Fail'
            table.append(f"| {mode} / {dose} | {counts[dose]} | {rows[0]['log_loss']:.6f} | {rows[1]['log_loss']:.6f} | {d} | {gate} |")
    detail=[]
    for arm,c in comparisons.items():
        lo,hi=c['pool_bootstrap_logloss_improvement_95pct_interval'];halves=c['pool_half_loss_improvements'];d=c['mean_deltas'];detail.append(f"- **{arm}:** loss gain {d['logloss_improvement']:+.6f}; 95% paired pool interval [{lo:+.6f}, {hi:+.6f}]; pool-half gains {halves[0]:+.6f} / {halves[1]:+.6f}; mean Pair AP delta {d['pair_ap']:+.6f}; mean sparse-prevalence AP delta {d['sparse_ap']:+.6f}.")
    dose_text=[]
    for mode,c in ev['net_high_vs_low_after_resampling_change'].items():
        lo,hi=c['pool_bootstrap_net_loss_improvement_95pct_interval'];dose_text.append(f"- **{mode}:** increasing from {counts['low']} to {counts['high']} unique pairs changes the loss benefit by {c['logloss_improvement']:+.6f} after subtracting the change in real-resampling controls; 95% interval [{lo:+.6f}, {hi:+.6f}].")
    outcome=('No augmentation arm passes the frozen utility gate.' if not passed else 'Utility-screen passes: '+', '.join(passed)+'.')+' '+('No arm passes both realism and utility, so this study does not justify broad expansion or neural training.' if not ready else 'Passing both gates supports a separate replication, not production promotion.')
    report=f'''# Conditional soft play and unique-data dose response

{outcome}

This new user-authorized study preserves the previous experiment's failed gates. Six fixed detector fits took **{seconds:.1f} seconds** locally; paired simulation took {gen['seconds']/60:.1f} minutes. The VM was not restarted.

![Data-size screen](dose-screen.png)

## What was tested

Three controllers replay the same **32 pools / 96,000 exogenous hand scenarios**. Each arm produces 96,000 serialized hands, for 288,000 counterfactual histories in storage; these are **not 288,000 independent scenarios**. Initial hole cards, player profiles, seating schedules and activity periods match exactly across arms. Settled balances can diverge as a consequence of different actions. Reset randomness was separated from seat-schedule randomness to keep the pairing valid.

1. Previous public-state soft-play rule, with identical paired seeds.
2. Strength-aware rule: permit a new call replacing a fold only above the original soft-call 25th-percentile card-strength score within the current street. Keep the earlier aggression-suppression rules.
3. Learned conditional policy: fixed depth-3 action models for soft and normal play, using only own cards and public state/history. It proposes selected calls/checks when passive-action probability exceeds the normal policy; it never sees partner private cards, future cards, validation labels or detector scores. The existing episode/intensity gate still applies.

The learned policy is trained on 26,625 actions from 92 original soft-play pairs, with a normal comparator trained on 220,444 annotated non-target-pair actions. Three training-pool folds exclude their scored pools from both fits. Its mean action log-loss improvement is {fit['oof_logloss_improvement']:.6f}; all three folds improve. This establishes conditional action-prediction signal within the development source, not the utility of sampled full histories. Final controller fitting and all calibration stay within the original 297 outer-training pools.

## Behavior and realism

| Controller | Supported soft pairs | Preflop calls | Postflop checks | Mean strength among preflop calls | Source AUC ↓ |
|---|---:|---:|---:|---:|---:|
| Original training | 92 | {original['preflop']['equal_pair_mean_fold_check_call_raise'][2]:.2%} | {original['postflop']['equal_pair_mean_fold_check_call_raise'][1]:.2%} | {mismatch['preflop_call']['own_strength']:.3f} | — |
{chr(10).join(mechanism)}

Rates are equal-pair means over all supported targets. Separate matched-three-arm summaries for {diag['common_three_arm_supported_pairs']} supported pairs are saved in `diagnostics.json`; pairing does not make unsupported labels valid. Source classifiers use four pool-grouped folds and only original training pools. The preset source-AUC ceiling is .85. These are small, repeated development diagnostics, not an independent realism certification.

## Does more unique data improve detection?

The common-supported strength/learned targets define nested cohorts: **{counts['low']} unique target pairs** in scenarios 0–15 versus **{counts['high']}** in scenarios 0–31. All four ordinary policy families occur in the larger cohort. Each target contributes exactly its full, 70% and 40% history views; no target is repeated to fill a quota. Real controls use the same number of unique original soft-play pairs and the same three view types, nested by dose. All arms retain the same 4,209 original rows and 1,217 features. The baseline generator is a paired diagnostic, not a third detector treatment.

| Detector arm | Unique added pairs | Full multiclass loss ↓ | Shortened loss ↓ | Mean gain vs matched control ↑ | Utility gate |
|---|---:|---:|---:|---:|---|
{chr(10).join(table)}

{chr(10).join(detail)}

The effect of increasing data size, adjusted for the resampling-control change:

{chr(10).join(dose_text)}

Six model/prediction receipts were frozen before the evaluator read labels. Both full and shortened histories use the existing 100 held pools / 457 pairs. The two label-independent hashed pool halves are unchanged from the prior study. The 1,000 bootstrap replicates resample whole pools and keep paired predictions together. **These validation pools have been reused**; related histories and pool halves are not independent tests. No private-leaderboard or prize-outcome estimate follows from this screen. The official evidence ranker was not trained or evaluated here.

## Remaining mismatch

The strength rule brings mean card strength among preflop calls close to the original (.558 versus .565), but calling becomes too rare: 22.2% with the strength rule and 24.2% with the learned policy versus 39.4% originally. A post-screen generated-data check found 34.8% preflop calls during the learned policy's active episodes and 14.2% outside them. This suggests another possible mismatch: the learned model already averages normal and soft play, then the simulator applies additional episode, probability and allowed-transition gates. Those can weaken its effect further. Different state distributions also contribute; this diagnosis does not establish gating as the sole cause or justify tuning against validation results. See `activation-diagnostic.json`.

A next training-only experiment could calibrate this mixture and test complete-policy sampling against selective interventions, preserving supported labels and paired seeds. An evidence-ranking test is another distinct option because current pair AP is near ceiling. Neither follow-up was run, and neither is guaranteed to help.

## Integrity, limits and stopping decision

- All three batches pass all-row accounting, card uniqueness, supported-label and balance-ledger checks. {sum(r['counterfactual_action_changes_verified'] for r in replays.values()):,} actual intervention changes were independently checked against the ordinary action resolved at that state, through {sum(r['hands'] for r in replays.values()):,} legally replayed histories and {sum(r['actions'] for r in replays.values()):,} actions.
- The public/own-card model inputs match independent extraction at {diag['controller_feature_parity_rows']} before-action states. Partner-card invariance and inactive/zero-intensity checks pass.
- Each arm has 960 synthetic player slots, using {audits['strength']['prototype_sources']} unique original-player prototypes, maximum reuse {audits['strength']['maximum_prototype_reuse']}. The same slots are intentionally paired across arms. Four ordinary policy models still share one training source; they are not independent real-world populations.
- Source-model features are cross-fitted as before. Controller fitting, prototype calibration and generation do not use original outer-held or evaluation pools.
- The six detector fits are separate from eight small action-policy fits (six OOF, two final) and twelve small source-diagnostic fits. No neural model, RL training, hyperparameter search, post-result seed selection, or post-result generator refit was run.
- All seven protected submission CSV hashes remain unchanged. No Kaggle upload or production promotion. The VM's previous stop request remains the last observed provider action; final billing state is not independently verified.

More data can help when it expands useful variation with faithful labels. This experiment tests that hypothesis at a small scale instead of assuming row count is evidence of improvement. Preserve all outcomes and use the controller/size results to choose a separately declared next experiment; do not revise this study's thresholds or erase earlier failures.

Reproduction files: `study-plan.json`, `controller-fit.json`, `diagnostics.json`, per-arm audit/replay receipts, `trial/freeze.json`, all six model receipts and `trial/evaluation.json`. Raw histories, features, models and failed comparisons are retained locally.
'''
    (H/'RESULTS.md').write_text(report,encoding='utf-8')
    summary={'completed_at':datetime.now(timezone.utc).isoformat(),'cohort_sizes':counts,'utility_passes':passed,'both_realism_and_utility_passes':ready,'detector_fit_seconds':seconds,'generation_seconds':gen['seconds'],'source_auc':{m:diag['results'][m]['source_auc'] for m in modes},'mean_loss_improvements':{a:c['mean_deltas']['logloss_improvement'] for a,c in comparisons.items()},'net_dose_effect':ev['net_high_vs_low_after_resampling_change'],'protected_unchanged':all(sha(R/p)==v for p,v in plan['protected'].items()),'no_production_promotion':True,'vm_not_restarted':True}
    assert summary['protected_unchanged'];(H/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    paragraph=f"**September 10, conditional soft-play data-size study complete:** three paired generators replayed 32 pools / 96,000 exogenous scenarios (288,000 counterfactual histories, not independent scenarios). Strength-aware and learned public/own-card policies were tested at {counts['low']} vs {counts['high']} unique target pairs, with matched real-resampling controls and six fixed tree fits ({seconds:.1f}s). {outcome} Controller source AUCs: "+', '.join(f"{m} {diag['results'][m]['source_auc']:.4f}" for m in modes)+". All paired seats/cards/profiles and before-action feature checks pass, as do full accounting and causal replay checks. Prior study failures are unchanged. All work local; no VM restart, neural/RL run, new submission or production promotion. Report: `work/poker/experiments/softplay_conditional_20260910/RESULTS.md`.\n\n"
    p=W/'STATUS.md';s=p.read_text(encoding='utf-8');start=s.index('**September 10, conditional soft-play follow-up running:**') if '**September 10, conditional soft-play follow-up running:**' in s else s.index('**September 10, conditional soft-play data-size study complete:**');end=s.index('\n\n',start)+2;s=s[:start]+paragraph+s[end:];p.write_text(s,encoding='utf-8')
    memory=W/'research/POKER_RESEARCH_MEMORY.md'
    if '**September 10, conditional soft-play data-size study complete:**' not in memory.read_text(encoding='utf-8'):
     with memory.open('a',encoding='utf-8') as f:f.write('\n\n'+paragraph+'The user favors making useful additional data, with simple, fast model screens first. Maintain nested unique-target doses and equally sized controls; paired counterfactual histories must not be counted as independent populations.\n')
if __name__=='__main__':main()
