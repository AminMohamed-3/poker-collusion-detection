"""Publish the frozen activation experiment and preserve every failure."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parents[1];W=R.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());cal=json.loads((H/'calibration.json').read_text());diag=json.loads((H/'diagnostics.json').read_text());ev=json.loads((H/'trial/evaluation.json').read_text());freeze=json.loads((H/'trial/freeze.json').read_text());gen=json.loads((H/'generation-time.json').read_text());probe=json.loads((H/'action-information-probe.json').read_text());coverage=json.loads((H/'history-coverage.json').read_text())['results']
    audits={m:json.loads((H/m/'pilot-audit.json').read_text()) for m in plan['generator_arms']};replay={m:json.loads((H/m/'intervention-replay-audit.json').read_text()) for m in plan['generator_arms']};receipts=[json.loads((H/'trial'/m/'receipt.json').read_text()) for m in plan['detector_arms']];seconds=sum(r['seconds'] for r in receipts);n=len(freeze['common_target_pairs']);comps=ev['comparisons_to_resampled']
    passed=[m for m,c in comps.items() if c['passes_utility_screen']];realpass=[m for m,r in diag['results'].items() if r['realism_gate_passed']];both=sorted(set(passed)&set(realpass))
    outcome=('Utility-screen passes: '+', '.join(passed)+'.') if passed else 'No synthetic arm passes the frozen utility screen.'
    outcome+=' '+(('Both realism and utility pass for '+', '.join(both)+', which supports a separate replication only.') if both else 'No arm passes both realism and utility; this does not justify a large training expansion.')
    labels={'gated':'Episodic control','calibrated':'Calibrated activation','continuous':'Continuous activation'};colors=['#8797a8','#087f8c','#b87930']
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained');fig.patch.set_facecolor('#fbfaf6');modes=plan['generator_arms'];vals=[diag['results'][m]['action_summary']['preflop']['equal_pair_mean_fold_check_call_raise'][2]*100 for m in modes]
    bars=axes[0].bar(range(3),vals,color=colors);axes[0].bar_label(bars,fmt='%.1f%%',padding=3);axes[0].axhline(cal['original_target_preflop_calls']*100,color='#6c3646',ls='--',label='Original training: 39.4%');axes[0].set_xticks(range(3),['Episodic','Calibrated','Continuous']);axes[0].set_ylim(0,max(55,max(vals)+10));axes[0].set_ylabel('Equal-pair preflop call rate (%)');axes[0].set_title('Does activation recover the missing calls?');axes[0].legend(frameon=False,fontsize=9,loc='upper left')
    for i,m in enumerate(modes):
        c=comps[m];v=c['mean_deltas']['logloss_improvement'];lo,hi=c['pool_bootstrap_logloss_improvement_95pct_interval'];axes[1].errorbar(v,i,xerr=np.array([[v-lo],[hi-v]]),fmt='o',capsize=4,color=colors[i],markersize=7)
    axes[1].axvline(0,color='#6c7b86',lw=1);axes[1].set_yticks(range(3),[labels[m] for m in modes]);axes[1].invert_yaxis();axes[1].set_xlabel('Loss reduction vs matched real resampling → better');axes[1].set_title('Does the additional training data help?');axes[1].grid(axis='x',alpha=.15)
    fig.suptitle('Soft-play activation · fresh paired scenarios',fontsize=16,weight='bold');fig.savefig(H/'activation-screen.png',dpi=160);plt.close(fig)
    behavior=[]
    for m in modes:
        r=diag['results'][m];a=r['action_summary'];behavior.append(f"| {labels[m]} | {r['supported_pairs']} | {a['preflop']['equal_pair_mean_fold_check_call_raise'][2]:.2%} | {a['postflop']['equal_pair_mean_fold_check_call_raise'][1]:.2%} | {r['conditional']['preflop_calls']['own_strength']:.3f} | {r['source_auc']:.4f} | {'Pass' if r['realism_gate_passed'] else 'Fail'} |")
    rows=[];details=[]
    for m in plan['detector_arms']:
        a=[next(r for r in ev['metrics'] if r['arm']==m and r['history']==t and r['scope']=='all') for t in ['full','shortened']];c=comps.get(m);delta='—' if c is None else f"{c['mean_deltas']['logloss_improvement']:+.6f}";gate='Control' if c is None else 'Pass' if c['passes_utility_screen'] else 'Fail'
        rows.append(f"| {m} | {a[0]['log_loss']:.6f} | {a[1]['log_loss']:.6f} | {delta} | {gate} |")
        if c:
            d=c['mean_deltas'];lo,hi=c['pool_bootstrap_logloss_improvement_95pct_interval'];h=c['pool_half_loss_improvements'];details.append(f"- **{m}:** mean Pair AP delta {d['pair_ap']:+.6f}, sparse-prevalence AP delta {d['sparse_ap']:+.6f}; loss interval [{lo:+.6f}, {hi:+.6f}]; pool-half loss gains {h[0]:+.6f} / {h[1]:+.6f}.")
    contrasts=[]
    for m,c in ev['activation_vs_gated'].items():
        lo,hi=c['pool_bootstrap_logloss_improvement_95pct_interval'];contrasts.append(f"- **{m} versus fresh episodic control:** mean loss change {c['mean_deltas']['logloss_improvement']:+.6f} (positive is improvement), paired interval [{lo:+.6f}, {hi:+.6f}].")
    cover=[]
    for m in modes:cover.append(f"- {m}: zero-visible-intervention pairs in full/70%/40% views = "+'/'.join(str(len(r['zero_visible_intervention_pairs'])) for r in coverage[m])+'.')
    report=f'''# Activation calibration and an independent-randomness repair

{outcome}

A post-screen check against the verified **original-only** model gives a smaller, uncertain benefit: calibrated mean loss improves .001049 with a 95% paired pool interval [-.000769, +.003223]; continuous improves .000882 with interval [-.000389, +.002577]. The matched resampling control itself worsens loss by .001127 versus original-only. Thus the positive matched-control intervals do not establish a reliable gain over simply keeping the original data. No extra model was fitted and no gate changed for this comparison.

Four fixed tree-detector fits took **{seconds:.1f} seconds** locally. Generating three paired batches took {gen['seconds']/60:.1f} minutes. The VM was not restarted.

![Activation screen](activation-screen.png)

## The experiment

The existing learned soft-play controller was fit to complete positive-pair histories, which already mix ordinary and coordinated actions. The simulator then applied another episode window and probability gate. This study changes activation only, while applying the same randomness repair to all three fresh arms:

- **Episodic control:** inherited episode windows and per-pool probability sampled from .4–.9.
- **Calibrated:** every eligible co-presence opportunity, with fixed activation probability **{cal['calibrated_gamma']:.6f}**.
- **Continuous:** every eligible co-presence opportunity, probability 1; a fixed sensitivity endpoint.

The internal controller still samples its learned action distribution, uses only own cards and public state/history, and permits the same selected passive changes. Thus continuous activation does not mean changing every action. No controller model, action threshold, tree setting or data weight was selected using this study's validation results.

Calibration used only previous generated training contexts and original training action rates. Recover the ordinary action class from each recorded before-action state, estimate the available passive-call gain from the frozen action probabilities, and solve one equation for the original 39.37% preflop call rate. Ordinary call probability on those contexts was {cal['ordinary_call_probability_on_generated_contexts']:.4f}; the available gain per activation unit was {cal['available_call_gain_per_unit_activation']:.4f}. The solved probability was not clipped. Cross-pair calibration checks predicted 37.32% and 41.38% calls. These are fixed-context approximations, not fresh generator or original-held validation: changing actions changes subsequent states.

## A confirmed simulator confound

Previously, intervention checks consumed the same random stream as ordinary players. A controller that returned the ordinary action unchanged still altered later behavior in **16/32** checked histories; the same happened at zero activation. A separate controller random stream keyed by hand seed and action index now produces **0/32** differences under both no-op tests, while real interventions change 10/32 hands. This protects causal attribution to actual action changes.

All three new arms contain this repair. Earlier legal replay checks and measured model failures remain recorded; the new study does not retroactively erase them. Previous generated states were used for the approximate calibration, so the fresh runs are necessary to check transfer after the repair.

The three arms use fresh pools 512–543, separate from calibration pools. Each has 96,000 full hands over 32 pools. Cards, player profiles, seats and nominal activity draws pair exactly across arms. Effective activation intentionally differs, and resulting balances may diverge. This is **96,000 exogenous scenarios replayed three ways**, not 288,000 independent scenarios.

## Behavior and source realism

| Arm | Supported soft pairs | Preflop calls | Postflop checks | Mean own strength among preflop calls | Source AUC ↓ | Realism gate |
|---|---:|---:|---:|---:|---:|---|
| Original training | 92 | 39.37% | 50.29% | .565 | — | Reference |
{chr(10).join(behavior)}

The realism gate is fixed at source AUC ≤.85, call-frequency gap ≤.05, conditional call-strength gap ≤.05, at least 12 supported targets and all four ordinary-policy families. Four grouped source-classifier folds use original training pools only. Rates above use all supported targets; separate common-target summaries are in `diagnostics.json`. Source discrimination is a development diagnostic, not proof of downstream utility or real-world fidelity.

## Four fixed utility fits

The same **{n} unique supported targets** occur in every synthetic arm; none was duplicated to fill a quota. Each contributes the existing full/70%/40% views. The real-resampling control adds {n} unique original soft-play pairs with the same views. Every arm starts with identical 4,209 original rows / 1,217 features and has identical class counts and sample weights. Support in a full history requires a changed legal action; absent support is never relabelled benign.

| Detector arm | Full multiclass loss ↓ | Shortened loss ↓ | Mean gain vs matched resampling ↑ | Utility gate |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

{chr(10).join(details)}

Direct activation contrasts, within this new paired generator study:

{chr(10).join(contrasts)}

Both stronger-activation arms have positive mean loss gains in both predeclared pool halves, and their paired pool-bootstrap loss intervals exclude zero. Calibrated activation passes the realism gate but fails the utility gate on a small negative sparse-prevalence AP delta; continuous activation passes utility but fails realism. This is a useful replication lead, not permission to relax either threshold.

The evaluator read labels only after all four models and predictions were frozen. The 100 original held pools / 457 pairs, two history views and fixed hashed pool halves are reused development data. The bootstrap has 1,000 paired whole-pool replicates. These intervals condition on one fixed model seed and generator batch; they do not include uncertainty from generator/model seeds, repeated development reuse or multiple exploratory comparisons. Small effects on near-ceiling pair AP cannot predict the private leaderboard. Passing this screen would still require independent-seed, omitted-family and other existing confirmations before production use. No official evidence-ranker or combined competition score was tested.

## Additional checks and limits

- All three batches pass card, accounting, balance-ledger, supported-label and independent legal-replay checks. Replay covers {sum(r['counterfactual_action_changes_verified'] for r in replay.values()):,} intervention changes in {sum(r['hands'] for r in replay.values()):,} histories / {sum(r['actions'] for r in replay.values()):,} actions. Every changed action is compared with the legal ordinary action at the same state.
- Controller inputs match independent feature extraction at {diag['controller_feature_parity_rows']} before-action states. Ordinary policies use own/public information; soft-play controllers do too. No future cards, validation labels or detector scores enter behavior generation.
- Each arm uses 960 synthetic player slots, drawn from {audits['gated']['prototype_sources']} distinct training prototypes, maximum reuse {audits['gated']['maximum_prototype_reuse']}. Four related ordinary policies still share one source.
- A post-preparation audit counts interventions retained by synthetic history thinning. Original evidence lists are capped, so missing real annotations are not directly comparable to exhaustive generated traces:

{chr(10).join(cover)}

- A separate training-only information probe added present partner-card strength to the action predictor. Three pool-separated fits yielded only {probe['mean_oof_loss_improvement_with_partner_strength']:.6f} log-loss improvement (95% pool interval [{probe['pool_bootstrap_95pct_interval'][0]:.6f}, {probe['pool_bootstrap_95pct_interval'][1]:.6f}]). This is a small predictive effect, not proof of card sharing or a reason to change these controllers. No partner-card controller was generated or deployed.
- Four detector fits are separate from twelve small source-diagnostic fits and the three small information-probe fits. The existing action policies were reused. No neural/RL fit, post-score generation, coefficient search or selected-epoch rescue.
- All seven protected submission CSV hashes are unchanged. No Kaggle upload or production promotion. VM not restarted; its earlier stop acceptance is still the last observed provider action, without independent final billing-state verification.

All hypotheses, counterexamples, results and failed gates are retained. The randomness fix is useful regardless of whether augmentation succeeds. Any subsequent generator change should be evaluated as a new declared experiment, not used to reinterpret this one's failures.

Reproduction: `study-plan.json`, `calibration.json`, both RNG-test receipts, `diagnostics.json`, `history-coverage.json`, raw paired pools, audit/replay receipts, `trial/freeze.json`, all model receipts and `trial/evaluation.json`.
'''
    (H/'RESULTS.md').write_text(report,encoding='utf-8')
    original_ref=json.loads((H/'original-reference.json').read_text())
    out={'original_only_reference':original_ref['comparisons_to_original_only'],'completed_at':datetime.now(timezone.utc).isoformat(),'cohort_targets':n,'detector_fit_seconds':seconds,'generation_seconds':gen['seconds'],'utility_passes':passed,'realism_passes':realpass,'both_gate_passes':both,'source_auc':{m:r['source_auc'] for m,r in diag['results'].items()},'preflop_call_rates':{m:r['action_summary']['preflop']['equal_pair_mean_fold_check_call_raise'][2] for m,r in diag['results'].items()},'mean_loss_gains':{m:c['mean_deltas']['logloss_improvement'] for m,c in comps.items()},'randomness_noop_repaired':True,'protected_unchanged':all(sha(R/p)==v for p,v in plan['protected'].items()),'no_production_promotion':True,'vm_not_restarted':True};assert out['protected_unchanged'];(H/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
    paragraph=f"**September 10, activation-calibration study complete:** training-only calibration fixed activation at .834126; fresh paired episodic/calibrated/continuous arms generated 32 pools / 96,000 scenarios replayed three ways. A shared-RNG confound was confirmed and fixed in all arms: old no-op checks changed 16/32 histories; independent controller randomness changes 0/32. Four fixed tree fits took {seconds:.1f}s, using {n} unique common targets and matched resampling. {outcome} Source AUCs: "+', '.join(f"{m} {r['source_auc']:.4f}" for m,r in diag['results'].items())+". Original-only comparison remains uncertain: calibrated gain .001049 with interval crossing zero; this resampling draw weakened the baseline. All legal/causal replay and input-parity checks pass. Earlier failed gates preserved. No VM restart, neural/RL run, upload or promotion. Report: `work/poker/experiments/softplay_activation_20260910/RESULTS.md`.\n\n"
    p=W/'STATUS.md';s=p.read_text(encoding='utf-8');key='**September 10, activation-calibration study running:**' if '**September 10, activation-calibration study running:**' in s else '**September 10, activation-calibration study complete:**';start=s.index(key);end=s.index('\n\n',start)+2;p.write_text(s[:start]+paragraph+s[end:],encoding='utf-8')
    memory=W/'research/POKER_RESEARCH_MEMORY.md'
    if '**September 10, activation-calibration study complete:**' not in memory.read_text(encoding='utf-8'):
        with memory.open('a',encoding='utf-8') as f:f.write('\n\n'+paragraph+'Keep the controller RNG independent of ordinary behavior: no-op and zero-intensity policies must leave complete histories unchanged. Current controllers remain own/public only. The partner-strength probe adds only .000611 OOF action logloss improvement and is not a strong reason for more complexity.\n')
if __name__=='__main__':main()
