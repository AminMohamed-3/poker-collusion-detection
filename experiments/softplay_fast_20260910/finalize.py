"""Record the complete fast screen, its failures and immutable local backup."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parents[1];W=R.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());ev=json.loads((H/'trial/evaluation.json').read_text());realism=json.loads((H/'realism.json').read_text());audit=json.loads((H/'repaired/pilot-audit.json').read_text());replay=json.loads((H/'repaired/intervention-replay-audit.json').read_text());rv=json.loads((H/'real-views.json').read_text());mismatch=json.loads((H/'remaining-mismatch.json').read_text())['results']
    receipts=[json.loads((H/'trial'/a/'receipt.json').read_text()) for a in plan['arms']];seconds=sum(r['seconds'] for r in receipts)
    names={'original':'Original only','resampled':'Matched resampling','old_soft':'Old soft-play simulation','repaired_soft':'Repaired soft-play simulation','real_history_views':'Extra real-history views'}
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.8),layout='constrained');fig.patch.set_facecolor('#fbfaf6')
    colors=['#263b53','#b4bbc3','#087f8c']
    x=np.arange(2);width=.23
    for i,(label,key) in enumerate([('Original training','original'),('Old simulation','old_synthetic'),('Repaired simulation','repaired')]):
        vals=[realism['gaps'][c][key]*100 for c in ['preflop','postflop']]
        bars=axes[0].bar(x+(i-1)*width,vals,width,label=label,color=colors[i]);axes[0].bar_label(bars,fmt='%.1f%%',fontsize=9,padding=3)
    axes[0].set_xticks(x,['Preflop calls','Postflop checks']);axes[0].set_ylim(0,65);axes[0].set_ylabel('Equal-pair mean action rate (%)');axes[0].set_title('Action frequencies move closer');axes[0].legend(frameon=False,fontsize=9,loc='upper left')
    arms=['old_soft','repaired_soft','real_history_views'];ys=np.arange(len(arms));vals=[];lo=[];hi=[]
    for arm in arms:
        c=ev['comparisons_to_resampled'][arm];v=c['mean_deltas']['logloss_improvement'];a,b=c['pool_bootstrap_logloss_improvement_95pct_interval'];vals.append(v);lo.append(v-a);hi.append(b-v)
    axes[1].errorbar(vals,ys,xerr=np.array([lo,hi]),fmt='o',color='#087f8c',capsize=4,markersize=7);axes[1].axvline(0,color='#73818f',lw=1);axes[1].set_yticks(ys,[names[a] for a in arms]);axes[1].invert_yaxis();axes[1].set_xlabel('Loss reduction vs matched resampling → better');axes[1].set_title('Benefit is not yet reliable');axes[1].grid(axis='x',alpha=.15)
    fig.suptitle('Soft-play repair · fast CPU screen',fontsize=16,weight='bold');fig.savefig(H/'fast-screen.png',dpi=160);plt.close(fig)
    table=[]
    for arm in plan['arms']:
        rows=[next(r for r in ev['metrics'] if r['arm']==arm and r['scope']=='all' and r['history']==tag) for tag in ['full','shortened']]
        comp=ev['comparisons_to_resampled'].get(arm)
        delta='—' if comp is None else f"{comp['mean_deltas']['logloss_improvement']:+.6f}"
        gate='Control' if comp is None else 'Pass' if comp['passes_utility_screen'] else 'Fail'
        table.append(f"| {names[arm]} | {rows[0]['log_loss']:.6f} | {rows[1]['log_loss']:.6f} | {delta} | {gate} |")
    new=ev['comparisons_to_resampled']['repaired_soft'];ci=new['pool_bootstrap_logloss_improvement_95pct_interval'];halves=new['pool_half_mean_logloss_improvement'];fallbacks=[len(r['full_history_fallback_pairs']) for r in rv['views']]
    report=f'''# Soft-play realism and a fast augmentation screen

The rule repair improves the targeted action frequencies and average detector loss, but **fails both the overall realism gate and the validation-consistency gate**. Retain the code and the evidence; do not scale generation or start neural training on this result. Five fixed LightGBM detector fits took **{seconds:.1f} seconds total** locally. No VM training was needed.

![Fast screen](fast-screen.png)

## What changed

The old controller mostly suppressed raises and checked in heads-up play. The new controller also permits low-pressure preflop continuation while a partner remains active, inexpensive calls facing a partner, and selected multiway checks. Four fixed public-state variants are crossed with four related ordinary-policy families. The existing episodic and probabilistic activation remains. Soft-play intervention decisions do not inspect partner cards or detector predictions. They use no validation labels or future cards.

The training-only diagnosis and repair produce these equal-pair action rates:

| Behavior | Original training | Old simulation | Repaired simulation |
|---|---:|---:|---:|
| Preflop call | 39.37% | 19.18% | 31.49% |
| Postflop check | 50.29% | 34.47% | 41.89% |

Calling/checking gaps shrink, but grouped soft-play source-classifier AUC remains **{realism['source_auc']:.6f}**, versus {realism['old_source_auc']:.6f} before; the predeclared ceiling is .85. This test has 92 original, 10 old synthetic and 12 repaired synthetic positive pairs. The batches use different seeds; this is not a paired causal ablation of generator code. The small AUC change is not convincing evidence of improved overall distribution realism.

## Fast utility experiment

One fixed incumbent tree recipe, 1,217 features, one fixed seed, no hyperparameter or epoch search. All arms start with the same 1,403 original pairs across 297 pools and their three existing views: 4,209 rows. Each augmented arm adds 180 pair draws / 540 rows with identical class counts: 120 none, 12 transfer, 40 soft play and 8 isolation. Sample weights and tree parameters are fixed.

Old/repaired simulation replaces **only the extra soft-play rows**. All other added rows match the resampling control exactly. Old simulation has 10 unique soft pairs; repaired has 12. Cycling those pairs to 40 draws changes training weights, not the number of independent examples. Real-history augmentation creates three random hash-selected views (.50/.65/.85) of 180 unique original training pairs; positive views with no listed evidence fall back to their full history. Fallback counts are {fallbacks}. Existing cross-fitted conditional references and leave-current-hand-out public style baselines are reused.

The five models and predictions were frozen before one evaluator read validation labels. Validation uses the existing 100 disjoint outer pools / 457 pairs, full and shortened histories, plus two predeclared label-independent pool halves. **Those pools have been reused in earlier research.** Neither the halves nor the history views are independent datasets or a substitute for omitted-family confirmation.

| Training arm | Full log loss ↓ | Shortened log loss ↓ | Mean loss reduction vs resampling ↑ | Utility gate |
|---|---:|---:|---:|---|
{chr(10).join(table)}

The repaired arm improves average loss by **{new['mean_deltas']['logloss_improvement']:.6f}**, mean pair AP by {new['mean_deltas']['pair_ap']:+.6f}, and mean AP at hypothetical 1% prevalence by {new['mean_deltas']['sparse_ap']:+.6f}. The gain is uneven: loss improvements in the two pool halves are **{halves[0]:+.6f}** and **{halves[1]:+.6f}**. The 1,000-replicate paired pool-bootstrap 95% interval is **[{ci[0]:+.6f}, {ci[1]:+.6f}]**. That fails the frozen requirement for nonnegative improvement in both halves. Class-specific soft-play AP is essentially unchanged. Most AP values are already close to one on this development split, so these small gains cannot predict the private leaderboard.

Old soft-play augmentation hurts both histories; its bootstrap loss interval is entirely negative. Extra views of real histories fail too: nearly unchanged mean loss with worse pair ranking. No additional fit, blend, seed selection or threshold change was attempted after scoring. Four tiny source-classification diagnostic fits are separate from the five downstream detector fits; there were no neural or RL runs.

## What remains wrong

A post-screen, training-only descriptive check suggests that matching marginal action rates is insufficient. Mean own-card strength among preflop calls is **{mismatch['real_soft']['preflop_call']['own_strength']:.3f}** in original soft-play pairs, **{mismatch['old_soft']['preflop_call']['own_strength']:.3f}** in old simulation, and **{mismatch['repaired_soft']['preflop_call']['own_strength']:.3f}** after repair. The repair increases calls on weaker cards and widens this particular gap. Mean card strength for postflop checks is {mismatch['real_soft']['postflop_check']['own_strength']:.3f} original versus {mismatch['repaired_soft']['postflop_check']['own_strength']:.3f} repaired. These are the existing feature's strength scores, not equity probabilities or new target labels.

The next narrow hypothesis is **conditional soft-play behavior**: match actions jointly with own-card strength, price, street and who applied pressure, while preserving normal-policy variability and episodic coordination. This should first be checked on training-pool-separated action data. A future generator comparison should use paired seeds and hold out entire controller variants. A density-matching or learned controller must not be rewarded merely for defeating the source classifier, and original held pools must remain outside policy learning. No such follow-up training was started in this screen.

## Data integrity and resource use

- {audit['hands']:,} unique full hand payloads, 16 pools, 480 synthetic players, {audit['pairs']} eligible pairs and {audit['positive_pairs']} supported soft-play targets. All four ordinary-policy families represented; four target cells were omitted rather than assigned unsupported positives. This is not broad behavior-family coverage.
- All serialized accounting, card uniqueness and {audit['transitions']:,} balance transitions pass. Exact continuation is {audit['exact_balance_continuation']:.4%}, versus 96.7974% originally. This is a diagnostic match, not a universal assumption about hidden data.
- Independent replay checks all {replay['counterfactual_action_changes_verified']} changed-action traces plus a random sample: {replay['hands']} hands, {replay['actions']:,} legal actions, exact preaction state and final net chips. Each traced change is compared with the resolved ordinary action at the same state.
- 288 intervention tests establish invariance to partner private cards, no effect when the partner is absent, and no effect at zero intensity.
- {audit['prototype_sources']} unique original training-player prototypes among 480 assignments; maximum reuse {audit['maximum_prototype_reuse']}. Four ordinary policies are still related models from one source.
- Source/calibration/reference pools remain the original 297 training pools. All seven protected submission CSV hashes are unchanged. No Kaggle upload or production promotion occurred.
- The reopened VM was idle. Vast accepted stopping instance 50489879 at **2026-09-10 16:05:56.896 UTC**; SSH then closed. Final provider stopped/billing state was not independently observed. Disk retained; current experiments are local and complete.

Reproducibility: `study-plan.json`, `trial/freeze.json`, `trial/evaluation.json`, `realism.json`, `remaining-mismatch.json`, `repaired/pilot-audit.json`, `repaired/intervention-replay-audit.json`, model receipts and raw pool files. All arms and failed gates are retained. No scheduler or automatic submission was created.
'''
    (H/'RESULTS.md').write_text(report,encoding='utf-8')
    summary={'completed_at':datetime.now(timezone.utc).isoformat(),'five_detector_fit_seconds':seconds,'realism_gate_passed':False,'utility_gate_passed':new['passes_utility_screen'],'repaired_loss_improvement':new['mean_deltas']['logloss_improvement'],'loss_interval':ci,'pool_half_loss_improvements':halves,'protected_unchanged':all(sha(R/p)==v for p,v in plan['protected'].items()),'no_production_promotion':True,'no_vm_training':True,'source_auc':realism['source_auc'],'next_lead':'Conditional card-strength/price/action realism; no follow-up fits begun.'}
    assert summary['protected_unchanged']
    (H/'summary.json').write_text(json.dumps(summary,indent=2))
    status=W/'STATUS.md';text=status.read_text(encoding='utf-8');marker='## Current work\n\n'
    paragraph=f"**September 10, fast soft-play screen complete:** local-only 48,000-hand pilot; five fixed tree detector fits took {seconds:.1f} seconds. Public-state soft-play rules increase preflop calling 19.18% → 31.49% (original 39.37%) and postflop checking 34.47% → 41.89% (original 50.29%). Full causal/legal replay verifies 673 changed actions across 440 hands. Overall soft-play source AUC remains .9846, failing the .85 ceiling. Repaired augmentation improves mean held loss .001284 and pair AP .000189 over matched resampling, but one fixed pool half worsens and the bootstrap interval crosses zero, so the consistency gate fails. Old simulation and extra real-history views also fail. No neural/RL run, scale-up, production promotion or upload. Remaining training-only lead: conditional hand strength/price/pressure realism; new calls are on weaker cards than original soft play. Seven protected CSVs unchanged. The idle, reopened VM accepted another stop request at 16:05:56.896 UTC, then closed SSH; final provider state not independently verified. Report: `work/poker/experiments/softplay_fast_20260910/RESULTS.md`.\n\n"
    assert marker in text and 'fast soft-play screen complete' not in text;status.write_text(text.replace(marker,marker+paragraph,1),encoding='utf-8')
    memory=W/'research/POKER_RESEARCH_MEMORY.md'
    with memory.open('a',encoding='utf-8') as f:f.write('\n\n'+paragraph+'The latest user explicitly asked for fast/simple screens before more models. Keep the five-arm frozen failure; do not treat its positive overall mean as a pass. Five downstream fits plus four small source-diagnostic fits ran locally. The RL fallback remains untested.\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
