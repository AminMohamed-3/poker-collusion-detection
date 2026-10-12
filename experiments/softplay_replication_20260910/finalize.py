"""Write an exhaustive concise report from immutable replication results."""
import os
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signed(v):return f'{v:+.6f}'
def main():
    plan=json.loads((H/'study-plan.json').read_text());freeze=json.loads((H/'trial/freeze.json').read_text());e=json.loads((H/'trial/evaluation.json').read_text());diag=json.loads((H/'diagnostics.json').read_text());supp=json.loads((H/'supplementary-audit.json').read_text())
    assert all(sha(H/k)==v for k,v in json.loads((H/'analysis-freeze.json').read_text())['sources'].items())
    assert all(sha(R/p)==v for p,v in plan['protected'].items())
    for a,h in plan['generator_plan_sha256'].items():
        assert sha(H/a/'pilot-plan.json')==h
        pp=json.loads((H/a/'pilot-plan.json').read_text());assert all(sha(H/a/p)==v for p,v in pp['source_hashes'].items())
    passed=[m for m,v in e['primary'].items() if v['passes_replication_gate']]
    verdict=('No activation setting passes the frozen replication gate.' if not passed else 'Replication gate passed for: '+', '.join(passed)+'.')
    lines=['# Fresh-seed soft-play augmentation replication','',verdict,'',
      'Two fresh 32-pool populations contain 192,000 distinct dealt scenarios, replayed under calibrated and continuous activation (384,000 histories). These are paired counterfactual histories and share the same learned policy source. Generator code, learned policies, environment parameters and calibrated activation probability 0.834125975 were unchanged.',
      '',f"Each synthetic arm adds {freeze['common_dose']} unique supported soft-play pairs, each with full/70%/40% views, to the original 4,209 training rows. All four ordinary-policy families remain represented. Original-only, three predetermined real-resampling draws and four synthetic arms were each fitted with three fresh model seeds: 24 small CPU tree fits in {e['fit_seconds']:.1f} seconds. No VM was started.",
      '', 'The primary estimate averages individual-model metrics over all three model seeds and both generated populations. It does not pick a seed, population, control or ensemble. All 24 models and predictions were frozen before loading validation labels.',
      '', '## Primary comparisons','',
      'Positive loss gain means lower multiclass logloss. Intervals resample whole validation pools while preserving all paired predictions; they are conditional on the fitted seeds and populations.',
      '', '| Activation | Baseline | Mean loss gain | 95% pool interval | Pair AP change | Sparse AP change | Utility gate |',
      '|---|---|---:|---|---:|---:|---|']
    for m,v in e['primary'].items():
        for b in ['original','mean_resampling']:
            c=v[b];d=c['mean_deltas'];ci=c['pool_bootstrap_logloss_improvement_95pct_interval']
            lines.append(f"| {m} | {b} | {signed(d['logloss_improvement'])} | [{signed(ci[0])}, {signed(ci[1])}] | {signed(d['pair_ap'])} | {signed(d['sparse_ap'])} | {'PASS' if c['utility_gate']['passed'] else 'FAIL'} |")
    lines+=['','The fixed utility gate requires mean loss gain at least .001, nonnegative mean Pair AP and sparse AP changes, no history Pair AP loss below -.001, nonnegative gain in both fixed pool halves, and positive mean loss gain in every generated population and every model seed, against both baselines. The combined gate also requires realism to pass in both fresh populations.','', '## Seed and population sensitivity','', '| Activation | Baseline | Population 0 gain | Population 1 gain | Seed 19087 gain | Seed 29501 gain | Seed 40709 gain | Failed checks |','|---|---|---:|---:|---:|---:|---:|---|']
    for m,v in e['primary'].items():
        for b in ['original','mean_resampling']:
            c=v[b];values=[x['mean_deltas']['logloss_improvement'] for x in c['by_batch'].values()]+[x['mean_deltas']['logloss_improvement'] for x in c['by_model_seed'].values()]
            failed=', '.join(k for k,val in c['utility_gate']['checks'].items() if not val) or 'none'
            lines.append('| '+m+' | '+b+' | '+' | '.join(map(signed,values))+' | '+failed+' |')
    lines+=['','## Resampling controls','', 'Controls are compared with original-only using the same model seeds. A negative gain means resampling made the baseline worse.','', '| Control | Mean loss gain versus original | Pair AP change | Sparse AP change |','|---|---:|---:|---:|']
    for c,v in e['resampling_vs_original'].items():
        d=v['mean_deltas'];lines.append(f"| {c} | {signed(d['logloss_improvement'])} | {signed(d['pair_ap'])} | {signed(d['sparse_ap'])} |")
    lines+=['','The continuous treatment\'s mean gain over original-only is almost matched by resampling real soft-play examples (.001038 versus .000998). This supports a reweighting explanation for much of the observed gain; it does not establish that synthetic histories add useful new information. Every primary pool-bootstrap interval crosses zero. Both activation settings also worsen the same predefined validation-pool half against original-only. These outcomes do not justify scaling this generator or spending on neural training.',
      '', '## Training-only realism','', '| Generated arm | Supported targets | Source AUC | Preflop call rate | Call-strength mean | Realism gate |','|---|---:|---:|---:|---:|---|']
    for a,v in diag['results'].items():
        lines.append(f"| {a} | {v['supported_pairs']} | {v['source_auc']:.4f} | {v['action_summary']['preflop']['equal_pair_mean_fold_check_call_raise'][2]:.4f} | {v['conditional']['preflop_calls']['own_strength']:.4f} | {'PASS' if v['realism_gate_passed'] else 'FAIL'} |")
    lines+=['','Realism uses the unchanged training-only criteria: pool-grouped source AUC <=.85, preflop calling and call-strength gaps <=.05, at least 12 supported targets, four policy families. Original training references are .393700 calling and .564799 call strength. Source distinguishability and downstream utility answer different questions.','', '## Audit and limitations','']
    audits={a:json.loads((H/a/'pilot-audit.json').read_text()) for a in plan['generator_arms']};replays={a:json.loads((H/a/'intervention-replay-audit.json').read_text()) for a in plan['generator_arms']}
    lines += [f"- All 384,000 serialized histories pass accounting, card, balance-ledger and supported-label checks. Independent replay checks {sum(v['hands'] for v in replays.values()):,} hands and {sum(v['counterfactual_action_changes_verified'] for v in replays.values()):,} changed actions, with exact before-action state and settlement agreement.",
      f"- Cards, seats, profiles and episode draws match across counterfactual arms for all 64 pools. The controller's before-action features match independently reconstructed features for {diag['controller_feature_parity_rows']} audited rows. No-op/zero-activation tests pass on 32 fresh seeds.",
      f"- All {supp['dealt_scenarios_checked']:,} checked deal payloads are distinct across the two populations after excluding identifiers. This does not make within-player histories independent or supply a new real-world policy source.",
      '- Learned policies, calibration and source diagnostics use the original 297 training pools. The original 100 validation pools / 457 pairs have been reused across research and are not an untouched holdout. No leaderboard feedback was used here.',
      '- Pool intervals do not include the full uncertainty from choosing a generator, model seeds or earlier research. Two populations and three seeds provide a sensitivity check, not a precise estimate of that distribution. Multiple exploratory comparisons are unadjusted.',
      '- Generated positive pairs require a real changed action. Every selected pair retains at least eight changed-action hands even in its 40% view; none loses all visible interventions. Original evidence annotations are capped, so zero annotations cannot be compared directly with exhaustive synthetic traces.',
      '- History thinning reuses the full-history player-style baselines and is not a complete regeneration of histories and all derived context.',
      '- Only soft-play pair classification was tested. Evidence ranking, omitted-behavior robustness and production admission were not tested. Related ordinary-policy families are not independent data sources.',
      '- All seven protected submission CSVs remain byte-identical. No Kaggle upload, neural fit, VM restart or production promotion occurred. Earlier failed experiments retain their original results.',
      '', '![Replication loss contrasts](replication-screen.png)','',
      '## Reproduction','', '`study-plan.json` fixes the design before generation. `analysis-freeze.json` hashes the evaluator before fits. `trial/freeze.json` records training arrays, selected targets, all three control draws and source hashes. `trial/evaluation.json` contains every per-run comparison, both pool halves, both histories and secondary fixed seed-ensemble results. All raw histories, causal traces, feature arrays, models and receipts are archived.','']
    (H/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharex=True)
    colors={'calibrated':'#2563a6','continuous':'#c77424'}
    for ax,b,title in zip(axes,['original','mean_resampling'],['Versus original-only','Versus mean of three resampling controls']):
        for i,(m,v) in enumerate(e['primary'].items()):
            c=v[b];gain=c['mean_deltas']['logloss_improvement'];lo,hi=c['pool_bootstrap_logloss_improvement_95pct_interval']
            ax.plot([lo,hi],[i,i],color=colors[m],linewidth=3);ax.scatter([gain],[i],s=75,color=colors[m],zorder=4)
            ax.scatter([x['mean_deltas']['logloss_improvement'] for x in c['by_model_seed'].values()],[i+.13]*3,marker='|',s=110,color=colors[m],alpha=.75)
        ax.axvline(0,color='#333333',linewidth=1);ax.axvline(.001,color='#999999',linestyle=':',label='Required mean gain .001')
        ax.set_yticks([0,1],['Calibrated','Continuous']);ax.set_ylim(1.5,-.5);ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.18);ax.set_xlabel('Mean reduction in multiclass logloss')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fresh-seed soft-play augmentation replication',fontsize=15,fontweight='bold')
    fig.text(.5,.025,'Dots: mean of 2 populations x 3 seeds. Bars: conditional 95% pool bootstrap. Ticks: individual seed means.\nReused 100 validation pools; these are exploratory uncertainty intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,.92]);fig.savefig(H/'replication-screen.png',dpi=170);plt.close(fig)
    result={'verdict':verdict,'replication_passes':passed,'primary':e['primary'],'diagnostics':diag['results'],'selected_targets_per_arm':freeze['common_dose'],'fit_count':e['fit_count'],'fit_seconds':e['fit_seconds'],'generation_seconds':json.loads((H/'generation-time.json').read_text())['seconds'],'supported_pairs_by_batch':diag['common_supported_pairs_by_batch'],'audit_hands':sum(v['hands'] for v in audits.values()),'replayed_hands':sum(v['hands'] for v in replays.values()),'changed_actions_verified':sum(v['counterfactual_action_changes_verified'] for v in replays.values()),'protected_submissions_unchanged':True,'production_admitted':False,'no_vm_used':True,'source_sha256':sha(__file__)}
    (H/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['primary','diagnostics']},indent=2))
if __name__=='__main__':main()
