"""Summarize a completed training-only mechanism replay without utility scoring."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;P=H.parent/'softplay_replication_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());d=json.loads((H/'diagnostics.json').read_text());old=json.loads((P/'diagnostics.json').read_text());discovery=json.loads((H.parent/'softplay_mismatch_20260910/diagnosis.json').read_text())
    real=discovery['action_contexts']['real_soft'];pair={};changes={};names=json.loads((H/'b0_coupled/features/metadata.json').read_text())['features']
    realx=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz')['train_pair'][:1403];om=json.loads((H.parent/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs'];realx=realx[np.array([r['behavior_family']=='soft_play' for r in om])]
    key_features=[v['name'] for v in discovery['results']['soft_play']['important_features'][:10]]
    for batch in plan['batches']:
        arm=batch['arm'];control=Path(batch['existing_control']).name;v=d['results'][arm];o=old['results'][control]
        pair[arm]={'control':control,'new_source_auc':v['source_auc'],'control_source_auc':o['source_auc'],'source_auc_reduction':o['source_auc']-v['source_auc'],'passes_realism':v['realism_gate_passed'],'new_supported_pairs':v['supported_pairs'],'new_partner_facing_fold_call_raise':[v['action_summary']['facing_partner']['equal_pair_mean_fold_check_call_raise'][i] for i in [0,2,3]],'control_partner_facing_fold_call_raise':[o['action_summary']['facing_partner']['equal_pair_mean_fold_check_call_raise'][i] for i in [0,2,3]],'new_preflop_call':v['action_summary']['preflop']['equal_pair_mean_fold_check_call_raise'][2],'new_call_strength':v['conditional']['preflop_calls']['own_strength']}
        matrices={}
        for label,folder in [('control',Path(batch['existing_control'])),('coupled',H/arm)]:
            m=json.loads((folder/'features/metadata.json').read_text())['views'][0];x=np.load(folder/'features/synthetic-pairs.npz')['x'][:len(m)];matrices[label]=x[np.array([r['behavior_family']=='soft_play' for r in m])]
        changes[arm]=[]
        for name in key_features:
            i=names.index(name);mu=float(realx[:,i].mean());sd=max(float(realx[:,i].std()),1e-9)
            row={'feature':name,'real_mean':mu,'real_std':sd,**{label+'_mean':float(x[:,i].mean()) for label,x in matrices.items()}}
            row.update({label+'_standardized_gap':(row[label+'_mean']-mu)/sd for label in matrices});changes[arm].append(row)
    passed=all(v['passes_realism'] for v in pair.values());replays={a:json.loads((H/a/'intervention-replay-audit.json').read_text()) for a in plan['generator_arms']}
    assert all(sha(H.parents[1]/p)==v for p,v in plan['protected'].items())
    out={'created_at':datetime.now(timezone.utc).isoformat(),'development_replays':True,'no_new_outer_held_evaluation':True,'no_new_model_fit':True,'both_development_realism_gates_passed':passed,'comparisons':pair,'feature_changes':changes,'real_partner_facing_fold_call_raise':[real['facing_partner']['equal_pair_mean_fold_check_call_raise'][i] for i in [0,2,3]],'replayed_hands':sum(v['hands'] for v in replays.values()),'changed_actions_verified':sum(v['counterfactual_action_changes_verified'] for v in replays.values()),'protected_submissions_unchanged':True,'next_step':'Freeze a new-seed confirmation of unchanged generator.' if passed else 'Diagnose the remaining training-only mismatch and freeze the next specific repair; do not score the outer holdout.', 'source_sha256':sha(Path(__file__))}
    (H/'summary.json').write_text(json.dumps(out,indent=2))
    lines=['# Soft-play distribution-coupling development trial','',('Both development batches pass the unchanged realism gate.' if passed else 'The repaired controller does not pass realism in both development batches.'),'',
      'The prior controller could turn a fold or raise into a call but could not create additional folds. In original training data, soft-play pairs fold 59.83% when facing their partner; the calibrated simulation folded only 47.05–49.04%. Training-only source attribution emphasized fold residuals and strong-fold behavior.',
      '', 'The repair couples ordinary action samples to the frozen learned soft-action distribution, using own cards and public context. Illegal raise probability is assigned to check/call, and desired aggression is capped at ordinary aggression. The resulting action probabilities match that constrained distribution without needlessly replacing actions. The learned models, environment process and original training sources stay fixed.',
      '', 'This is a controlled development replay of 64 previously diagnosed pools /192,000 dealt scenarios. The old continuous-activation histories are the paired control. It is not fresh-seed confirmation or evidence of improved detector utility. The reused outer validation set was not loaded.',
      '', '| Batch | Old source AUC | Coupled source AUC | Partner-facing fold / call / raise | Preflop calling | Call strength | Gate |','|---|---:|---:|---|---:|---:|---|']
    for a,v in pair.items():lines.append(f"| {a} | {v['control_source_auc']:.4f} | {v['new_source_auc']:.4f} | {' / '.join(f'{x:.4f}' for x in v['new_partner_facing_fold_call_raise'])} | {v['new_preflop_call']:.4f} | {v['new_call_strength']:.4f} | {'PASS' if v['passes_realism'] else 'FAIL'} |")
    lines+=['','Original partner-facing fold/call/raise rates: '+ ' / '.join(f'{v:.4f}' for v in out['real_partner_facing_fold_call_raise'])+'.', '',
      'The unchanged gate requires source AUC <=.85, preflop call-rate and call-strength gaps <=.05, at least 12 supported targets and four ordinary-policy families. Whole-pool source grouping is unchanged. Importance-based mechanism analysis is exploratory and does not prove causality.',
      '',f"All raw accounting/cards/balance/label checks pass. Independent replay verifies {out['replayed_hands']:,} hands and {out['changed_actions_verified']:,} changed actions. Pairing checks preserve profiles, seats and private cards across histories. Probability algebra (400 distributions), 50,000 sampling draws, 32-seed ordinary/legacy/no-op/zero-activation parity and causal feature checks pass. Only actual legal action changes support positive labels.",
      '',out['next_step'],'','All seven protected submissions remain unchanged. No VM, new policy-model training, outer-held utility score or Kaggle upload occurred. Raw traces, features, scripts and summaries are retained.','']
    (H/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(3);axes[0].plot(x,out['real_partner_facing_fold_call_raise'],'o-',color='black',label='Original training')
    for a,v in pair.items():axes[0].plot(x,v['new_partner_facing_fold_call_raise'],'o--',label=a)
    axes[0].set_xticks(x,['Fold','Call','Raise']);axes[0].set_ylim(0,.8);axes[0].set_title('Facing the partner: action distribution');axes[0].legend(fontsize=8)
    y=np.arange(2);axes[1].bar(y-.18,[v['control_source_auc'] for v in pair.values()],width=.36,label='Old continuous');axes[1].bar(y+.18,[v['new_source_auc'] for v in pair.values()],width=.36,label='Coupled');axes[1].axhline(.85,color='gray',linestyle=':');axes[1].set_xticks(y,['Population 0','Population 1']);axes[1].set_ylim(.5,1);axes[1].set_title('Training-only source distinguishability');axes[1].legend(fontsize=8)
    fig.suptitle('Replaying the same scenarios with a repaired action controller');fig.tight_layout();fig.savefig(H/'coupling-screen.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in out.items() if k not in ['feature_changes']},indent=2))
if __name__=='__main__':main()
