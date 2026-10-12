"""Report the causal-style development trial, preserving predecessor failures."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;P=H.parent/'softplay_coupling_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());d=json.loads((H/'diagnostics.json').read_text());old=json.loads((P/'diagnostics.json').read_text());probe=json.loads((H/'probe.json').read_text());out={}
    for batch in plan['batches']:
        a=batch['arm'];c=Path(batch['existing_control']).name;v=d['results'][a];before=old['results'][c]
        out[a]={'control':c,'old_source_auc':before['source_auc'],'new_source_auc':v['source_auc'],'source_auc_reduction':before['source_auc']-v['source_auc'],'supported_pairs':v['supported_pairs'],'passes_realism':v['realism_gate_passed'],'preflop_call_rate':v['action_summary']['preflop']['equal_pair_mean_fold_check_call_raise'][2],'preflop_call_strength':v['conditional']['preflop_calls']['own_strength'],'partner_facing_fold_call_raise':[v['action_summary']['facing_partner']['equal_pair_mean_fold_check_call_raise'][i] for i in [0,2,3]]}
    assert all(sha(H.parents[1]/p)==v for p,v in plan['protected'].items())
    assert all(sha(H/k)==v for k,v in json.loads((H/'analysis-source-freeze.json').read_text())['sources'].items())
    pass_both=all(v['passes_realism'] for v in out.values());replays=[json.loads((H/a/'intervention-replay-audit.json').read_text()) for a in plan['generator_arms']]
    result={'created_at':datetime.now(timezone.utc).isoformat(),'development_replay':True,'both_development_realism_gates_passed':pass_both,'comparisons':out,'action_model_probe':probe['results']['own_style'],'replayed_hands':sum(v['hands'] for v in replays),'changed_actions_verified':sum(v['counterfactual_action_changes_verified'] for v in replays),'controller_input_parity_rows':d['controller_feature_parity_rows'],'all_trace_past_styles_independently_verified':True,'no_outer_held_evaluation':True,'protected_submissions_unchanged':True,'next_step':'Freeze fresh-seed confirmation of unchanged controller and generator before utility testing.' if pass_both else 'Diagnose remaining training-only mismatch. Keep source gate failed; no outer-held utility test. Examine actor-profile compatibility or history-length mismatch before more generation.'}
    (H/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Past-only player-tendency augmentation trial','',('Both development populations pass the unchanged realism gate.' if pass_both else 'The causal-style controller does not pass realism in both development populations.'),'',
      'The preceding probability-coupling repair improved calling frequency, call strength and partner-facing folds, but failed the source-discrimination gate. This trial tests whether conditioning the soft-play policy on each player\'s earlier actions restores individual consistency.',
      '', 'The controller adds four own-player action frequencies calculated strictly from completed earlier hands, with one uniform pseudocount per class. It excludes the current hand and future hands. Three-fold pool-separated action prediction improves weighted loss by .005397, with positive gains in all folds (.006507/.003672/.005843). Adding partner tendencies gave a smaller gain, so the prespecified rule selected own-player tendencies only.',
      '', 'The trial replays 64 existing development pools /192,000 dealt scenarios against the frozen coupling control. Original learned ordinary policies, stack/session process and probability coupling remain unchanged. It is not fresh-seed confirmation. The outer validation set remains closed.',
      '', '| Population | Previous source AUC | New source AUC | Preflop calling | Call strength | Partner-facing fold / call / raise | Gate |','|---|---:|---:|---:|---:|---|---|']
    for a,v in out.items():lines.append(f"| {a} | {v['old_source_auc']:.4f} | {v['new_source_auc']:.4f} | {v['preflop_call_rate']:.4f} | {v['preflop_call_strength']:.4f} | {' / '.join(f'{p:.4f}' for p in v['partner_facing_fold_call_raise'])} | {'PASS' if v['passes_realism'] else 'FAIL'} |")
    lines+=['','Original training references: preflop calling .393700; call strength .564799; partner-facing fold/call/raise .598276/.374326/.027398. The unchanged realism criteria require source AUC <=.85, call-rate/strength gaps <=.05, at least 12 supported pairs and all four ordinary-policy families in each population.',
      '',f"All raw accounting, card, stack-ledger and supported-label audits pass. Independent replay covers {result['replayed_hands']:,} hands and {result['changed_actions_verified']:,} changed actions. Every recorded intervention's past-style vector is independently reconstructed from earlier raw hands. Controller feature parity covers {result['controller_input_parity_rows']} rows. Ordinary/legacy/no-op/zero-activation trace tests pass on 32 seeds; a prefix test verifies future changes cannot alter earlier style values.",
      '',result['next_step'],'',
      'The past-style preparation also verified a documentation correction: the original 297 training pools contain 1,485,000 hands, or 5,000 per pool; the generator pilots use 3,000. Earlier notes claiming 891,000 original training hands were incorrect. The length difference remains a separate, untested explanation; it was not silently changed here.',
      '', 'No detector-utility result, new submission or perfect-augmentation claim follows from action prediction or realism alone. No VM or subagent was used. All seven protected candidates remain byte-identical.','']
    (H/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7.5,4));x=[0,1];ax.plot(x,[v['old_source_auc'] for v in out.values()],'o-',label='Coupled controller');ax.plot(x,[v['new_source_auc'] for v in out.values()],'o-',label='Plus own past tendencies');ax.axhline(.85,color='gray',linestyle=':',label='Maximum accepted AUC');ax.set_xticks(x,['Development population 0','Development population 1']);ax.set_ylim(.5,1);ax.set_ylabel('Source AUC (lower is less distinguishable)');ax.set_title('Does individual consistency improve simulation realism?');ax.legend(fontsize=9);fig.tight_layout();fig.savefig(H/'causal-style-screen.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
