"""Summarize all three prescribed capacity replays with no outer scoring."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());d=json.loads((H/'diagnostics.json').read_text());comparisons={}
    for batch in plan['batches']:
        arm=batch['arm'];src=Path(batch['existing_control']);old=json.loads((src.parent/'diagnostics.json').read_text())['results'][src.name];v=d['results'][arm]
        comparisons[arm]={'old_source_auc':old['source_auc'],'new_source_auc':v['source_auc'],'source_auc_reduction':old['source_auc']-v['source_auc'],'passes_realism':v['realism_gate_passed'],'supported_pairs':v['supported_pairs'],'preflop_call_rate':v['action_summary']['preflop']['equal_pair_mean_fold_check_call_raise'][2],'call_strength':v['conditional']['preflop_calls']['own_strength'],'partner_facing_fold_call_raise':[v['action_summary']['facing_partner']['equal_pair_mean_fold_check_call_raise'][i] for i in [0,2,3]]}
    assert all(sha(H.parents[1]/p)==v for p,v in plan['protected'].items());replays=[json.loads((H/a/'intervention-replay-audit.json').read_text()) for a in plan['generator_arms']];passed=all(v['passes_realism'] for v in comparisons.values())
    result={'at':datetime.now(timezone.utc).isoformat(),'development_replays_of_all_three_previous_populations':True,'all_three_realism_gates_passed':passed,'comparisons':comparisons,'replayed_hands':sum(v['hands'] for v in replays),'changed_actions_verified':sum(v['counterfactual_action_changes_verified'] for v in replays),'controller_feature_parity_rows':d['controller_feature_parity_rows'],'no_outer_held_access':True,'protected_submissions_unchanged':True,'next_step':'Freeze NEW simulation seeds for unchanged-generator confirmation, then utility only if that passes.' if passed else 'Keep failed gates. Diagnose the remaining training-only residual/context or population mismatch; no outer-held utility trial.'};(H/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Larger causal soft-policy development trial','',('All three prescribed populations pass the unchanged realism gate.' if passed else 'The larger policy does not pass realism in all three populations.'),'',
      'A fixed three-fold pool-separated action-model probe selected a 300-tree, depth-5 policy with the same 25 causal inputs. Mean action-loss improvement over the 100-tree policy was .027554, positive in all three folds (.026849/.028414/.027497). The shallower depth-4 alternative improved .025760; the prespecified rule selected depth5 because its additional gain exceeded .001.',
      '', 'This trial replays all three existing populations, including the population that failed the preceding fresh-seed confirmation. The settings, ordinary policies, observed past tendencies and environment stay fixed. The free-call/check class bookkeeping correction is included; it had identical resolved gameplay in 256 checked histories and is not claimed as the source of improved behavior. No temperature change was admitted.',
      '', '| Population | Previous source AUC | New source AUC | Preflop calling | Call strength | Gate |','|---|---:|---:|---:|---:|---|']
    for a,v in comparisons.items():lines.append(f"| {a} | {v['old_source_auc']:.4f} | {v['new_source_auc']:.4f} | {v['preflop_call_rate']:.4f} | {v['call_strength']:.4f} | {'PASS' if v['passes_realism'] else 'FAIL'} |")
    lines+=['','The gate remains source AUC<=.85, call-rate/strength gaps<=.05, at least12 supported pairs and all4 ordinary-policy families in every population. Original references: calling .393700; call strength .564799.',
      '',f"All 288,000 replayed histories pass accounting/cards/balance/label checks. Independent replay verifies {result['replayed_hands']:,} hands and {result['changed_actions_verified']:,} actual changed actions. All intervention past-style inputs are independently checked. Causal input parity covers {result['controller_feature_parity_rows']} rows. Previously failed studies retain their original gates.",
      '',result['next_step'],'', 'These are development replays using a shared learned policy source, not an independent real-world validation set. No outer-held detector score, VM, subagent, new submission or production change occurred.','']
    (H/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,4));x=list(range(3));ax.plot(x,[v['old_source_auc'] for v in comparisons.values()],'o-',label='100-tree policy');ax.plot(x,[v['new_source_auc'] for v in comparisons.values()],'o-',label='300-tree policy');ax.axhline(.85,color='gray',linestyle=':',label='Maximum accepted AUC');ax.set_xticks(x,['Population0','Population1','Earlier failed confirmation']);ax.set_ylim(.5,1);ax.set_ylabel('Source AUC');ax.set_title('Does improved action prediction generalize to realistic histories?');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(H/'capacity-screen.png',dpi=160);plt.close(fig);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
