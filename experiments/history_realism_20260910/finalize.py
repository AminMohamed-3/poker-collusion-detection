"""Verify frozen artifacts and report both successful realism and failed coverage."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent;R=H.parents[1]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def check(name):
    d=H/name;plan=read(d/'pilot-plan.json');generation=read(d/'pilot-generation.json')
    assert generation['plan_sha256']==sha(d/'pilot-plan.json')
    for file,value in plan['source_hashes'].items():assert sha(d/file)==value,file
    for file,value in read(d/'auditor-freeze.json').items():assert sha(H/file)==value,file
    for pool in generation['pools']:
        folder=d/'pilot'/f"pool-{pool['pool_id']:04d}"
        for file,value in pool['files'].items():assert sha(folder/file)==value,file
    coarse=read(d/'pilot-audit.json');pair=read(d/'pair-domain-audit.json');temporal=read(d/'temporal-audit.json');balance=read(d/'balance-audit.json');replay=read(d/'replay-audit.json')
    schedule=read(d/'seat-schedule-audit.json') if (d/'seat-schedule-audit.json').exists() else None
    cells=set();excluded=[]
    for pool in generation['pools']:
        folder=d/'pilot'/f"pool-{pool['pool_id']:04d}";labels=pl.read_csv(folder/'development_labels.csv');p=read(folder/'provenance.json')
        if pool['behavior']=='none' or labels['label'].sum()>0:cells.add((pool['policy_family'],pool['behavior']))
        if pool['behavior']!='none' and not labels['label'].sum():
            target=tuple(sorted(p['target']));lookup={tuple(sorted((r['player_1'],r['player_2']))):r['pair_id'] for r in labels.iter_rows(named=True)}
            excluded.append({'pool_id':pool['pool_id'],'policy_family':pool['policy_family'],'behavior':pool['behavior'],'no_eligible_positive_emitted':True,'no_eligible_evidence':p['known_positive_omitted_for_no_intervention']})
    required={(p,b) for p in range(4) for b in ['none','directed_transfer','soft_play','coordinated_isolation']}
    realism=bool(pair['auc']<=.8 and temporal['passed'] and not coarse['domain_discrepancy_flag'] and balance['continuation_gate_passed'] and balance['all_stack_transitions_verified'] and replay['all_serialized_actions_replayed_legally'] and (schedule and schedule['schedule_gate_passed']))
    result={'plan_sha256':sha(d/'pilot-plan.json'),'hands':coarse['generated_hands'],'labelled_pairs':coarse['labelled_pairs'],'positive_pairs':coarse['positives'],'hand_source_auc':coarse['domain_group_oof_auc'],'pair_source_auc':pair['auc'],'temporal_source_auc':temporal['gates']['all']['auc'],'attendance_source_auc':temporal['gates']['attendance']['auc'],'balance_source_auc':temporal['gates']['balance_process']['auc'],'preflop_end_fraction':coarse['preflop_ending_fraction_synthetic'],'exact_balance_continuation_fraction':balance['exact_balance_continuation_fraction'],'verified_stack_transitions':balance['transitions'],'emitted_cells':len(cells),'missing_cells':[{'policy_family':p,'behavior':b} for p,b in sorted(required-cells)],'excluded_targets':excluded,'coverage_passed':balance['coverage_passed'],'history_realism_gates_passed':realism,'all_gates_passed':realism and balance['coverage_passed'],'replayed_hands':replay['hands'],'replayed_actions':replay['actions'],'source_files_and_all_generated_payloads_verified':True}
    (d/'final-audit.json').write_text(json.dumps(result,indent=2));return result

results={name:check(name) for name in ['persistent_balance','persistent_sessions','session_confirmation']}
assert read(H/'persistent_sessions/pilot-plan.json')['source_hashes']==read(H/'session_confirmation/pilot-plan.json')['source_hashes']
for file,receipt in [('temporal_audit.py','temporal-auditor-freeze.json'),('audit_seat_schedule.py','seat-auditor-freeze.json')]:assert sha(H/file)==read(H/receipt)['sha256']
protected=read(H.parent/'data_expansion/augmentation/plan.json')['protected'];assert all(sha(R/p)==v for p,v in protected.items())

# Both final batches are retained, including all unsupported intended targets.
# Verify that combining them does not create identifier or full-payload copies.
fingerprints=[];hand_ids=[];player_ids=set();table_ids=set()
for version in ['persistent_sessions','session_confirmation']:
    for folder in sorted((H/version/'pilot').glob('pool-*')):
        h=pl.read_parquet(folder/'hands.parquet');s=pl.read_parquet(folder/'seats.parquet')
        hand_ids.extend(h['hand_id'].to_list());player_ids.update(s['player_id'].to_list());table_ids.update(h['table_id'].to_list())
        records=s.sort('hand_id','seat_no').group_by('hand_id',maintain_order=True).agg(pl.struct(pl.exclude('hand_id','player_id')).alias('payload')).join(h.select('hand_id','board_cards','big_blind'),on='hand_id')
        for r in records.iter_rows(named=True):fingerprints.append(hashlib.sha256(json.dumps([r['payload'],r['board_cards'],r['big_blind']],sort_keys=True).encode()).digest())
assert len(hand_ids)==len(set(hand_ids))==192000
assert len(fingerprints)==len(set(fingerprints))
assert len(player_ids)==1920 and len(table_ids)==64
summary={'completed_at':datetime.now(timezone.utc).isoformat(),'versions':results,'history_realism_confirmed_on_fresh_seeds':all(results[v]['history_realism_gates_passed'] for v in ['persistent_sessions','session_confirmation']),'all_prespecified_confirmation_gates_passed':results['session_confirmation']['all_gates_passed'],'large_scale_expansion_admitted':False,'rl_training_started':False,'original_held_or_evaluation_data_used':False,'production_submissions_unchanged':True,'protected_sha256':protected,'final_recipe_both_batches':{'hands':192000,'pools':64,'players':1920,'unique_full_payloads':len(set(fingerprints)),'pairs':sum(results[v]['labelled_pairs'] for v in ['persistent_sessions','session_confirmation']),'positives':sum(results[v]['positive_pairs'] for v in ['persistent_sessions','session_confirmation'])},'limits':'Original calibration pools reused; the confirmation changes synthetic seeds, not the original population or policy source. Detectable residual domain differences remain. No detector-utility result follows from these audits.'}
(H/'summary.json').write_text(json.dumps(summary,indent=2))

old=read(H.parent/'data_expansion/profile_repair/pair-domain-audit.json')['auc']
names=['Previous profile sampler','Persistent balances','Balances + sessions','Fresh-seed confirmation']
values=[old]+[results[n]['pair_source_auc'] for n in results]
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,ax=plt.subplots(figsize=(9,4.4));bars=ax.barh(names,values,color=['#9299a5','#6b86a7','#24806d','#176450'])
    ax.invert_yaxis();ax.set_xlim(.5,1);ax.axvline(.8,color='#a3452f',linestyle='--',linewidth=1.4,label='Preset ceiling (.8)')
    for bar,v in zip(bars,values):ax.text(v+.006,bar.get_y()+bar.get_height()/2,f'{v:.3f}',va='center',fontweight='bold')
    ax.set_xlabel('Pair-history source AUC · lower is better; .5 is chance')
    ax.set_title('Persistent balances and sessions remove the dominant history artifacts',loc='left',pad=16,fontsize=13)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='lower right',frameon=False)
    fig.tight_layout();fig.savefig(H/'history-realism.png',dpi=180);plt.close(fig)
except ImportError:
    pass

lines=['# History realism repair — September 10, 2026','','The two identified history defects are repaired, and the source-realism checks pass again on fresh generated seeds. The broader expansion gate remains unpassed: one policy/behavior cell has no supported positive in the confirmation batch. This distinction is preserved rather than changing the criteria.','','## What was wrong','','1. **Balances were independent between hands.** In original training histories,96.797% of next starting stacks equal the previous balance plus that hand’s net result. The earlier simulator instead drew a fresh stack from a player-specific distribution. That preserved single-hand marginals but destroyed serial dependence and changed pair-history extremes.','2. **Seating was independent between hands.** Original players mostly continue in the same physical seat for a session. Redrawing six players every hand preserved rough shared-hand counts but destroyed run lengths and absence gaps. A whole-player temporal audit detected that perfectly,even after balances were repaired.','','## Repairs','','Carry forward the rules engine’s exact settled balance. Fit only from the297original training pools: reset below40or above250big blinds,plus a1.5024%spontaneous reset probability; reset values80/100/120/150BB follow their empirical frequencies. Initial stacks are sampled jointly with the training-player style prototype. No original hand,card deal or outcome is replayed as generated data.','','For seating,fit independent seat-exit probability3.5093%and whole-table refresh probability1.5266%. The fitted turnover-count distribution has total-variation distance.000274from the observed training counts. Existing seats persist otherwise,and replacement players come from those previously absent. Button positions are independently sampled as observed in this organizer dataset; this is not a claim about real-world dealer rotation. Engine input order and serialized physical seats are mapped consistently.','','The four related ordinary-action models,intervention rules,outer-training boundary,full1,217-feature pair audit and thresholds remain fixed. No stack features were removed to make the source audit pass.','','## Results','','| Version | Hand source AUC | Pair source AUC | Whole-player temporal AUC | Balance continuation | Emitted cells |','|---|---:|---:|---:|---:|---:|',f'| Previous profile sampler | .6190 | {old:.4f} | Not measured | Independent redraw |16/16 |']
for name,title in zip(results,['Persistent balances','Balances + sessions','Fresh-seed confirmation']):
    r=results[name];lines.append(f"| {title} | {r['hand_source_auc']:.4f} | {r['pair_source_auc']:.4f} | {r['temporal_source_auc']:.4f} | {100*r['exact_balance_continuation_fraction']:.3f}% | {r['emitted_cells']}/16 |")
lines+=['','Source AUC is the ability to distinguish original from generated data;closer to.5is better for this diagnostic. Each preset ceiling is.8. The temporal audit includes attendance gaps,stack autocorrelation,reset rates and outcome summaries. It groups entire player pools between classifier folds. Calibration and diagnostic original pools are reused;these are not untouched original-data estimates.','','![Pair-history source discrimination](history-realism.png)','','## Confirmation and validity','','The second session batch uses exactly the same generator,ordinary-policy models and fitted parameters as the first. Only random seeds and identifier namespaces change. Both96,000-hand batches are retained. Across them there are192,000unique full hand payloads,64pools,1,920players and no identifier collisions. They share a policy source and are not1,920independent learned policies.','','Every serialized inter-appearance balance transition and seat-change ledger is checked. Both batches pass card/accounting,reset,session-frequency,hand-domain and whole-player history checks. Independent game replay covers640hands and5,933actions across the two final batches. Seventy-two physical-seat remappings preserve all cards,actions and payouts. Existing submissions remain byte-identical.','','## Remaining limitation','','The fresh-seed batch fails the unchanged16-cell positive-coverage requirement. Missing cells: '+json.dumps(results['session_confirmation']['missing_cells'])+'. Unsupported intended targets stay excluded;no evidence was invented or threshold relaxed. The two-batch union has broader coverage,but that does not turn the failed per-batch confirmation criterion into a pass.','','Residual source differences remain,especially action-surprise and outcome summaries. Passing the history-realism diagnostics does not establish real-world equivalence,independent external diversity or improved detection. No original-held utility trial,10×expansion,RL run,VM restart or Kaggle upload was performed. The identified history-artifact mini-goal is supported by the replicated results;labelled behavior coverage and downstream utility remain separate work.','','[Machine-readable summary](summary.json) · [Transition diagnosis](stack-transitions.json) · [Attendance diagnosis](attendance-diagnosis.json) · [Confirmation plan](confirmation-plan.json) · [Confirmation audit](session_confirmation/final-audit.json)','']
# Keep prose readable while leaving numeric values unchanged.
text='\n'.join(lines)
import re
text=re.sub(r'(?<=[A-Za-z%])(?=\d)', ' ',text)
text=re.sub(r'(?<=\d)(?=[A-Za-z])',' ',text)
text=text.replace('pools;these','pools; these').replace('diagnostic.Each','diagnostic. Each').replace(',even',', even').replace(',plus',', plus').replace(',and',', and').replace(',outer',', outer').replace(',full',', full').replace(',card',', card').replace(',ordinary',', ordinary').replace(',stack',', stack').replace(',reset',', reset').replace(',actions',', actions').replace(',1,',', 1,').replace(',64',', 64').replace(';no','; no').replace(';closer','; closer').replace('.5is','.5 is').replace('.8. ','.8. ')
text=re.sub(r',(?=[A-Za-z])',', ',text)
text=re.sub(r';(?=\S)','; ',text)
text=re.sub(r'(?<=%)(?=[A-Za-z])',' ',text)
text=re.sub(r'(?<=[A-Za-z])\.(?=\d)',' .',text)
text=text.replace('histories,96.797%','histories, 96.797%').replace('trial,10×expansion','trial, 10× expansion')
missing=results['session_confirmation']['missing_cells']
if missing==[{'policy_family':0,'behavior':'soft_play'}]:
    text=text.replace('Missing cells: '+json.dumps(missing)+'.','The missing combination is policy family 0 with soft play.')
(H/'RESULTS.md').write_text(text,encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['versions','protected_sha256']},indent=2))
