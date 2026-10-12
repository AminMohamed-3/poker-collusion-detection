"""Create an isolated frozen pilot that carries forward engine-settled balances."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).resolve().parent;S=H.parent/'data_expansion/profile_repair';D=H/'persistent_balance'
assert not D.exists();D.mkdir()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

for name in ['generator.py','learned_policy.py','test_generator.py','models.joblib','fit.json','conditional_action_features.py','advanced_features.py','sampler.json','profiles.json','training-calibration.json']:
    shutil.copy2(S/name,D/name)
shutil.copy2(S/'training-calibration.json',H/'training-calibration.json')
shutil.copy2(H/'balance-process.json',D/'balance-process.json')
g=(D/'generator.py').read_text().replace("HERE.parent/'vendor'","HERE.parents[1]/'data_expansion'/'vendor'").replace("f'SP{pool_id:05d}H{hand_number:05d}'","f'SB{pool_id:05d}H{hand_number:05d}'").replace("f'SPPOOL{pool_id:05d}'","f'SBPOOL{pool_id:05d}'")
(D/'generator.py').write_text(g)
p=(S/'run_pilot.py').read_text().replace("f'SPP{pool_id:04d}","f'SBP{pool_id:04d}")
p=p.replace("PROFILES=json.loads((HERE/'profiles.json').read_text())['profiles']","PROFILES=json.loads((HERE/'profiles.json').read_text())['profiles']\nBALANCE=json.loads((HERE/'balance-process.json').read_text())")
p=p.replace("    stack_center={p:rng.uniform(70,170) for p in players}","    balances={p:round(BALANCE['first_stack_by_player'][prototypes[p]['source_player_id']]*blind) for p in players}\n    previously_seen=set(); resets=[]")
old="        stacks=[stack_draw(rng,blind,prototypes[p]['stack_bb_quantiles']) for p in sitting]"
new="""        for player in sitting:
            if player in previously_seen:
                boundary=not BALANCE['min_balance_bb']*blind<=balances[player]<=BALANCE['max_balance_bb']*blind
                spontaneous=(not boundary) and rng.random()<BALANCE['spontaneous_reset_probability']
                if boundary or spontaneous:
                    before=balances[player]
                    balances[player]=round(blind*rng.choices(BALANCE['reset_stack_bb'],BALANCE['reset_weights'])[0])
                    resets.append({'hand_number':h,'player_id':player,'before':before,'after':balances[player],'reason':'boundary' if boundary else 'spontaneous'})
        stacks=[balances[p] for p in sitting]"""
assert old in p;p=p.replace(old,new)
old='        hands.append(hand);seats.extend(ss);actions.extend(aa)'
new="""        hands.append(hand);seats.extend(ss);actions.extend(aa)
        for seat in ss:
            player=seat['player_id'];balances[player]=seat['starting_stack']+seat['net_chips'];previously_seen.add(player)
            assert balances[player]>=0"""
assert old in p;p=p.replace(old,new)
p=p.replace("    (dest/'interventions.json').write_text(json.dumps(traces))","    (dest/'interventions.json').write_text(json.dumps(traces))\n    (dest/'balance-resets.json').write_text(json.dumps(resets))")
p=p.replace('with ProcessPoolExecutor(max_workers=2)','with ProcessPoolExecutor(max_workers=4)')
(D/'run_pilot.py').write_text(p)
compile(g,str(D/'generator.py'),'exec');compile(p,str(D/'run_pilot.py'),'exec')
plan=json.loads((S/'pilot-plan.json').read_text())
plan['created_at']=datetime.now(timezone.utc).isoformat();plan['parent_plan_sha256']=sha(S/'pilot-plan.json');plan['workers']=4
for row in plan['pools']:row['seed']+=400000
plan['source_hashes']={n:sha(D/n) for n in [*plan['source_hashes'],'balance-process.json']}
plan['repair']='Carry engine-settled balances forward across appearances. Empirical40/250BB reset boundaries,1.5024% spontaneous reset,training-only reset/initial distributions. Preserve action policies,attendance,mechanisms,labelling rule and all1,217 audited pair features.'
plan['design']='32fresh pools,30players per pool,3000hands each,4policy families crossed with3coordination mechanisms plusnone;2pools per cell. Weighted attendance inherited unchanged; exact engine balances persist.'
plan['next_stage']='Legal/replay/reset-ledger/coverage audits and full-feature source audit before any expansion. No original-held utility trial or RL training in this mini-goal.'
plan['realism_gates']={'pair_domain_auc_max':.8,'hand_domain_auc_max':.8,'preflop_ending_absolute_gap_max':.10,'original_balance_continuation_fraction':.967974495464757,'balance_continuation_absolute_gap_max':.015,'minimum_emitted_cells':16}
(D/'pilot-plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps({'path':str(D),'plan_sha256':sha(D/'pilot-plan.json'),'hands':96000,'new_source_fits':False}))
