"""Preserve seats across hands, keeping the verified balance process unchanged."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).resolve().parent;S=H/'persistent_balance';D=H/'persistent_sessions'
assert not D.exists();D.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=['generator.py','learned_policy.py','test_generator.py','models.joblib','fit.json','conditional_action_features.py','advanced_features.py','sampler.json','profiles.json','training-calibration.json','balance-process.json']
for name in names:shutil.copy2(S/name,D/name)
shutil.copy2(H/'attendance-process.json',D/'attendance-process.json')
g=(D/'generator.py').read_text().replace("f'SB{pool_id:05d}H{hand_number:05d}'","f'SS{pool_id:05d}H{hand_number:05d}'").replace("f'SBPOOL{pool_id:05d}'","f'SSPOOL{pool_id:05d}'")
g=g.replace("target=None, family='none', intensity=.5, active_episode=True):","target=None, family='none', intensity=.5, active_episode=True, button_seat=5):")
g=g.replace("'button_seat':5,","'button_seat':button_seat,").replace("'seat_no':i,","'seat_no':(i+button_seat+1)%6,")
(D/'generator.py').write_text(g)
p=(S/'run_pilot.py').read_text().replace("f'SBP{pool_id:04d}","f'SSP{pool_id:04d}")
p=p.replace("BALANCE=json.loads((HERE/'balance-process.json').read_text())","BALANCE=json.loads((HERE/'balance-process.json').read_text())\nATTENDANCE=json.loads((HERE/'attendance-process.json').read_text())")
p=p.replace("    participation={p:math.exp(rng.gauss(0,SAMPLER['participation_log_sd'])) for p in players}","    seated=rng.sample(players,6); seat_events=[]")
old="""        sitting=sorted(players,key=lambda p:-math.log(max(1e-12,rng.random()))/participation[p])[:6]
        rng.shuffle(sitting)"""
new="""        before=seated.copy()
        mode='unchanged'
        if h:
            if rng.random()<ATTENDANCE['table_refresh_probability']:
                seated=rng.sample(players,6);mode='refresh'
            else:
                exits=[i for i in range(6) if rng.random()<ATTENDANCE['seat_exit_probability']]
                available=[p for p in players if p not in before]
                for i,newcomer in zip(exits,rng.sample(available,len(exits))):seated[i]=newcomer
                if exits:mode='turnover'
        assert len(set(seated))==6
        if mode!='unchanged':seat_events.append({'hand_number':h,'mode':mode,'before':before,'after':seated.copy()})
        button=rng.randrange(6)
        sitting=[seated[(button+1+i)%6] for i in range(6)]"""
assert old in p;p=p.replace(old,new)
p=p.replace("blind,target,spec['behavior'],intensity,active)","blind,target,spec['behavior'],intensity,active,button_seat=button)")
p=p.replace("    (dest/'balance-resets.json').write_text(json.dumps(resets))","    (dest/'balance-resets.json').write_text(json.dumps(resets))\n    (dest/'seat-events.json').write_text(json.dumps(seat_events))")
p=p.replace("'participation':participation,","'attendance_process':ATTENDANCE,")
(D/'run_pilot.py').write_text(p);compile(p,str(D/'run_pilot.py'),'exec');compile(g,str(D/'generator.py'),'exec')
plan=json.loads((S/'pilot-plan.json').read_text());plan['created_at']=datetime.now(timezone.utc).isoformat();plan['parent_plan_sha256']=sha(S/'pilot-plan.json')
for row in plan['pools']:row['seed']+=400000
plan['source_hashes']={n:sha(D/n) for n in [*plan['source_hashes'],'attendance-process.json']}
plan['repair']='Keep engine-settled balances; replace independent per-hand seating with fitted persistent physical seats,independent seat exits and occasional full refresh. Sample physical button independently as observed in original training. Rotate engine inputs and output seat labels consistently.'
plan['design']='32fresh pools,30players per pool,3000hands,4policy families by4mechanism conditions;persistent seats and balances.'
plan['realism_gates']['whole_player_temporal_domain_auc_max']=.8
(D/'pilot-plan.json').write_text(json.dumps(plan,indent=2))
for old,new in [('prepare_balance_features.py','prepare_session_features.py'),('balance_pair_domain_audit.py','session_pair_domain_audit.py'),('audit_balance_pilot.py','audit_session_pilot.py'),('replay_balance_pilot.py','replay_session_pilot.py')]:
    source=(H/old).read_text().replace("D=H/'persistent_balance'","D=H/'persistent_sessions'")
    if old=='replay_balance_pilot.py':
        source=source.replace("stacks=[r['starting_stack'] for r in seats];", "seats.sort(key=lambda r:(r['seat_no']-hand['button_seat']-1)%6)\n        stacks=[r['starting_stack'] for r in seats];")
    (H/new).write_text(source);compile(source,str(H/new),'exec')
auditnames=['prepare_session_features.py','session_pair_domain_audit.py','audit_session_pilot.py','replay_session_pilot.py','temporal_audit.py']
(D/'auditor-freeze.json').write_text(json.dumps({n:sha(H/n) for n in auditnames},indent=2))
print(json.dumps({'plan_sha256':sha(D/'pilot-plan.json'),'path':str(D)}))
