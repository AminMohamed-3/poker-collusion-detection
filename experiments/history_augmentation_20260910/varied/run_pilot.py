"""Generate a fixed policy-by-behavior pilot with pool-level provenance."""
import os
os.environ['POLARS_MAX_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
import polars as pl
from generator import Profile,simulate_hand
from learned_policy import sample_style
import math

HERE=Path(__file__).resolve().parent
FAMILIES=['none','directed_transfer','soft_play','coordinated_isolation']
SAMPLER=json.loads((HERE/'sampler.json').read_text())
PROFILES=json.loads((HERE/'profiles.json').read_text())['profiles']
BALANCE=json.loads((HERE/'balance-process.json').read_text())
ATTENDANCE=json.loads((HERE/'attendance-process.json').read_text())
def stack_draw(rng,blind,q):
    x=rng.random()*(len(q)-1); i=int(x); bb=q[i]+(q[min(i+1,len(q)-1)]-q[i])*(x-i)
    return max(40*blind,min(250*blind,round(bb*blind)))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run_pool(spec):
    pool_id=spec['pool_id']; dest=HERE/'pilot'/f'pool-{pool_id:04d}'
    if (dest/'receipt.json').exists():
        receipt=json.loads((dest/'receipt.json').read_text())
        assert all(sha(dest/p)==h for p,h in receipt['files'].items())
        return receipt
    dest.mkdir(parents=True,exist_ok=False)
    BALANCE={**globals()['BALANCE'],**spec['balance']}
    ATTENDANCE={**globals()['ATTENDANCE'],**spec['attendance']}
    rng=random.Random(spec['seed']); players=[f'SSP{pool_id:04d}U{i:02d}' for i in range(30)]
    prototypes=dict(zip(players,rng.sample(PROFILES,30)))
    profiles={p:Profile(spec['policy_family'],rng.uniform(-.035,.2),rng.uniform(-.12,.14),rng.uniform(.04,.22),rng.uniform(.015,.095),rng.uniform(.5,1.3),tuple(prototypes[p]['style'])) for p in players}
    target=tuple(rng.sample(players,2)) if spec['behavior']!='none' else None
    candidates=[pair for pair in itertools.combinations(players,2) if set(pair)!=set(target or [])]
    selected=([target] if target else [])+rng.sample(candidates,4 if target else 5)
    pair_ids={tuple(sorted(pair)):f'SSP{pool_id:04d}PAIR{i}' for i,pair in enumerate(selected)}
    blind=rng.choices([2,4,10],[.542,.357,.101])[0]
    intensity=rng.uniform(.4,.9); onset=rng.uniform(.05,.35); duration=rng.uniform(.35,.6)
    balances={p:round(BALANCE['first_stack_by_player'][prototypes[p]['source_player_id']]*blind) for p in players}
    previously_seen=set(); resets=[]
    seated=rng.sample(players,6); seat_events=[]
    tilt_player=rng.choice(players); tilt_onset=rng.uniform(.1,.7)
    hands=[];seats=[];actions=[];traces=[];counts={p:0 for p in pair_ids.values()};evidence=[]
    began=time.perf_counter()
    for h in range(spec['hands']):
        before=seated.copy()
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
        sitting=[seated[(button+1+i)%6] for i in range(6)]
        active=onset<=h/spec['hands']<onset+duration
        for player in sitting:
            if player in previously_seen:
                boundary=not BALANCE['min_balance_bb']*blind<=balances[player]<=BALANCE['max_balance_bb']*blind
                spontaneous=(not boundary) and rng.random()<BALANCE['spontaneous_reset_probability']
                if boundary or spontaneous:
                    before=balances[player]
                    balances[player]=round(blind*rng.choices(BALANCE['reset_stack_bb'],BALANCE['reset_weights'])[0])
                    resets.append({'hand_number':h,'player_id':player,'before':before,'after':balances[player],'reason':'boundary' if boundary else 'spontaneous'})
        stacks=[balances[p] for p in sitting]
        pf=[profiles[p] for p in sitting]
        if tilt_onset<=h/spec['hands']<tilt_onset+.12 and tilt_player in sitting:
            i=sitting.index(tilt_player); pf[i]=replace(pf[i],aggression=pf[i].aggression+.18,tightness=pf[i].tightness-.1,noise=.25)
        hand,ss,aa,tt=simulate_hand(spec['seed']*10000+h,pool_id,h,sitting,pf,stacks,blind,target,spec['behavior'],intensity,active,button_seat=button)
        hands.append(hand);seats.extend(ss);actions.extend(aa)
        for seat in ss:
            player=seat['player_id'];balances[player]=seat['starting_stack']+seat['net_chips'];previously_seen.add(player)
            assert balances[player]>=0
        for pair,pair_id in pair_ids.items():
            if set(pair)<=set(sitting): counts[pair_id]+=1
        if tt:
            assert target and set(target)<=set(sitting)
            pair_id=pair_ids[tuple(sorted(target))]
            traces.append({'hand_id':hand['hand_id'],'pair_id':pair_id,'interventions':tt})
            essential={'directed_transfer':{'donor_calls_partner_while_behind','donor_folds_stronger_hand'},'soft_play':{'partner_check','partner_raise_suppressed'},'coordinated_isolation':{'pressure_outsiders'}}[spec['behavior']]
            if any(t['reason'] in essential for t in tt): evidence.append({'pair_id':pair_id,'hand_id':hand['hand_id']})
    labels=[]
    for pair,pair_id in pair_ids.items():
        positive=target is not None and set(pair)==set(target)
        # Never manufacture a positive hand when the intervention did not act.
        if positive and not evidence: continue
        if counts[pair_id]<50: continue
        labels.append({'pair_id':pair_id,'player_1':pair[0],'player_2':pair[1],
                       'label':int(positive),'label_status':'synthetic_intervention' if positive else 'synthetic_non_target',
                       'behavior_family':spec['behavior'] if positive else 'none'})
    eligible={r['pair_id'] for r in labels}
    evidence=[r for r in evidence if r['pair_id'] in eligible]
    for name,rows in [('hands',hands),('seats',seats),('actions',actions)]:
        frame=pl.DataFrame(rows)
        if name=='hands': frame=frame.with_columns((pl.col('started_at')*60_000_000).cast(pl.Datetime('us','UTC')))
        frame.write_parquet(dest/(name+'.parquet'))
    pl.DataFrame(labels).write_csv(dest/'development_labels.csv')
    pl.DataFrame(evidence,schema={'pair_id':pl.String,'hand_id':pl.String}).write_csv(dest/'development_evidence.csv')
    (dest/'interventions.json').write_text(json.dumps(traces))
    (dest/'balance-resets.json').write_text(json.dumps(resets))
    (dest/'seat-events.json').write_text(json.dumps(seat_events))
    provenance={'spec':spec,'profiles':{p:asdict(v) for p,v in profiles.items()},'target':target,
                'intensity':intensity,'onset':onset,'duration':duration,'blind':blind,
                'prototype_sources':{p:prototypes[p]['source_player_id'] for p in players},'balance_process':{k:v for k,v in BALANCE.items() if k!='first_stack_by_player'},'attendance_process':ATTENDANCE,'tilt_player':tilt_player,'tilt_onset':tilt_onset,'shared_hands':counts,
                'known_positive_omitted_for_no_intervention':bool(target and not evidence)}
    (dest/'provenance.json').write_text(json.dumps(provenance,indent=2,default=float))
    receipt={'pool_id':pool_id,'policy_family':spec['policy_family'],'behavior':spec['behavior'],
             'hands':len(hands),'actions':len(actions),'pairs':len(labels),'positive_pairs':sum(r['label'] for r in labels),
             'intervention_hands':len(evidence),'seconds':time.perf_counter()-began,
             'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}}
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2))
    return receipt

def main():
    plan_path=HERE/'pilot-plan.json'
    assert plan_path.exists()
    plan=json.loads(plan_path.read_text())
    assert all(sha(HERE/p)==h for p,h in plan['source_hashes'].items())
    summaries=[]
    with ProcessPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(run_pool,s) for s in plan['pools']]):
            r=future.result();summaries.append(r)
            print(json.dumps({k:v for k,v in r.items() if k!='files'}),flush=True)
    receipt={'plan_sha256':sha(plan_path),'completed_at':datetime.now(timezone.utc).isoformat(),
             'pools':sorted(summaries,key=lambda r:r['pool_id'])}
    (HERE/'pilot-generation.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
