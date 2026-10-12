"""Final bounded training-only pilot: preserve player-conditioned stack/style variation."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent;R=H.parents[1];S=H/'history_repair';D=H/'profile_repair'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not D.exists();D.mkdir()
pools=json.loads((H/'training-calibration.json').read_text())['training_pools']
h=pl.read_parquet(R/'data/hands.parquet').filter((pl.col('phase')=='development')&pl.col('table_id').is_in(pools)).select('hand_id','big_blind')
s=pl.scan_parquet(R/'data/seats.parquet').join(h.lazy(),on='hand_id',how='inner').select('player_id',(pl.col('starting_stack')/pl.col('big_blind')).alias('stack_bb')).collect().group_by('player_id').agg(pl.col('stack_bb')).sort('player_id')
a=pl.scan_parquet(R/'data/actions.parquet').join(h.lazy().select('hand_id'),on='hand_id',how='inner').with_columns(pl.when(pl.col('action')=='fold').then(0).when(pl.col('action')=='check').then(1).when((pl.col('action')=='call')|((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call')))).then(2).otherwise(3).alias('kind')).group_by('player_id').agg(*[(pl.col('kind')==k).sum().alias('c'+str(k)) for k in range(4)]).collect()
s=s.join(a,on='player_id',validate='1:1');profiles=[]
for row in s.iter_rows(named=True):
    counts=np.array([row['c'+str(k)] for k in range(4)]);profiles.append({'source_player_id':row['player_id'],'style':((counts+1)/(counts.sum()+4)).tolist(),'stack_bb_quantiles':np.quantile(row['stack_bb'],np.linspace(0,1,65)).tolist()})
(D/'profiles.json').write_text(json.dumps({'training_pools':pools,'profiles':profiles,'source_audit_sha256':sha(S/'pair-domain-audit.json'),'no_original_hands_cards_or_outcomes_copied':True}))
for p in S.iterdir():
    if p.name in ['generator.py','test_generator.py','models.joblib','fit.json','conditional_action_features.py','advanced_features.py','training-calibration.json','learned_policy.py','sampler.json']:shutil.copy2(p,D/p.name)
p=(S/'run_pilot.py').read_text()
p=p.replace("SAMPLER=json.loads((HERE/'sampler.json').read_text())","SAMPLER=json.loads((HERE/'sampler.json').read_text())\nPROFILES=json.loads((HERE/'profiles.json').read_text())['profiles']")
p=p.replace('def stack_draw(rng,blind):\n    q=SAMPLER[\'stack_bb_quantiles\']; ','def stack_draw(rng,blind,q):\n    ')
p=p.replace("f'SHP{pool_id:04d}","f'SPP{pool_id:04d}")
needle="    profiles={p:Profile(spec['policy_family'],rng.uniform(-.035,.2),rng.uniform(-.12,.14),rng.uniform(.04,.22),rng.uniform(.015,.095),rng.uniform(.5,1.3),sample_style(rng)) for p in players}"
assert needle in p
p=p.replace(needle,"    prototypes=dict(zip(players,rng.sample(PROFILES,30)))\n    profiles={p:Profile(spec['policy_family'],rng.uniform(-.035,.2),rng.uniform(-.12,.14),rng.uniform(.04,.22),rng.uniform(.015,.095),rng.uniform(.5,1.3),tuple(prototypes[p]['style'])) for p in players}")
p=p.replace('stacks=[stack_draw(rng,blind) for p in sitting]',"stacks=[stack_draw(rng,blind,prototypes[p]['stack_bb_quantiles']) for p in sitting]")
p=p.replace('if counts[pair_id]<20: continue','if counts[pair_id]<50: continue')
p=p.replace("                'participation':participation,", "                'prototype_sources':{p:prototypes[p]['source_player_id'] for p in players},'participation':participation,")
(D/'run_pilot.py').write_text(p)
p=(D/'generator.py').read_text().replace("f'SH{pool_id:05d}H{hand_number:05d}'","f'SP{pool_id:05d}H{hand_number:05d}'").replace("f'SHPOOL{pool_id:05d}'","f'SPPOOL{pool_id:05d}'");(D/'generator.py').write_text(p)
plan=json.loads((S/'pilot-plan.json').read_text());plan['created_at']=datetime.now(timezone.utc).isoformat();plan['parent_plan_sha256']=sha(S/'pilot-plan.json')
plan['source_hashes']={n:sha(D/n) for n in ['generator.py','run_pilot.py','learned_policy.py','test_generator.py','models.joblib','fit.json','conditional_action_features.py','advanced_features.py','sampler.json','profiles.json']}
for pool in plan['pools']:pool['seed']+=200000
plan['validity_gate']['minimum_shared_hands_per_labelled_pair']=50
plan['repair']='Third and final bounded ordinary-history repair in this session: sample player-conditioned stack CDF and public action-style jointly from training-only prototypes, then fresh cards/actions/attendance. Require50 shared hands for a supervised pair, guided by training labels having minimum57. Preserve excluded pairs as unlabelled; drop their evidence. No utility results used; no claim of new independent source policies.'
plan['next_stage']='Run legal/hand/pair source audits only. Preserve and report failure too. Do not automatically launch further generator versions or a new held-data utility trial in this session.'
(D/'pilot-plan.json').write_text(json.dumps(plan,indent=2))
with tarfile.open(H/'profile-repair-package.tar.gz','w:gz') as t:
    for p in D.iterdir():
        if p.is_file():t.add(p,arcname=p.name)
print(json.dumps({'plan_sha256':sha(D/'pilot-plan.json'),'package_sha256':sha(H/'profile-repair-package.tar.gz'),'prototypes':len(profiles)}))
