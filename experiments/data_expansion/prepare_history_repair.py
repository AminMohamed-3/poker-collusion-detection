"""One training-only sampler repair; separate from the frozen utility trial."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
import polars as pl
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];S=H/'learned';D=H/'history_repair'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (D/'pilot-plan.json').exists();D.mkdir(exist_ok=True)
pools=json.loads((H/'training-calibration.json').read_text())['training_pools']
h=pl.read_parquet(R/'data/hands.parquet').filter((pl.col('phase')=='development')&pl.col('table_id').is_in(pools)).select('hand_id','big_blind')
s=pl.scan_parquet(R/'data/seats.parquet').join(h.lazy(),on='hand_id',how='inner').select('player_id',(pl.col('starting_stack')/pl.col('big_blind')).alias('stack_bb')).collect()
quantiles=np.quantile(s['stack_bb'].to_numpy(),np.linspace(0,1,4097)).tolist()
sampler={'training_pools':pools,'stack_bb_quantiles':quantiles,'sampling':'Inverse-CDF interpolation using all eligible training-seat stacks; independently draw each fresh stack and round in chips, not big blinds. No reuse of original hands/cards/outcomes.','participation_log_sd':.4,'tilt_active':True,'source_audit_sha256':sha(S/'pair-domain-audit.json')}
(D/'sampler.json').write_text(json.dumps(sampler,indent=2))
for name in ['generator.py','test_generator.py','models.joblib','fit.json','conditional_action_features.py','advanced_features.py','training-calibration.json']:shutil.copy2(S/name,D/name)
p=(S/'learned_policy.py').read_text()
needle='    p[0 if obs.to_call==0 else 1]=0'
assert needle in p
p=p.replace(needle,"    # Public ordinary-player variation, including the runner's episodic tilt.\n    p[0]*=math.exp(1.5*(profile.tightness-.08))\n    p[3]*=math.exp(2*profile.aggression)\n    p=p**(1/max(.8,1+(profile.noise-.12)))\n"+needle)
(D/'learned_policy.py').write_text(p)
p=(S/'run_pilot.py').read_text()
p=p.replace('from learned_policy import sample_style','from learned_policy import sample_style\nimport math')
p=p.replace("FAMILIES=['none','directed_transfer','soft_play','coordinated_isolation']","FAMILIES=['none','directed_transfer','soft_play','coordinated_isolation']\nSAMPLER=json.loads((HERE/'sampler.json').read_text())\ndef stack_draw(rng,blind):\n    q=SAMPLER['stack_bb_quantiles']; x=rng.random()*(len(q)-1); i=int(x); bb=q[i]+(q[min(i+1,len(q)-1)]-q[i])*(x-i)\n    return max(40*blind,min(250*blind,round(bb*blind)))")
p=p.replace("f'SLP{pool_id:04d}","f'SHP{pool_id:04d}")
p=p.replace('    stack_center={p:rng.uniform(70,170) for p in players}',"    stack_center={p:rng.uniform(70,170) for p in players}\n    participation={p:math.exp(rng.gauss(0,SAMPLER['participation_log_sd'])) for p in players}")
p=p.replace('        sitting=rng.sample(players,6)',"        sitting=sorted(players,key=lambda p:-math.log(max(1e-12,rng.random()))/participation[p])[:6]\n        rng.shuffle(sitting)")
p=p.replace('        stacks=[max(40,min(250,round(rng.gauss(stack_center[p],30))))*blind for p in sitting]','        stacks=[stack_draw(rng,blind) for p in sitting]')
p=p.replace('        assert counts[pair_id]>=20','        if counts[pair_id]<20: continue')
p=p.replace("    for name,rows in [('hands',hands)","    eligible={r['pair_id'] for r in labels}\n    evidence=[r for r in evidence if r['pair_id'] in eligible]\n    for name,rows in [('hands',hands)")
p=p.replace("                'tilt_player':tilt_player", "                'participation':participation,'tilt_player':tilt_player")
start=p.index("    repair=json.loads((HERE/'serialization-repair.json').read_text())")
end=p.index('    summaries=[]',start)
p=p[:start]+"    assert all(sha(HERE/p)==h for p,h in plan['source_hashes'].items())\n"+p[end:]
p=p.replace(",'serialization_repair_sha256':sha(HERE/'serialization-repair.json')",'')
(D/'run_pilot.py').write_text(p)
p=(D/'generator.py').read_text().replace("f'SL{pool_id:05d}H{hand_number:05d}'","f'SH{pool_id:05d}H{hand_number:05d}'").replace("f'SLPOOL{pool_id:05d}'","f'SHPOOL{pool_id:05d}'")
(D/'generator.py').write_text(p)
plan=json.loads((S/'pilot-plan.json').read_text());plan['created_at']=datetime.now(timezone.utc).isoformat();plan['parent_plan_sha256']=sha(S/'pilot-plan.json')
plan['source_hashes']={n:sha(D/n) for n in ['generator.py','run_pilot.py','learned_policy.py','test_generator.py','models.joblib','fit.json','conditional_action_features.py','advanced_features.py','sampler.json']}
for pool in plan['pools']:pool['seed']+=200000
plan['repair']='Training-only history sampler repair: empirical stack CDF, chip-unit rounding, independent stack draws, heterogeneous player attendance with fixed log-sd .4, and actual benign tilt shifts in fold/raise propensity. Same32-pool pilot only; no extension to10x. No utility outcomes consulted.'
plan['next_stage']='Audit raw validity, hand and pair source distributions using only original training pools. Do not modify the already frozen learned-v2 utility experiment or fit a new held-out comparison until its complete results are reported and any next experiment is frozen.'
(D/'pilot-plan.json').write_text(json.dumps(plan,indent=2))
files=[p for p in D.iterdir() if p.is_file()]
with tarfile.open(H/'history-repair-package.tar.gz','w:gz') as t:
    for p in files:t.add(p,arcname=p.relative_to(D))
print(json.dumps({'plan_sha256':sha(D/'pilot-plan.json'),'package_sha256':sha(H/'history-repair-package.tar.gz'),'training_seat_rows':s.height,'stack_quantiles':len(quantiles)}))
