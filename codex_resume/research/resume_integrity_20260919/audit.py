import os
os.environ['POLARS_MAX_THREADS']='2'
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import polars as pl
OUT=Path(__file__).resolve().parent;CR=OUT.parent.parent;R=CR.parent;C=R/'Claude Version'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((OUT/'plan.json').read_text())
for p,h in plan['sources'].items():assert sha(Path(p))==h,p
labels=pl.read_csv(R/'data/development_labels.csv').select('pair_id','player_1','player_2','label','behavior_family')
audit=pl.read_parquet(C/'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet',columns=['pair_id','hand_id'])
keys=pl.read_parquet(C/'experiments/step254_mechanical_tensor_20260917/step254_features.parquet',columns=['pair_id','hand_id'])
pools=pl.read_parquet(R/'outputs_v2/pairs_development.parquet',columns=['pair_id','table_id'])
meta=labels.filter(pl.col('label')==1).join(pools,on='pair_id',validate='1:1')
ap=set(audit['pair_id']);tp=set(meta['pair_id'])-ap
assert len(ap)==162 and len(tp)==210 and meta.height==372
pre=json.loads((CR/'experiments/tabpfn3_relative_20260912/preflight.json').read_text())
reports={}
for family in ['all','soft_play','directed_transfer','coordinated_isolation']:
 m=meta if family=='all' else meta.filter(pl.col('behavior_family')==family)
 a=m.filter(pl.col('pair_id').is_in(sorted(ap)));t=m.filter(pl.col('pair_id').is_in(sorted(tp)))
 ats=set(a['table_id']);tts=set(t['table_id']);aps=set(a['player_1'])|set(a['player_2']);tps=set(t['player_1'])|set(t['player_2'])
 ak=keys.filter(pl.col('pair_id').is_in(a['pair_id'].to_list()));tk=keys.filter(pl.col('pair_id').is_in(t['pair_id'].to_list()))
 commonhands=set(ak['hand_id'])&set(tk['hand_id'])
 reports[family]={'train_pairs':t.height,'audit_pairs':a.height,'train_pools':len(tts),'audit_pools':len(ats),'pair_overlap':len(set(t['pair_id'])&set(a['pair_id'])),'pool_overlap':len(tts&ats),'pair_player_overlap':len(aps&tps),'physical_hand_overlap_all_candidates':len(commonhands),'train_pairs_in_historical_outer_pools':t.filter(pl.col('table_id').is_in(pre['excluded_old_outer_pools'])).height,'train_pools_in_historical_outer_pools':len(tts&set(pre['excluded_old_outer_pools'])),'train_pair_hand_rows':tk.height,'audit_pair_hand_rows':ak.height}
cache=CR/'research/semi_supervised_full_features_20260913';recs=[json.loads((cache/f'batch_{i:03}.json').read_text()) for i in range(85)]
for j in recs:
 assert j['exact_raw_parity'] and j['prior_sample_raw_parity']
 assert sha(cache/j['hand_file'])==j['hand_sha256'] and sha(cache/j['pair_file'])==j['pair_sha256']
completion={'batches':len(recs),'pools':len(set(p for x in recs for p in x['pools'])),'pairs':sum(x['pairs'] for x in recs),'pair_hand_rows':sum(x['pair_hand_rows'] for x in recs),'known_hand_rows':sum(x['known_hand_rows'] for x in recs),'prior_sample_hand_rows':sum(x['prior_sample_hand_rows'] for x in recs),'all_hashes_verified':True,'all_raw_parities_exact':True}
protected={}
for p,h in plan['protected'].items():
 actual=sha(Path(p));assert actual==h,p;protected[p]=actual
result={'utc':datetime.now(timezone.utc).isoformat(),'step277_split':reports,'full_feature_completion':completion,'protected':protected,'plan_sha256':sha(OUT/'plan.json'),'new_fits':0,'new_quality_evaluations':0,'evidence_truth_values_read':False,'claim_limit':'Disjointness check only; does not establish predictive improvement or untuned confirmation. Historical outer pools used as training cannot serve as protected confirmation for this model.'}
assert not (OUT/'result.json').exists()
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['protected']},indent=2))
