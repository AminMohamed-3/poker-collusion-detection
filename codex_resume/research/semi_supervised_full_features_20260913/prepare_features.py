import os
os.environ['POLARS_MAX_THREADS']='2'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,time,argparse,gc
from pathlib import Path
from datetime import datetime,timezone
import polars as pl
import numpy as np
OUT=Path(__file__).resolve().parent;R=OUT.parent.parent;ROOT=R.parent
sys.stdout.reconfigure(encoding='utf-8');sys.path.insert(0,str(ROOT/'Claude Version/src'));sys.path.insert(0,str(ROOT/'src'))
from pu_pairs.build_allpairs_fold import base_features,v2_features
from robust_pair_experiment import aggregate
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):
 assert not p.exists(),str(p);p.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
def initialize():
 inventory=R/'research/unlabeled_development_inventory_20260913/development-pair-inventory.parquet';x=pl.read_parquet(inventory).filter(pl.col('shared_hands')>=38).sort('pair_id');assert x.height==66521 and x.filter(pl.col('label')<0).height==65734
 pools=sorted(x['table_id'].unique());assert len(pools)==169;batchpools=[pools[i:i+2] for i in range(0,len(pools),2)]
 old=R/'research/semi_supervised_features_20260913';pair=R/'experiments/semi_supervised_pair_yield_20260913';evidence=R/'experiments/semi_supervised_evidence_yield_20260913'
 plan={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Full-corpus deterministic feature preparation following low agreement yield in sample64. No new teachers, thresholds or quality evaluation.','source_inventory_sha256':sha(inventory),'selected_pairs':x.height,'known_pairs':787,'unknown_pairs':65734,'per_pool_unknown_cap':None,'sampling':'All eligible unknown pairs in the original169alloweddevelopmentpools; no outcome-based IDs or subset selection','batch_pools':batchpools,'parity_pools':['T0710278A','T00AE1F5C','T08B7DB49'],'previous_sample_cache':str(old),'previous_sample_agreement_result_sha256':sha(evidence/'yield-result.json'),'sources':{str(p):sha(p) for p in [Path(__file__),ROOT/'Claude Version/src/pu_pairs/build_allpairs_fold.py',ROOT/'src/advanced_features.py',ROOT/'src/robust_pair_experiment.py',old/'feature-plan.json',pair/'plan.json',evidence/'plan.json']},'learned_conditional_features_used':False,'feature_count_expected':163,'old_outer_pools_excluded':True,'fitting_boundary':'Retain exact old teacher fits and per-fold train/held pool boundaries; only enlarge unknown train corpus. New student work requires a separate frozen protocol and gate.','required_parity':'Every known raw feature and every prior sampled unknown raw feature must match existing sources exactly.'}
 x.write_parquet(OUT/'selected-pairs.parquet');write(OUT/'feature-plan.json',plan);print(json.dumps({k:v for k,v in plan.items() if k not in ['batch_pools','sources']}),flush=True)
def build(pools,tag):
 plan=json.loads((OUT/'feature-plan.json').read_text());assert all(sha(Path(p))==s for p,s in plan['sources'].items())
 receipt=OUT/f'{tag}.json'
 if receipt.exists():
  old=json.loads(receipt.read_text());assert sha(OUT/old['hand_file'])==old['hand_sha256'];assert sha(OUT/old['pair_file'])==old['pair_sha256'];print(json.dumps({'verified_existing':tag}),flush=True);return
 t=time.perf_counter();selected=pl.read_parquet(OUT/'selected-pairs.parquet').filter(pl.col('table_id').is_in(pools));p=selected.select('pair_id','player_1','player_2','table_id')
 h=pl.scan_parquet(ROOT/'data/hands.parquet').filter((pl.col('phase')=='development')&pl.col('table_id').is_in(pools)).collect()
 raw=v2_features(base_features(h,p),p).sort('pair_id','hand_id');keys=['pair_id','hand_id','table_id'];features=[c for c in raw.columns if c not in keys];assert len(features)==163
 got=raw.group_by('pair_id').len().rename({'len':'actual_hands'}).join(selected.select('pair_id','shared_hands'),on='pair_id',validate='1:1');assert got.height==selected.height and got.filter(pl.col('actual_hands')!=pl.col('shared_hands')).height==0
 old=pl.scan_parquet(str(ROOT/'outputs_v2/hands_development/batch_*.parquet')).filter(pl.col('table_id').is_in(pools)).collect().sort('pair_id','hand_id');old=old.join(selected.filter(pl.col('label')>=0).select('pair_id'),on='pair_id',how='semi').sort('pair_id','hand_id')
 rebuilt=raw.join(old.select('pair_id').unique(),on='pair_id',how='semi').sort('pair_id','hand_id');assert old.select(keys).equals(rebuilt.select(keys))
 X=old.select(features).to_numpy();Z=rebuilt.select(features).to_numpy();exact=np.array_equal(X,Z,equal_nan=True);diff=np.abs(X-Z);maxdiff=float(np.nanmax(diff)) if diff.size else 0
 compare={'known_hand_rows':old.height,'known_pairs':old['pair_id'].n_unique(),'exact_raw_parity':exact,'max_abs_difference':maxdiff,'unequal_columns':[f for k,f in enumerate(features) if not np.array_equal(X[:,k],Z[:,k],equal_nan=True)]}
 if not exact:
  write(OUT/f'{tag}-parity-failure.json',compare);print(json.dumps(compare),flush=True);raise AssertionError('Stop for raw feature parity review')
 oldsample=pl.scan_parquet(str(Path(plan['previous_sample_cache'])/'batch_*-hands.parquet')).filter(pl.col('table_id').is_in(pools)).collect().sort('pair_id','hand_id')
 newsubset=raw.join(oldsample.select('pair_id').unique(),on='pair_id',how='semi').sort('pair_id','hand_id');assert oldsample.select(keys).equals(newsubset.select(keys));assert np.array_equal(oldsample.select(features).to_numpy(),newsubset.select(features).to_numpy(),equal_nan=True),'Old unknown sample raw parity failed'
 compare['prior_sample_raw_parity']=True;compare['prior_sample_hand_rows']=oldsample.height
 pairframe=aggregate(raw,features); assert pairframe.height==selected.height
 hand_file=f'{tag}-hands.parquet';pair_file=f'{tag}-pairs.parquet';raw.write_parquet(OUT/hand_file);pairframe.write_parquet(OUT/pair_file)
 report={'pools':pools,'pairs':selected.height,'unknown_pairs':selected.filter(pl.col('label')<0).height,'pair_hand_rows':raw.height,'raw_features':len(features),'pair_features':pairframe.width-2,'seconds':time.perf_counter()-t,'hand_file':hand_file,'pair_file':pair_file,'hand_sha256':sha(OUT/hand_file),'pair_sha256':sha(OUT/pair_file),**compare}
 write(receipt,report);print(json.dumps(report),flush=True);gc.collect()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['init','parity','build']);ap.add_argument('--batch',type=int,nargs='*');args=ap.parse_args()
 if args.mode=='init':initialize()
 elif args.mode=='parity':build(json.loads((OUT/'feature-plan.json').read_text())['parity_pools'],'parity')
 else:
  batches=json.loads((OUT/'feature-plan.json').read_text())['batch_pools']
  for k in (args.batch if args.batch is not None else range(len(batches))):build(batches[k],f'batch_{k:03}')
