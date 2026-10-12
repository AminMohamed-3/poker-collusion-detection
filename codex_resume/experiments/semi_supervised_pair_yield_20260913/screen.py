import os
os.environ['POLARS_MAX_THREADS']='2'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,time,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,polars as pl,lightgbm as lgb
OUT=Path(__file__).resolve().parent;R=OUT.parent.parent;ROOT=R.parent;CACHE=R/'research/semi_supervised_features_20260913'
sys.stdout.reconfigure(encoding='utf-8');sys.path.insert(0,str(ROOT/'src'))
from robust_pair_experiment import aggregate,thin,MODEL_PARAMS,LABELS,matrix
from tail_pair_experiment import tails
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):
 assert not p.exists(),str(p);p.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
def initialize():
 pre=R/'experiments/tabpfn3_relative_20260912/preflight.json';j=json.loads(pre.read_text());selected=CACHE/'selected-pairs.parquet'
 plan={'created_utc':datetime.now(timezone.utc).isoformat(),'question':'Does the frozen small development-only corpus supply enough teacher-selected whole positive pairs for a controlled evidence pilot?','scope':'First-salt yield screen only, no held outcome metrics or evidence labels; insufficient yield is not evidence against full-scale semi-supervision.','sampling':'Use already frozen codex-ssl64-v1:pair_id SHA256 sample64 perallowedpool without reselection','pair_teacher':'Four-class LightGBM, full/70%/40% known training views, eachweight1/3; raw+v2 aggregate and tails, no conditional learned features','pair_parameters':MODEL_PARAMS,'view_seeds':{'70':7307,'40':7304},'filter':{'full_positive_score_at_least':.95,'view70_positive_score_at_least':.95,'same_winning_positive_family':True,'max_selected_per_pool':2,'sort':'minimum(full_positive,view70_positive) descending then pair_id ascending'},'gate':{'minimum_distinct_selected_pairs_across_five_cells':15,'minimum_original_pools':10,'if_fail':'Stop before any evidence teacher/student fit or quality evaluation; preserve gate, consider separate larger corpus protocol.','if_pass':'Prepare a separately frozen evidence-teacher agreement and matched student pilot; do not treat selected pairs as verified positives.'},'salt':j['partitions'][0],'reference_pools':j['reference_pools'],'excluded_old_outer_pools':j['excluded_old_outer_pools'],'sources':{str(p):sha(p) for p in [Path(__file__),pre,selected,CACHE/'feature-plan.json',ROOT/'src/robust_pair_experiment.py',ROOT/'src/tail_pair_experiment.py']},'probabilities_not_calibrated_truth':True,'no_new_evidence_outcome_reads':True,'no_test_data':True}
 write(OUT/'plan.json',plan);print(json.dumps({'plan_frozen':True,'teacher_fits':5,'threshold':.95,'minimum_pairs':15,'minimum_pools':10}),flush=True)
def prepare():
 plan=json.loads((OUT/'plan.json').read_text());assert all(sha(Path(p))==s for p,s in plan['sources'].items());cp=json.loads((CACHE/'feature-plan.json').read_text());meta=pl.read_parquet(CACHE/'selected-pairs.parquet')
 views={'full':[],'v70':[],'known_v40':[]};receipt=[]
 for i,pools in enumerate(cp['batch_pools']):
  rp=CACHE/f'batch_{i:03}.json';j=json.loads(rp.read_text());assert j['exact_raw_parity'];hp=CACHE/j['hand_file'];pp=CACHE/j['pair_file'];assert sha(hp)==j['hand_sha256'] and sha(pp)==j['pair_sha256']
  raw=pl.read_parquet(hp);names=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']];events=[c for c in names if c.startswith('v2_')]
  for name,fraction,seed in [('full',1,None),('v70',.7,7307),('known_v40',.4,7304)]:
   z=raw
   if name=='known_v40':z=z.join(meta.filter(pl.col('label')>=0).select('pair_id'),on='pair_id',how='semi')
   if fraction<1:z=thin(z,fraction,seed)
   a=pl.read_parquet(pp) if name=='full' else aggregate(z,names)
   a=a.join(tails(z,events),on=['pair_id','table_id'],validate='1:1').join(meta.select('pair_id','player_1','player_2','label','behavior_family'),on='pair_id',validate='1:1').sort('pair_id');views[name].append(a)
  receipt.append({'batch':i,'receipt_sha256':sha(rp),'hand_sha256':j['hand_sha256'],'pair_sha256':j['pair_sha256']})
  if i%15==0:print(json.dumps({'pair_views_prepared_batches':i+1,'total_batches':len(cp['batch_pools'])}),flush=True)
 outputs={}
 for name,v in views.items():
  x=pl.concat(v).sort('pair_id');assert x['pair_id'].n_unique()==x.height;p=OUT/f'{name}.parquet';assert not p.exists();x.write_parquet(p);outputs[name]={'rows':x.height,'sha256':sha(p)}
 x=pl.read_parquet(OUT/'full.parquet');features=[c for c in x.columns if c not in ['pair_id','table_id','player_1','player_2','label','behavior_family']]
 write(OUT/'frames-frozen.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'features':features,'feature_count':len(features),'outputs':outputs,'source_batches':receipt,'plan_sha256':sha(OUT/'plan.json')});print(json.dumps({'frames_frozen':True,'features':len(features),'outputs':outputs}),flush=True)
def fit():
 plan=json.loads((OUT/'plan.json').read_text());assert all(sha(Path(p))==s for p,s in plan['sources'].items());fr=json.loads((OUT/'frames-frozen.json').read_text());assert sha(OUT/'plan.json')==fr['plan_sha256'];assert all(sha(OUT/f'{n}.parquet')==j['sha256'] for n,j in fr['outputs'].items());features=fr['features']
 fs={n:pl.read_parquet(OUT/f'{n}.parquet') for n in fr['outputs']};full=fs['full'];mapping=plan['salt']['mapping'];allp=set(full['table_id']);assert not allp&set(plan['excluded_old_outer_pools']);reports=[];selected_all=[]
 for fold in range(5):
  dest=OUT/f'fold{fold}';assert not dest.exists();dest.mkdir();held={p for p,k in mapping.items() if k==fold};trainp=allp-held
  train=pl.concat([x.filter(pl.col('table_id').is_in(sorted(trainp))&(pl.col('label')>=0)) for x in fs.values()]).sort('pair_id')
  query=full.filter(pl.col('table_id').is_in(sorted(trainp))&(pl.col('label')<0)).sort('pair_id');q70=fs['v70'].join(query.select('pair_id'),on='pair_id',how='semi').sort('pair_id');assert query['pair_id'].equals(q70['pair_id'])
  heldrows=full.filter(pl.col('table_id').is_in(sorted(held)));hp=set(heldrows['player_1'])|set(heldrows['player_2']);tp=set(train['player_1'])|set(train['player_2'])|set(query['player_1'])|set(query['player_2']);assert not hp&tp and not held&set(train['table_id']) and not held&set(query['table_id'])
  y=np.array([LABELS[f] for f in train['behavior_family']],dtype=np.int32);assert set(y)=={0,1,2,3};X=matrix(train,features);Q=matrix(query,features);Q70=matrix(q70,features)
  t=time.perf_counter();model=lgb.LGBMClassifier(**plan['pair_parameters']);model.fit(X,y,sample_weight=np.full(train.height,1/3));pr=model.predict_proba(Q);ps=model.predict_proba(Q70);mf=dest/'pair-teacher.txt';model.booster_.save_model(str(mf));reloaded=lgb.Booster(model_file=str(mf));assert np.array_equal(pr,reloaded.predict(Q)) and np.array_equal(ps,reloaded.predict(Q70))
  risk=1-pr[:,0];risk70=1-ps[:,0];family=np.argmax(pr[:,1:],axis=1)+1;family70=np.argmax(ps[:,1:],axis=1)+1
  outputs=query.select('pair_id','table_id').with_columns(pl.Series('risk_full',risk),pl.Series('risk_v70',risk70),pl.Series('family_full',family),pl.Series('family_v70',family70),pl.Series('selection_score',np.minimum(risk,risk70)))
  candidates=outputs.filter((pl.col('risk_full')>=.95)&(pl.col('risk_v70')>=.95)&(pl.col('family_full')==pl.col('family_v70'))).sort(['table_id','selection_score','pair_id'],descending=[False,True,False]).group_by('table_id',maintain_order=True).head(2).sort('pair_id')
  outputs.write_parquet(dest/'unknown-scores.parquet');candidates.write_parquet(dest/'selected.parquet');selected_all.append(candidates.with_columns(pl.lit(fold).alias('fold')))
  rep={'fold':fold,'seconds':time.perf_counter()-t,'train_views_rows':train.height,'train_known_pairs':train['pair_id'].n_unique(),'train_pools':len(trainp),'held_pools':len(held),'query_unknown_pairs':query.height,'selected_pairs':candidates.height,'selected_pools':candidates['table_id'].n_unique(),'selected_family_counts':candidates.group_by('family_full').len().to_dicts(),'selected_unknown_ground_truth':False,'exact_saved_model_replay':True,'held_pool_player_overlap':0,'physical_hand_exclusion_by_original_pool':True,'model_sha256':sha(mf),'scores_sha256':sha(dest/'unknown-scores.parquet'),'selected_sha256':sha(dest/'selected.parquet')};write(dest/'fit.json',rep);reports.append(rep);print(json.dumps(rep),flush=True)
 combined=pl.concat(selected_all);distinct=combined.unique('pair_id');passed=distinct.height>=15 and distinct['table_id'].n_unique()>=10;combined.write_parquet(OUT/'selected-all-folds.parquet')
 result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'YIELD_PASS' if passed else 'INSUFFICIENT_YIELD_STOP','teacher_fits':5,'distinct_selected_pairs':distinct.height,'distinct_selected_pools':distinct['table_id'].n_unique(),'selected_appearances':combined.height,'passes_yield_gate':passed,'folds':reports,'no_evidence_labels_read':True,'no_held_quality_metrics':True,'no_verified_pseudo_truth':True,'next_action':plan['gate']['if_pass' if passed else 'if_fail']};write(OUT/'yield-result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['init','prepare','fit']);a=p.parse_args();{'init':initialize,'prepare':prepare,'fit':fit}[a.mode]()
