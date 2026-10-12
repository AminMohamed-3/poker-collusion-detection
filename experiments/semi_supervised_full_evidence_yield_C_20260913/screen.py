import os
# Claude copy (Amin request, 2026-09-13); inputs read from the Codex tree read-only; outputs stay in OUT. Source sha256 f60f4b7df4e4e6c18c27dacecd6da4832053aea3148f1d60c24bb30c0521ca3d.
os.environ['POLARS_MAX_THREADS']='2'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,time,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,polars as pl,lightgbm as lgb
OUT=Path(__file__).resolve().parent;R=Path('C:/Data/Projects/Money Making/work/poker/Codex Resume');ROOT=R.parent;CACHE=R/'research/semi_supervised_full_features_20260913';OLD=R/'experiments/semi_supervised_evidence_yield_20260913';OLDPAIR=R/'experiments/semi_supervised_pair_yield_20260913';FULLPAIR=OUT.parent/'semi_supervised_full_pair_yield_C_20260913'
sys.stdout.reconfigure(encoding='utf-8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):
 assert not p.exists(),str(p);p.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
def matrix(x,cols):return np.nan_to_num(x.select(cols).to_numpy().astype(np.float32),nan=0,posinf=0,neginf=0)
def initialize():
 old=json.loads((OLD/'plan.json').read_text());source=[Path(__file__),OLD/'plan.json',OLD/'yield-result.json',OLDPAIR/'plan.json',CACHE/'feature-plan.json']
 for k in range(5):
  source += [OLD/f'fold{k}/base107.txt',OLD/f'fold{k}/rawv2_163.txt',OLD/f'fold{k}/teacher-probabilities.parquet',OLD/f'fold{k}/agreement.parquet',OLDPAIR/f'fold{k}/selected.parquet']
 plan={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Full-corpus evidence agreement yield with exact original sample teachers, no refits, no evidence truth or held metrics. Expansion after sampleyield5pairs is disclosed adaptation of corpus size.','sources':{str(p):sha(p) for p in source},'feature_views':old['feature_views'],'agreement':old['agreement'],'pseudo_weight_caps':old['pseudo_weight_caps'],'yield_gate':old['yield_gate'],'new_fits':0,'old_sample_control':'Score union of full selected pairs and prior sample candidates. All prior probabilities and agreement values must reproduce exactly before accepting the enlarged corpus.','upstream_requirements':'All85fullfeaturebatches exactknown+priorsample parity/hashverified; fullpairadapter frozen and terminal with oldsamplepredictionparity and yield pass.','no_threshold_or_teacher_change':True}
 write(OUT/'plan.json',plan);print(json.dumps({'plan_frozen':True,'new_fits':0,'reuse_evidence_teachers':10}),flush=True)
def score():
 plan=json.loads((OUT/'plan.json').read_text());assert all(sha(Path(p))==v for p,v in plan['sources'].items());up=json.loads((FULLPAIR/'yield-result.json').read_text());assert up['passes_yield_gate'];oldplan=json.loads((OLD/'plan.json').read_text());parent=json.loads((OLDPAIR/'plan.json').read_text());mapping=parent['salt']['mapping'];meta=pl.read_parquet(CACHE/'selected-pairs.parquet')
 upstream={str(FULLPAIR/'yield-result.json'):sha(FULLPAIR/'yield-result.json')};source_batches=[]
 cp=json.loads((CACHE/'feature-plan.json').read_text())
 for k in range(len(cp['batch_pools'])):
  rp=CACHE/f'batch_{k:03}.json';j=json.loads(rp.read_text());assert j['exact_raw_parity'] and j['prior_sample_raw_parity'];assert sha(CACHE/j['hand_file'])==j['hand_sha256'];source_batches.append({'batch':k,'receipt_sha256':sha(rp),'hand_sha256':j['hand_sha256']})
 for k in range(5):
  p=FULLPAIR/f'fold{k}/selected.parquet';upstream[str(p)]=sha(p)
 write(OUT/'score-inputs-frozen.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'upstream':upstream,'full_feature_batches':source_batches,'plan_sha256':sha(OUT/'plan.json')})
 ids=pl.concat([pl.read_parquet(p).select('pair_id') for p in [FULLPAIR/f'fold{k}/selected.parquet' for k in range(5)]+[OLDPAIR/f'fold{k}/selected.parquet' for k in range(5)]]).unique()
 raw=pl.concat([pl.read_parquet(b).join(ids,on='pair_id',how='semi') for b in sorted(CACHE.glob('batch_*-hands.parquet'))]).sort('pair_id','hand_id');assert not set(raw['table_id'])&set(parent['excluded_old_outer_pools']);reports=[];accepted_all=[]
 for fold in range(5):
  dest=OUT/f'fold{fold}';assert not dest.exists();dest.mkdir();t=time.perf_counter();held={p for p,k in mapping.items() if k==fold};chosen=pl.read_parquet(FULLPAIR/f'fold{fold}/selected.parquet');oldchosen=pl.read_parquet(OLDPAIR/f'fold{fold}/selected.parquet');queryids=pl.concat([chosen.select('pair_id'),oldchosen.select('pair_id')]).unique();query=raw.join(queryids,on='pair_id',how='semi').sort('pair_id','hand_id');assert not held&set(query['table_id'])
  probs=[]
  for name,features in plan['feature_views'].items():probs.append(lgb.Booster(model_file=str(OLD/f'fold{fold}/{name}.txt')).predict(matrix(query,features)))
  z=query.select('pair_id','hand_id','table_id').with_columns(pl.Series('p_base',probs[0]),pl.Series('p_v2',probs[1]),pl.Series('p_mean',(probs[0]+probs[1])/2));prior=pl.read_parquet(OLD/f'fold{fold}/teacher-probabilities.parquet').sort('pair_id','hand_id');check=z.join(prior.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi').sort('pair_id','hand_id');assert prior.equals(check),'Prior evidence teacher probabilities changed'
  allrows=[];targets=[];newids=set(chosen['pair_id'])
  for group in z.partition_by('pair_id',maintain_order=True):
   a=group.sort(['p_base','hand_id'],descending=[True,False]).head(5);b=group.sort(['p_v2','hand_id'],descending=[True,False]).head(5);overlap=len(set(a['hand_id'])&set(b['hand_id']));mx=float(group['p_mean'].max());accept=overlap>=4 and mx>=.5;pid=group['pair_id'][0];pool=group['table_id'][0];allrows.append({'pair_id':pid,'table_id':pool,'top5_overlap':overlap,'top_mean_probability':mx,'accepted':accept})
   if accept and pid in newids:
    q=np.clip(group['p_mean'].to_numpy(),.05,.95);q*=min(1.,5/q.sum());assert q.sum()<=5+1e-10;targets.append(group.with_columns(pl.Series('soft_listed_target',q),pl.lit(.15/group.height).alias('pseudo_row_weight')))
  all_diag=pl.DataFrame(allrows);prior_diag=pl.read_parquet(OLD/f'fold{fold}/agreement.parquet').sort('pair_id');check_diag=all_diag.join(prior_diag.select('pair_id'),on='pair_id',how='semi').sort('pair_id');assert prior_diag.equals(check_diag),'Prior evidence agreement changed'
  diag=all_diag.join(chosen.select('pair_id'),on='pair_id',how='semi').sort('pair_id');accepted=diag.filter(pl.col('accepted'));accepted_all.append(accepted.with_columns(pl.lit(fold).alias('fold')));diag.write_parquet(dest/'agreement.parquet');z.write_parquet(dest/'teacher-probabilities-with-parity.parquet');train_known_queries=meta.filter((pl.col('label')==1)&~pl.col('table_id').is_in(sorted(held))).height
  if targets:
   target=pl.concat(targets);n=target['pair_id'].n_unique();factor=min(1.,.15*train_known_queries/(.15*n));target=target.with_columns(pl.col('pseudo_row_weight')*factor);assert target.group_by('table_id').agg(pl.col('pseudo_row_weight').sum())['pseudo_row_weight'].max()<=.3+1e-9;assert target['pseudo_row_weight'].sum()<=.15*train_known_queries+1e-9;target.write_parquet(dest/'soft-targets.parquet')
  rep={'fold':fold,'candidate_pairs':chosen.height,'agreement_ge4_pairs':diag.filter(pl.col('top5_overlap')>=4).height,'top_mean_probability_ge05_pairs':diag.filter(pl.col('top_mean_probability')>=.5).height,'accepted_pairs':accepted.height,'accepted_pools':accepted['table_id'].n_unique(),'prior_sample_probability_parity':True,'prior_sample_agreement_parity':True,'train_known_positive_queries':train_known_queries,'seconds':time.perf_counter()-t,'new_fits':0,'held_quality_metrics':0,'agreement_sha256':sha(dest/'agreement.parquet'),'probabilities_sha256':sha(dest/'teacher-probabilities-with-parity.parquet')};write(dest/'receipt.json',rep);reports.append(rep);print(json.dumps(rep),flush=True)
 allx=pl.concat(accepted_all);distinct=allx.unique('pair_id');passed=distinct.height>=15 and distinct['table_id'].n_unique()>=10;allx.write_parquet(OUT/'accepted-all-folds.parquet');result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'YIELD_PASS' if passed else 'INSUFFICIENT_AGREEMENT_YIELD_STOP','new_fits':0,'distinct_accepted_pairs':distinct.height,'distinct_accepted_pools':distinct['table_id'].n_unique(),'accepted_appearances':allx.height,'passes_gate':passed,'folds':reports,'quality_evaluations':0,'student_fits':0,'all_old_sample_controls_exact':True,'next_action':plan['yield_gate']['success' if passed else 'failure']};write(OUT/'yield-result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['init','score']);a=ap.parse_args();{'init':initialize,'score':score}[a.mode]()
