import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',POLARS_MAX_THREADS='2')
import sys,site,hashlib,json,glob
from pathlib import Path
site.addsitedir(r'C:\Data\Projects\Money Making\work\poker\.venv\Lib\site-packages')
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
ROOT=Path(r'C:\Data\Projects\Money Making'); OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/current_comparison'; SCREEN=OUT.parent/'screen'
C=ROOT/'work/poker/Claude Version'; SALT='foundation-pf35-screen-v1:'
def fold(t): return int(hashlib.sha256((SALT+t).encode()).hexdigest()[:16],16)%2
def inner(t): return int(hashlib.sha256(('foundation-current-inner-v1:'+t).encode()).hexdigest()[:16],16)%2
def selector(seed): return LGBMClassifier(objective='multiclass',num_class=4,n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=25,reg_lambda=10,colsample_bytree=.8,random_state=seed,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 inst=pl.read_parquet(C/'experiments/viewdis_20260911/prefix_rescoring.parquet')
 raws=[]
 for f in (1,2):
  files=sorted(glob.glob(str(C/f'experiments/pu_pairs_20260911/fold{f}-prefix60/batch_*-pairs.parquet')))
  raws.append(pl.concat([pl.read_parquet(x) for x in files],how='vertical_relaxed').with_columns(pl.lit(f).alias('source_fold')))
 raw=pl.concat(raws,how='vertical_relaxed')
 # Some IDs have two legitimate conditional-policy views. Match the exact
 # historical instrument fold explicitly instead of relying on file order.
 got=inst.select('pair_id','table_id','fold').join(raw,left_on=['pair_id','table_id','fold'],right_on=['pair_id','table_id','source_fold'],validate='1:1'); assert got.height==779
 ev_ids=pl.read_parquet(C/'experiments/spillover_20260911/eval_rescoring.parquet').select('pair_id','table_id')
 ev=pl.read_parquet(C/'experiments/crossphase_20260911/eval_on_eval.parquet').join(ev_ids,on=['pair_id','table_id'],validate='1:1'); assert ev.height==909
 devtabs=set(pl.read_parquet(C/'experiments/crossphase_20260911/lab_on_dev.parquet',columns=['table_id'])['table_id'].to_list())
 eval_overlap=len(devtabs & set(ev['table_id'].to_list()))
 manifest=[]
 for f in (0,1):
  base=np.load(SCREEN/f'context_fold{f}.npz',allow_pickle=False); names=base['feature_name'].tolist(); assert len(names)==128
  m=np.array([fold(t)==f for t in inst['table_id'].to_list()]); held=got.filter(pl.Series(m)).select(names).to_numpy().astype('float32'); held[~np.isfinite(held)]=np.nan
  me=np.array([fold(t)==f for t in ev['table_id'].to_list()]); xe=ev.filter(pl.Series(me)).select(names).to_numpy().astype('float32'); xe[~np.isfinite(xe)]=np.nan
  np.savez_compressed(OUT/f'pf_context_fold{f}.npz',X_train=base['X_train'],y_train=base['y_train'],X_instrument=held,instrument_pair_id=np.array(inst['pair_id'].to_list(),dtype='U')[m],instrument_table_id=np.array(inst['table_id'].to_list(),dtype='U')[m],X_eval909=xe,eval_pair_id=np.array(ev['pair_id'].to_list(),dtype='U')[me],eval_table_id=np.array(ev['table_id'].to_list(),dtype='U')[me],feature_name=np.array(names,dtype='U'))
  manifest.append({'fold':f,'instrument_rows':int(m.sum()),'instrument_tables':len(set(np.array(inst['table_id'].to_list())[m])),'eval_rows':int(me.sum()),'eval_tables':len(set(np.array(ev['table_id'].to_list())[me])),'train_rows':len(base['y_train']),'features':128})
 # Nested PF meta-training contexts. Labels/selectors are restricted to the
 # outer-training half, then to the opposite inner table half.
 ld=pl.read_parquet(C/'experiments/crossphase_20260911/lab_on_dev.parquet').join(pl.read_csv(ROOT/'work/poker/data/development_labels.csv').select('pair_id','label','behavior_family'),on='pair_id',validate='1:1')
 fam={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}; yy=np.array([fam[v] if z==1 else 0 for z,v in zip(ld['label'],ld['behavior_family'])],dtype=np.int8)
 feats=[c for c,dt in ld.schema.items() if c not in {'pair_id','table_id','label','behavior_family'} and dt.is_numeric()]; assert len(feats)==653
 XX=ld.select(feats).to_numpy().astype('float32'); XX[~np.isfinite(XX)]=np.nan; ot=np.array([fold(t) for t in ld['table_id']],dtype=np.int8); it=np.array([inner(t) for t in ld['table_id']],dtype=np.int8); ih=np.array([inner(t) for t in inst['table_id']],dtype=np.int8); oh=np.array([fold(t) for t in inst['table_id']],dtype=np.int8)
 for outer in (0,1):
  for inn in (0,1):
   tr=(ot!=outer)&(it!=inn); hs=(oh!=outer)&(ih==inn); sm=selector(45000+10*outer+inn); sm.fit(XX[tr],yy[tr]); imp=sm.booster_.feature_importance(importance_type='gain'); order=sorted(range(len(feats)),key=lambda j:(-float(imp[j]),feats[j])); ix=np.array(order[:128]); xh=got.filter(pl.Series(hs)).select([feats[j] for j in ix]).to_numpy().astype('float32'); xh[~np.isfinite(xh)]=np.nan
   np.savez_compressed(OUT/f'pf_context_outer{outer}_inner{inn}.npz',X_train=XX[tr][:,ix],y_train=yy[tr],X_instrument=xh,instrument_pair_id=np.array(inst['pair_id'].to_list(),dtype='U')[hs],feature_name=np.array([feats[j] for j in ix],dtype='U'))
   manifest.append({'outer':outer,'inner':inn,'kind':'nested_meta_train','train_rows':int(tr.sum()),'instrument_rows':int(hs.sum()),'features':128})
 lab=pl.read_csv(ROOT/'work/poker/data/development_labels.csv').select('pair_id','label','behavior_family','label_status')
 e=inst.select('pair_id','table_id','shared_hands','cls','risk_prefix','risk_full').join(lab,on='pair_id',how='left')
 e.write_parquet(OUT/'instrument_evaluator.parquet')
 receipt={'salt':SALT,'routing':'physical table hash, never legacy fold integer; eval rows also routed to held hash fold because phases share tables','prefix_policy_view':'explicit source fold matched to instrument fold; no implicit unique/first','instrument_rows':779,'instrument_tables':inst['table_id'].n_unique(),'prefix_raw_coverage':got.height,'eval_rows':909,'eval_tables':ev['table_id'].n_unique(),'eval_tables_overlapping_development':eval_overlap,'confirmed_instrument_rows':e.filter(pl.col('label').is_not_null()).height,'confirmed_positive':e.filter(pl.col('label')==1).height,'confirmed_negative':e.filter(pl.col('label')==0).height,'contexts':manifest,'global_eval_raw_cache':str(C/'experiments/crossphase_20260911/eval_on_eval.parquet')}
 (OUT/'prep_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
if __name__=='__main__': main()
