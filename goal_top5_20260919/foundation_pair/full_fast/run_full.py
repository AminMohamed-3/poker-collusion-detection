import os,sys,site,json,time,hashlib,gc,argparse
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',POLARS_MAX_THREADS='2')
ROOT=Path(r'C:\Data\Projects\Money Making'); RT=ROOT/'work/poker/goal_top5_20260919/tabpfn35'; OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/full_fast'; C=ROOT/'work/poker/Claude Version'; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(RT)); import runtime35
import numpy as np, polars as pl, torch
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
SEED=35935; CK=RT/'tabpfn-v3.5-fast-20260909.safetensors'; SHA='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'; assert hashlib.sha256(CK.read_bytes()).hexdigest()==SHA
CUR=ROOT/'work/poker/goal_top5_20260919/outputs/submission_sp277_ci276_on_step272.csv'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def prepare():
 if (OUT/'full_context.npz').exists():
  d=np.load(OUT/'full_context.npz'); return d['X_train'],d['y_train'],d['X_eval'],d['pair_id'].tolist()
 lab=pl.read_csv(ROOT/'work/poker/data/development_labels.csv'); tr=pl.read_parquet(C/'experiments/crossphase_20260911/lab_on_dev.parquet').join(lab.select('pair_id','label','behavior_family'),on='pair_id',validate='1:1')
 fam={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}; y=np.array([fam[v] if z==1 else 0 for z,v in zip(tr['label'],tr['behavior_family'])],dtype=np.int8)
 feats=[c for c,dt in tr.schema.items() if c not in {'pair_id','table_id','label','behavior_family'} and dt.is_numeric()]; assert len(feats)==653
 X=tr.select(feats).to_numpy().astype('float32'); X[~np.isfinite(X)]=np.nan
 sm=LGBMClassifier(objective='multiclass',num_class=4,n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=25,reg_lambda=10,colsample_bytree=.8,random_state=SEED,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True); sm.fit(X,y); imp=sm.booster_.feature_importance(importance_type='gain'); ix=np.array(sorted(range(len(feats)),key=lambda j:(-float(imp[j]),feats[j]))[:128]); names=[feats[j] for j in ix]
 cur=pl.read_csv(CUR,columns=['pair_id']); ev=pl.read_parquet(C/'experiments/crossphase_20260911/eval_on_eval.parquet').join(cur.with_row_index('_order'),on='pair_id',validate='1:1').sort('_order'); assert ev.height==112540 and ev['pair_id'].to_list()==cur['pair_id'].to_list()
 XE=ev.select(names).to_numpy().astype('float32'); XE[~np.isfinite(XE)]=np.nan
 meta={'checkpoint_sha256':SHA,'seed':SEED,'members':4,'train_rows':len(y),'class_counts':{str(k):int((y==k).sum()) for k in range(4)},'feature_names':names,'train_source_sha256':sha(C/'experiments/crossphase_20260911/lab_on_dev.parquet'),'eval_source_sha256':sha(C/'experiments/crossphase_20260911/eval_on_eval.parquet'),'current_sha256':sha(CUR),'train_array_sha256':hashlib.sha256(X[:,ix].tobytes()).hexdigest(),'eval_keys_sha256':hashlib.sha256('\n'.join(ev['pair_id']).encode()).hexdigest()}; (OUT/'context_receipt.json').write_text(json.dumps(meta,indent=2))
 return X[:,ix],y,XE,ev['pair_id'].to_list()
def model(): return TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=SEED,n_preprocessing_jobs=1)
def predict(m,x): return np.concatenate([m.predict_proba(x[i:i+256]) for i in range(0,len(x),256)])
def main():
 a=argparse.ArgumentParser();a.add_argument('--replay',action='store_true');z=a.parse_args(); assert torch.cuda.is_available();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
 X,y,XE,ids=prepare();torch.cuda.reset_peak_memory_stats();t=time.monotonic();m=model();m.fit(X,y); fitsec=time.monotonic()-t
 if z.replay:
  p=predict(m,XE[:1024]);np.savez_compressed(OUT/'replay_first1024.npz',p=p);return
 t=time.monotonic();p0=predict(m,XE[:1024]);sec=time.monotonic()-t;peak=int(torch.cuda.max_memory_reserved());project=sec*len(XE)/1024
 bench={'fit_seconds':fitsec,'first1024_seconds':sec,'projected_full_seconds':project,'peak_reserved_bytes':peak,'pass':project<3600 and peak<3*1024**3};(OUT/'benchmark.json').write_text(json.dumps(bench,indent=2));assert bench['pass'],bench
 B=4096
 for s in range(0,len(XE),B):
  dest=OUT/f'batch_{s//B:03d}.parquet'
  if dest.exists(): continue
  p=predict(m,XE[s:min(s+B,len(XE))]);pl.DataFrame({'pair_id':ids[s:s+len(p)],'p_none':p[:,0],'p_dt':p[:,1],'p_sp':p[:,2],'p_ci':p[:,3]}).write_parquet(dest)
 parts=[pl.read_parquet(p) for p in sorted(OUT.glob('batch_*.parquet'))];allp=pl.concat(parts);assert allp.height==112540 and allp['pair_id'].to_list()==ids;allp.write_parquet(OUT/'fast_probabilities.parquet')
 (OUT/'runtime.json').write_text(json.dumps({'fit_seconds':fitsec,'benchmark':bench,'total_seconds':time.monotonic()-t,'peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'batches':len(parts)},indent=2))
if __name__=='__main__':main()
