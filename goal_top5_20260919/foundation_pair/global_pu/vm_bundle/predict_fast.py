"""Portable Fast 3.5 predictor. Held labels are deliberately absent."""
import argparse, gc, hashlib, json, os, time
from pathlib import Path
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="2"
os.environ.update(HF_HUB_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",TABPFN_NO_BROWSER="1")
import numpy as np
import torch
from tabpfn import TabPFNClassifier
HERE=Path(__file__).resolve().parent; C=HERE/'contexts'; O=HERE/'outputs'; O.mkdir(exist_ok=True)
MODEL=HERE/'tabpfn-v3.5-fast-20260909.safetensors'; EXPECT='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'; SEED=35935
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def model():
 return TabPFNClassifier(model_path=str(MODEL),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=SEED,n_preprocessing_jobs=1)
def fixed4(p,c):
 q=np.zeros((len(p),4));q[:,np.asarray(c,int)]=p;assert np.isfinite(q).all() and np.allclose(q.sum(1),1,atol=1e-5);return q
def chunks(m,x,b): return fixed4(np.concatenate([m.predict_proba(x[i:i+b]) for i in range(0,len(x),b)]),m.classes_)
def fold_run(f,batch):
 tr=np.load(C/f'train_fold{f}.npz',allow_pickle=False); torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(); t0=time.monotonic();m=model();m.fit(tr['X_train'],tr['y_train']);fit_s=time.monotonic()-t0
 files=sorted(C.glob(f'query_fold{f}_*.npz'));total=sum(len(np.load(p,allow_pickle=False)['pair_id']) for p in files); first=np.load(files[0],allow_pickle=False); t=time.monotonic();p0=chunks(m,first['X_held'][:1024],batch);sec=time.monotonic()-t
 projected=2*fit_s+sec/1024*total*2; gate={'fold':f,'fit_seconds':fit_s,'first1024_seconds':sec,'all_two_fold_projected_seconds':projected,'limit_seconds':3600}
 (O/f'projection_fold{f}.json').write_text(json.dumps(gate,indent=2)+'\n');print(gate,flush=True)
 if projected>3600: raise RuntimeError('Projected two-fold runtime exceeds frozen 60-minute gate')
 rec=[]
 for n,path in enumerate(files):
  d=np.load(path,allow_pickle=False); dest=O/f'pred_fold{f}_{n:03}.npz'
  if dest.exists():
   old=np.load(dest,allow_pickle=False);assert np.array_equal(old['pair_id'],d['pair_id']) and np.array_equal(old['table_id'],d['table_id']) and old['probabilities'].shape==(len(d['pair_id']),4)
   rec.append({'file':dest.name,'resumed':True,'sha256':sha(dest)});continue
  start=time.monotonic(); p=chunks(m,d['X_held'],batch); np.savez_compressed(dest,probabilities=p,pair_id=d['pair_id'],table_id=d['table_id'],classes=np.arange(4,dtype=np.int8))
  rec.append({'file':dest.name,'rows':len(p),'seconds':time.monotonic()-start,'sha256':sha(dest)})
 meta={'fold':f,'seed':SEED,'members':4,'batch':batch,'fit_mode':'low_memory','memory_saving_mode':True,'checkpoint_sha256':sha(MODEL),'train_sha256':sha(C/f'train_fold{f}.npz'),'peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'shards':rec}
 (O/f'runtime_fold{f}.json').write_text(json.dumps(meta,indent=2)+'\n');del m;gc.collect();torch.cuda.empty_cache()
def replay(f,batch):
 tr=np.load(C/f'train_fold{f}.npz',allow_pickle=False);q=np.load(C/f'query_fold{f}_000.npz',allow_pickle=False);m=model();m.fit(tr['X_train'],tr['y_train']);p=chunks(m,q['X_held'][:1024],batch);old=np.load(O/f'pred_fold{f}_000.npz',allow_pickle=False)['probabilities'][:1024];check={'fold':f,'exact':bool(np.array_equal(p,old)),'max_abs':float(np.max(np.abs(p-old)))};(O/f'replay_fold{f}.json').write_text(json.dumps(check,indent=2)+'\n');assert check['exact'],check
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['run','replay']);a.add_argument('--batch',type=int,choices=[64,256],default=256);z=a.parse_args();assert torch.cuda.is_available() and sha(MODEL)==EXPECT;torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
 if z.mode=='run':
  for f in (1,2):fold_run(f,z.batch)
 else:
  for f in (1,2):replay(f,z.batch)
if __name__=='__main__':main()
