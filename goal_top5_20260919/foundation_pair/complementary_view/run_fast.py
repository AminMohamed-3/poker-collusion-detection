import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
import sys,json,time,hashlib,argparse,gc
from pathlib import Path
ROOT=Path(r'C:\Data\Projects\Money Making');RT=ROOT/'work/poker/goal_top5_20260919/tabpfn35';sys.path.insert(0,str(RT));import runtime35
import numpy as np,torch
from tabpfn import TabPFNClassifier
OUT=Path(__file__).resolve().parent;CK=RT/'tabpfn-v3.5-fast-20260909.safetensors';EXPECT='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550';SEED=35935
assert hashlib.sha256(CK.read_bytes()).hexdigest()==EXPECT and torch.cuda.is_available();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
def fixed4(p,c):
 q=np.zeros((len(p),4));
 for j,k in enumerate(c):q[:,int(k)]=p[:,j]
 return q
def fit(path):
 d=np.load(path,allow_pickle=False);torch.cuda.reset_peak_memory_stats();t=time.monotonic()
 m=TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=SEED,n_preprocessing_jobs=1);m.fit(d['X_train'],d['y_train']);out=[]
 for i in range(0,len(d['X_held']),256):out.append(m.predict_proba(d['X_held'][i:i+256]))
 p=np.concatenate(out);peak=int(torch.cuda.max_memory_reserved());secs=time.monotonic()-t;assert peak<3*1024**3 and np.isfinite(p).all();return m,fixed4(p,np.asarray(m.classes_)),secs,peak
def main():
 a=argparse.ArgumentParser();a.add_argument('--replay',action='store_true');z=a.parse_args()
 if z.replay:
  _,p,s,peak=fit(OUT/'context_fold0.npz');np.savez_compressed(OUT/'replay_fold0.npz',p=p);(OUT/'replay_runtime.json').write_text(json.dumps({'seconds':s,'peak_reserved_bytes':peak},indent=2));return
 arr={};rec=[];start=time.monotonic()
 for f in (0,1):
  m,p,s,peak=fit(OUT/f'context_fold{f}.npz');arr[f'known_fold{f}']=p;rec.append({'name':f'known_fold{f}','seconds':s,'peak_reserved_bytes':peak})
  d=np.load(OUT/f'context_fold{f}_prefix60.npz',allow_pickle=False);parts=[];t=time.monotonic();torch.cuda.reset_peak_memory_stats()
  for i in range(0,len(d['X_held']),256):parts.append(m.predict_proba(d['X_held'][i:i+256]))
  pp=fixed4(np.concatenate(parts),np.asarray(m.classes_));peak2=int(torch.cuda.max_memory_reserved());assert peak2<3*1024**3;arr[f'prefix60_fold{f}']=pp;rec.append({'name':f'prefix60_fold{f}','seconds':time.monotonic()-t,'peak_reserved_bytes':peak2,'reused_fit':True});del m;gc.collect();torch.cuda.empty_cache()
  if f==0 and (time.monotonic()-start)*2>600:raise RuntimeError('projected runtime exceeds ten-minute cap')
 np.savez_compressed(OUT/'complementary_predictions.npz',**arr);(OUT/'runtime.json').write_text(json.dumps({'checkpoint_sha256':EXPECT,'seed':SEED,'members':4,'memory_fraction':.7,'batch':256,'total_seconds':time.monotonic()-start,'contexts':rec},indent=2)+'\n')
if __name__=='__main__':main()
