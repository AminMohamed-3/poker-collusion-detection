import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
import sys,json,time,hashlib,argparse,gc
from pathlib import Path
ROOT=Path(r'C:\Data\Projects\Money Making');RT=ROOT/'work/poker/goal_top5_20260919/tabpfn35';sys.path.insert(0,str(RT));import runtime35
import numpy as np,torch
from tabpfn import TabPFNClassifier
OUT=Path(__file__).resolve().parent;CK=RT/'tabpfn-v3.5-fast-20260909.safetensors';EXPECT='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550';SEED=35935
assert hashlib.sha256(CK.read_bytes()).hexdigest()==EXPECT and torch.cuda.is_available();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
def fixed(p,c):
 q=np.zeros((len(p),4),dtype=np.float64)
 for j,k in enumerate(c):q[:,int(k)]=p[:,j]
 return q
def predict(m,X):
 a=[]
 for i in range(0,len(X),256):a.append(m.predict_proba(X[i:i+256]))
 return fixed(np.concatenate(a),np.asarray(m.classes_))
def fit(f,b):
 d=np.load(OUT/f'context_fold{f}_bag{b}.npz',allow_pickle=False);torch.cuda.reset_peak_memory_stats();t=time.monotonic();m=TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=SEED,n_preprocessing_jobs=1);m.fit(d['X_train'],d['y_train']);p=predict(m,d['X_held']);pre=np.load(OUT/f'context_fold{f}_bag{b}_prefix60.npz',allow_pickle=False);pp=predict(m,pre['X_held']);peak=int(torch.cuda.max_memory_reserved());secs=time.monotonic()-t;assert peak<3*1024**3 and np.isfinite(p).all() and np.isfinite(pp).all();return p,pp,secs,peak
def main():
 a=argparse.ArgumentParser();a.add_argument('--replay',action='store_true');z=a.parse_args()
 if z.replay:
  p,pp,s,peak=fit(0,0);np.savez_compressed(OUT/'replay_fold0_bag0.npz',known=p,prefix60=pp);(OUT/'replay_runtime.json').write_text(json.dumps({'seconds':s,'peak_reserved_bytes':peak},indent=2)+'\n');return
 arr={};rec=[];start=time.monotonic()
 for f in (0,1):
  ps=[];pps=[]
  for b in range(4):
   p,pp,s,peak=fit(f,b);arr[f'known_fold{f}_bag{b}']=p;arr[f'prefix60_fold{f}_bag{b}']=pp;ps.append(p);pps.append(pp);rec.append({'fold':f,'bag':b,'seconds':s,'peak_reserved_bytes':peak});gc.collect();torch.cuda.empty_cache()
   if f==0 and b==0 and s*8>600:raise RuntimeError('projected runtime exceeds ten-minute cap')
  arr[f'known_fold{f}']=np.mean(np.stack(ps),axis=0,dtype=np.float64);arr[f'prefix60_fold{f}']=np.mean(np.stack(pps),axis=0,dtype=np.float64)
 np.savez_compressed(OUT/'balanced_predictions.npz',**arr);(OUT/'runtime.json').write_text(json.dumps({'checkpoint_sha256':EXPECT,'seed':SEED,'members_per_bag':4,'bags_per_fold':4,'average_dtype':'float64','memory_fraction':.7,'batch':256,'total_seconds':time.monotonic()-start,'fits':rec},indent=2)+'\n')
if __name__=='__main__':main()
