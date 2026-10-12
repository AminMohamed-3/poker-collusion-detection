import os, sys, json, time, hashlib, gc, argparse
from pathlib import Path
ROOT=Path(r'C:\Data\Projects\Money Making')
RT=ROOT/'work/poker/goal_top5_20260919/tabpfn35'
sys.path.insert(0,str(RT)); import runtime35  # noqa: E402
import numpy as np
import torch
from tabpfn import TabPFNClassifier

OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/screen'; SEED=35935
CK=RT/'tabpfn-v3.5-fast-20260909.safetensors'; EXPECT='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'
assert hashlib.sha256(CK.read_bytes()).hexdigest()==EXPECT and torch.cuda.is_available()
torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.70)
def fit(ctx,seed=SEED):
 d=np.load(ctx,allow_pickle=False); torch.cuda.reset_peak_memory_stats(); t=time.monotonic()
 m=TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=seed,n_preprocessing_jobs=1)
 m.fit(d['X_train'],d['y_train']); out=[]
 for i in range(0,len(d['X_held']),256): out.append(m.predict_proba(d['X_held'][i:i+256]))
 p=np.concatenate(out); classes=np.asarray(m.classes_,dtype=np.int8); elapsed=time.monotonic()-t; peak=int(torch.cuda.max_memory_reserved())
 assert np.isfinite(p).all() and peak<3*1024**3
 return m,p,classes,elapsed,peak
def fixed4(p,c):
 q=np.zeros((len(p),4));
 for j,k in enumerate(c): q[:,int(k)]=p[:,j]
 return q
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--replay-fold0',action='store_true'); a=ap.parse_args()
 if a.replay_fold0:
  m,p,c,s,peak=fit(OUT/'context_fold0.npz'); np.savez_compressed(OUT/'replay_fold0.npz',p=fixed4(p,c),classes=c)
  (OUT/'replay_fold0_runtime.json').write_text(json.dumps({'seconds':s,'peak_reserved_bytes':peak,'checkpoint_sha256':EXPECT},indent=2)); return
 rec=[]; known={}; arrays={}
 for f in (0,1):
  m,p,c,s,peak=fit(OUT/f'context_fold{f}.npz'); known[f]=m; arrays[f'known_fold{f}']=fixed4(p,c); rec.append({'context':f'known_fold{f}','seconds':s,'peak_reserved_bytes':peak,'classes':c.tolist()})
  d=np.load(OUT/f'context_fold{f}_prefix60.npz'); pp=[]; t=time.monotonic(); torch.cuda.reset_peak_memory_stats()
  for i in range(0,len(d['X_held']),256): pp.append(m.predict_proba(d['X_held'][i:i+256]))
  pp=np.concatenate(pp); peak2=int(torch.cuda.max_memory_reserved()); assert peak2<3*1024**3
  arrays[f'prefix60_fold{f}']=fixed4(pp,c); rec.append({'context':f'prefix60_fold{f}','seconds':time.monotonic()-t,'peak_reserved_bytes':peak2,'classes':c.tolist(),'reused_fit':True})
 for f in (0,1):
  for omit in (1,2,3):
   m,p,c,s,peak=fit(OUT/f'context_fold{f}_omit{omit}.npz'); arrays[f'omit{omit}_fold{f}']=fixed4(p,c); rec.append({'context':f'omit{omit}_fold{f}','seconds':s,'peak_reserved_bytes':peak,'classes':c.tolist()}); del m; gc.collect(); torch.cuda.empty_cache()
 # Estimated run is short after eight fits, so preserve the frozen shuffled controls.
 rng=np.random.default_rng(SEED)
 for f in (0,1):
  d=np.load(OUT/f'context_fold{f}.npz'); yp=rng.permutation(d['y_train']); tmp=OUT/f'_shuffle_fold{f}.npz'; np.savez_compressed(tmp,X_train=d['X_train'],y_train=yp,X_held=d['X_held'])
  m,p,c,s,peak=fit(tmp); arrays[f'shuffle_fold{f}']=fixed4(p,c); rec.append({'context':f'shuffle_fold{f}','seconds':s,'peak_reserved_bytes':peak,'classes':c.tolist()}); tmp.unlink(); del m; gc.collect(); torch.cuda.empty_cache()
 np.savez_compressed(OUT/'pf35_predictions.npz',**arrays)
 (OUT/'pf35_runtime.json').write_text(json.dumps({'seed':SEED,'members':4,'checkpoint_sha256':EXPECT,'memory_fraction':.7,'batch':256,'contexts':rec},indent=2))
if __name__=='__main__': main()
