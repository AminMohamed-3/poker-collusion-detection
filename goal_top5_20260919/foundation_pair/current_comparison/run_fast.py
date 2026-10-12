import os,sys,json,time,hashlib,gc
from pathlib import Path
ROOT=Path(r'C:\Data\Projects\Money Making'); RT=ROOT/'work/poker/goal_top5_20260919/tabpfn35'; sys.path.insert(0,str(RT)); import runtime35
import numpy as np, torch
from tabpfn import TabPFNClassifier
OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/current_comparison'; SEED=35935
CK=RT/'tabpfn-v3.5-fast-20260909.safetensors'; SHA='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'
assert hashlib.sha256(CK.read_bytes()).hexdigest()==SHA and torch.cuda.is_available(); torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.7)
def pred(m,x): return np.concatenate([m.predict_proba(x[i:i+256]) for i in range(0,len(x),256)])
def fixed(p,c):
 q=np.zeros((len(p),4));
 for j,k in enumerate(c): q[:,int(k)]=p[:,j]
 return q
def main():
 arr={}; rec=[]
 for f in (0,1):
  d=np.load(OUT/f'pf_context_fold{f}.npz'); torch.cuda.reset_peak_memory_stats(); t=time.monotonic()
  m=TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=SEED,n_preprocessing_jobs=1)
  m.fit(d['X_train'],d['y_train']); pi=fixed(pred(m,d['X_instrument']),m.classes_); pe=fixed(pred(m,d['X_eval909']),m.classes_)
  arr[f'instrument_p{f}']=pi; arr[f'instrument_id{f}']=d['instrument_pair_id']; arr[f'eval_p{f}']=pe; arr[f'eval_id{f}']=d['eval_pair_id']
  peak=int(torch.cuda.max_memory_reserved()); assert peak<3*1024**3
  rec.append({'fold':f,'seconds':time.monotonic()-t,'peak_reserved_bytes':peak,'instrument_rows':len(pi),'eval_rows':len(pe),'classes':[int(x) for x in m.classes_]})
  del m; gc.collect(); torch.cuda.empty_cache()
 for outer in (0,1):
  for inn in (0,1):
   d=np.load(OUT/f'pf_context_outer{outer}_inner{inn}.npz'); torch.cuda.reset_peak_memory_stats(); t=time.monotonic()
   m=TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=SEED,n_preprocessing_jobs=1)
   m.fit(d['X_train'],d['y_train']); pi=fixed(pred(m,d['X_instrument']),m.classes_)
   arr[f'outer{outer}_inner{inn}_p']=pi; arr[f'outer{outer}_inner{inn}_id']=d['instrument_pair_id']
   peak=int(torch.cuda.max_memory_reserved()); assert peak<3*1024**3
   rec.append({'outer':outer,'inner':inn,'kind':'nested_meta_train','seconds':time.monotonic()-t,'peak_reserved_bytes':peak,'instrument_rows':len(pi),'classes':[int(x) for x in m.classes_]})
   del m; gc.collect(); torch.cuda.empty_cache()
 np.savez_compressed(OUT/'fast_predictions.npz',**arr); (OUT/'fast_runtime.json').write_text(json.dumps({'checkpoint_sha256':SHA,'seed':SEED,'members':4,'contexts':rec},indent=2))
if __name__=='__main__': main()
