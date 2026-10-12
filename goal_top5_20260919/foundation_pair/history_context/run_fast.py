import os,sys,json,time,hashlib,gc,argparse
from pathlib import Path
ROOT=Path(r'C:/Data/Projects/Money Making'); OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/history_context'; RT=ROOT/'work/poker/goal_top5_20260919/tabpfn35'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=json.loads((OUT/'freeze.json').read_text());release=json.loads((OUT/'GPU_RELEASE.json').read_text()) if (OUT/'GPU_RELEASE.json').exists() else {}
assert release.get('freeze_sha256')==sha(OUT/'freeze.json'),'explicit hash-bound root GPU release absent'
for n,h in freeze['files'].items():assert sha(OUT/n)==h,(n,'hash mismatch')
sys.path.insert(0,str(RT)); import runtime35
import numpy as np,torch
from tabpfn import TabPFNClassifier
CK=RT/'tabpfn-v3.5-fast-20260909.safetensors'; EXPECT='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'; assert hashlib.sha256(CK.read_bytes()).hexdigest()==EXPECT
torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.70); SEED=35935
def one(path):
 d=np.load(path); torch.cuda.reset_peak_memory_stats(); t=time.monotonic(); m=TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=SEED,n_preprocessing_jobs=1); m.fit(d['X_train'],d['y_train']); out=[]
 for name in ('X_full','X_prefix'):
  p=[]
  for i in range(0,len(d[name]),256): p.append(m.predict_proba(d[name][i:i+256]))
  q=np.zeros((len(d[name]),4)); pp=np.concatenate(p)
  for j,k in enumerate(m.classes_): q[:,int(k)]=pp[:,j]
  assert np.isfinite(q).all() and np.allclose(q.sum(1),1); out.append(q)
 peak=int(torch.cuda.max_memory_reserved()); assert peak<3*1024**3
 return out,time.monotonic()-t,peak
def main():
 a=argparse.ArgumentParser();a.add_argument('--replay',action='store_true');z=a.parse_args()
 if z.replay:
  assert not (OUT/'replay.npz').exists() and not (OUT/'replay_receipt.json').exists()
  (pp,s,pk)=one(OUT/'context_prefix_fold0.npz'); src=np.load(OUT/'predictions.npz')['prefix_full_fold0']; assert np.array_equal(pp[0],src); np.savez_compressed(OUT/'replay.npz',p=pp[0]); (OUT/'replay_receipt.json').write_text(json.dumps({'exact':True,'seconds':s,'peak':pk},indent=2)); return
 assert not (OUT/'predictions.npz').exists() and not (OUT/'runtime.json').exists()
 arr={}; rec=[]; start=time.monotonic()
 for arm in ('prefix','duplicate'):
  for f in (0,1):
   (p,s,pk)=one(OUT/f'context_{arm}_fold{f}.npz'); arr[f'{arm}_full_fold{f}']=p[0];arr[f'{arm}_prefix_fold{f}']=p[1];rec.append({'arm':arm,'fold':f,'seconds':s,'peak_reserved':pk});gc.collect();torch.cuda.empty_cache()
   if len(rec)==1 and s*4>600: raise RuntimeError('projected runtime exceeds 10 minutes')
   if time.monotonic()-start>600: raise RuntimeError('cumulative runtime exceeds 10 minutes')
 np.savez_compressed(OUT/'predictions.npz',**arr); (OUT/'runtime.json').write_text(json.dumps({'checkpoint_sha256':EXPECT,'seed':SEED,'members':4,'batch':256,'memory_fraction':.70,'total_seconds':time.monotonic()-start,'fits':rec},indent=2))
if __name__=='__main__':main()
