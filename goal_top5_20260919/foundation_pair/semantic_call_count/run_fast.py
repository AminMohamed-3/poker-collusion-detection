import os,sys,json,time,hashlib,gc,argparse
from pathlib import Path
ROOT=Path(r'C:/Data/Projects/Money Making'); OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/semantic_call_count'; RT=ROOT/'work/poker/goal_top5_20260919/tabpfn35'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=json.loads((OUT/'freeze.json').read_text()); release=json.loads((OUT/'GPU_RELEASE.json').read_text()) if (OUT/'GPU_RELEASE.json').exists() else {}
assert release.get('freeze_sha256')==sha(OUT/'freeze.json'),'hash-bound root GPU release absent'
for n,h in freeze['files'].items(): assert sha(OUT/n)==h,(n,'hash mismatch')
sys.path.insert(0,str(RT)); import runtime35
import numpy as np,torch
from tabpfn import TabPFNClassifier
CK=RT/'tabpfn-v3.5-fast-20260909.safetensors'; EXPECT='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'; assert sha(CK)==EXPECT
torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.70); SEED=35935
def one(path):
 d=np.load(path); torch.cuda.reset_peak_memory_stats(); t=time.monotonic(); m=TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=SEED,n_preprocessing_jobs=1);m.fit(d['X_train'],d['y_train']); out=[]
 for name in ('X_full','X_prefix'):
  z=[]
  for i in range(0,len(d[name]),256): z.append(m.predict_proba(d[name][i:i+256]))
  pp=np.concatenate(z);q=np.zeros((len(pp),4))
  for j,k in enumerate(m.classes_):q[:,int(k)]=pp[:,j]
  assert np.isfinite(q).all() and np.allclose(q.sum(1),1);out.append(q)
 peak=int(torch.cuda.max_memory_reserved());assert peak<3*1024**3
 return out,time.monotonic()-t,peak
def main():
 a=argparse.ArgumentParser();a.add_argument('--replay',action='store_true');z=a.parse_args()
 if z.replay:
  assert (OUT/'predictions.npz').exists() and not (OUT/'replay.npz').exists() and not (OUT/'replay_receipt.json').exists(); (p,s,pk)=one(OUT/'context_fold0.npz'); src=np.load(OUT/'predictions.npz'); assert np.array_equal(p[0],src['full_fold0']) and np.array_equal(p[1],src['prefix_fold0']);np.savez_compressed(OUT/'replay.npz',full=p[0],prefix=p[1]);(OUT/'replay_receipt.json').write_text(json.dumps({'exact':True,'seconds':s,'peak_reserved':pk},indent=2)+'\n');return
 assert not (OUT/'predictions.npz').exists() and not (OUT/'runtime.json').exists();arr={};rec=[];start=time.monotonic()
 for f in (0,1):
  p,s,pk=one(OUT/f'context_fold{f}.npz');arr[f'full_fold{f}']=p[0];arr[f'prefix_fold{f}']=p[1];rec.append({'fold':f,'seconds':s,'peak_reserved':pk});gc.collect();torch.cuda.empty_cache()
  if len(rec)==1 and s*2>300:raise RuntimeError('projected runtime exceeds 5 minutes')
  if time.monotonic()-start>300:raise RuntimeError('cumulative runtime exceeds 5 minutes')
 np.savez_compressed(OUT/'predictions.npz',**arr);(OUT/'runtime.json').write_text(json.dumps({'checkpoint_sha256':EXPECT,'seed':SEED,'members':4,'batch':256,'memory_fraction':.70,'total_seconds':time.monotonic()-start,'fits':rec},indent=2)+'\n')
if __name__=='__main__':main()
