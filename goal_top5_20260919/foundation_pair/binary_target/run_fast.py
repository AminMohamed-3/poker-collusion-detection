import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
import sys,json,time,hashlib,argparse,gc
from pathlib import Path
ROOT=Path(r'C:/Data/Projects/Money Making');OUT=Path(__file__).resolve().parent;RT=ROOT/'work/poker/goal_top5_20260919/tabpfn35'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((OUT/'prep_manifest.json').read_text());release=json.loads((OUT/'GPU_RELEASE.json').read_text()) if (OUT/'GPU_RELEASE.json').exists() else {}
assert release.get('prep_manifest_sha256')==sha(OUT/'prep_manifest.json'),'explicit hash-bound root GPU release absent'
for n,h in manifest['artifact_hashes'].items():assert sha(OUT/n)==h,(n,'hash mismatch')
for n,h in manifest['frozen_source_hashes'].items():assert sha(OUT/n)==h,(n,'source hash mismatch')
sys.path.insert(0,str(RT));import runtime35
import numpy as np,torch
from tabpfn import TabPFNClassifier
CK=RT/'tabpfn-v3.5-fast-20260909.safetensors';EXPECT='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550';assert sha(CK)==EXPECT and torch.cuda.is_available();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
def pred(m,X):
 p=np.concatenate([m.predict_proba(X[i:i+256]) for i in range(0,len(X),256)]).astype(np.float64);c=np.asarray(m.classes_,dtype=np.int8);risk=1.0-p[:,list(c).index(0)];return risk,p,c
def fit(f):
 d=np.load(OUT/f'context_fold{f}.npz');t=time.monotonic();torch.cuda.reset_peak_memory_stats();m=TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=35935,n_preprocessing_jobs=1);m.fit(d['X_train'],d['y_train']);a,pa,c=pred(m,d['X_held']);q=np.load(OUT/f'query_prefix_fold{f}.npz');b,pb,c2=pred(m,q['X_held']);peak=int(torch.cuda.max_memory_reserved());s=time.monotonic()-t;assert peak<3*1024**3 and np.isfinite(a).all() and np.isfinite(b).all() and np.array_equal(c,c2);return a,b,pa,pb,c,s,peak
def main():
 z=argparse.ArgumentParser();z.add_argument('--replay',action='store_true');a=z.parse_args()
 if a.replay:
  assert (OUT/'predictions.npz').exists() and not (OUT/'replay.npz').exists() and not (OUT/'replay_runtime.json').exists();p,q,pp,pq,c,s,m=fit(0);src=np.load(OUT/'predictions.npz');assert np.array_equal(p,src['known_fold0']) and np.array_equal(q,src['prefix_fold0']);np.savez_compressed(OUT/'replay.npz',known=p,prefix=q,known_probabilities=pp,prefix_probabilities=pq,classes=c);(OUT/'replay_runtime.json').write_text(json.dumps({'exact':True,'seconds':s,'peak_reserved_bytes':m},indent=2));return
 assert not (OUT/'predictions.npz').exists() and not (OUT/'runtime.json').exists();arr={};rec=[];t=time.monotonic()
 for f in (0,1):
  p,q,pp,pq,c,s,m=fit(f);arr[f'known_fold{f}']=p;arr[f'prefix_fold{f}']=q;arr[f'known_probabilities_fold{f}']=pp;arr[f'prefix_probabilities_fold{f}']=pq;arr[f'classes_fold{f}']=c;rec.append({'fold':f,'seconds':s,'peak_reserved_bytes':m});gc.collect();torch.cuda.empty_cache()
  if f==0 and 2*s>300:raise RuntimeError('projected runtime exceeds five minutes')
  if time.monotonic()-t>300:raise RuntimeError('cumulative runtime exceeds five minutes')
 np.savez_compressed(OUT/'predictions.npz',**arr);(OUT/'runtime.json').write_text(json.dumps({'checkpoint_sha256':EXPECT,'seed':35935,'members':4,'risk_definition':'1 - float64(P(class0))','total_seconds':time.monotonic()-t,'fits':rec},indent=2)+'\n')
if __name__=='__main__':main()
