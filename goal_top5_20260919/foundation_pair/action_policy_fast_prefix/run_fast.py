import sys,json,hashlib,gc,time,argparse
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[2];AP=H.parent/'action_policy_fast';R=P/'goal_top5_20260919/tabpfn35';sys.path.insert(0,str(R));import runtime35
import numpy as np,torch
from tabpfn import TabPFNClassifier
CK=R/'tabpfn-v3.5-fast-20260909.safetensors';EXP='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(f):
 d=np.load(H/f'context_scored_prefix_fold{f}.npz',allow_pickle=False);full=np.load(AP/f'fast129_context_fold{f}.npz',allow_pickle=False);assert np.array_equal(d['X_train'],full['X_train'],equal_nan=True);m=TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=35935,n_preprocessing_jobs=1);m.fit(d['X_train'],d['y_train']);pred=lambda X:np.concatenate([m.predict_proba(X[i:i+256]) for i in range(0,len(X),256)]);p=pred(d['X_held']);q=pred(full['X_held']);assert np.array_equal(m.classes_,np.arange(4));
 for z,n in [(p,len(d['X_held'])),(q,len(full['X_held']))]:assert z.shape==(n,4) and np.isfinite(z).all() and (z>=0).all() and (z<=1).all() and np.allclose(z.sum(1),1,atol=1e-5)
 return p,q
def main():
 a=argparse.ArgumentParser();a.add_argument('--replay',action='store_true');x=a.parse_args();st=time.monotonic();man=H/'GPU_MANIFEST.json';rel=H/'GPU_RELEASE.json';assert rel.exists() and json.loads(rel.read_text())['manifest_sha256']==sha(man) and sha(CK)==EXP;mm=json.loads(man.read_text());assert all(sha(Path(k))==v for k,v in mm['sources'].items());torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
 if x.replay:
  assert (H/'PREFIX_GPU_RECEIPT.json').exists() and not(H/'PREFIX_REPLAY.npz').exists() and not(H/'PREFIX_REPLAY_RECEIPT.json').exists();gr=json.loads((H/'PREFIX_GPU_RECEIPT.json').read_text());assert gr['manifest_sha256']==sha(man) and sha(H/'PREFIX_PREDICTIONS.npz')==gr['predictions_sha256'];p,q=one(0);base=np.load(H/'PREFIX_PREDICTIONS.npz');assert np.array_equal(p,base['prefix_fold0']) and np.array_equal(q,base['full_fold0']);np.savez_compressed(H/'PREFIX_REPLAY.npz',prefix=p,full=q);(H/'PREFIX_REPLAY_RECEIPT.json').write_text(json.dumps({'manifest_sha256':sha(man),'predictions_sha256':gr['predictions_sha256'],'exact':True,'seconds':time.monotonic()-st,'sha256':sha(H/'PREFIX_REPLAY.npz')},indent=2));return
 assert not(H/'PREFIX_PREDICTIONS.npz').exists();out={};peaks=[]
 for f in [0,1]:torch.cuda.reset_peak_memory_stats();p,q=one(f);out[f'prefix_fold{f}']=p;out[f'full_fold{f}']=q;peaks.append(int(torch.cuda.max_memory_reserved()));gc.collect();torch.cuda.empty_cache()
 assert max(peaks)<3*1024**3 and time.monotonic()-st<600;np.savez_compressed(H/'PREFIX_PREDICTIONS.npz',**out);old=np.load(AP/'fast129_predictions.npz');assert all(np.array_equal(out[f'full_fold{f}'],old[f'fold{f}']) for f in [0,1]);(H/'PREFIX_GPU_RECEIPT.json').write_text(json.dumps({'manifest_sha256':sha(man),'predictions_sha256':sha(H/'PREFIX_PREDICTIONS.npz'),'full_integrity_exact':True,'seconds':time.monotonic()-st,'peak':max(peaks)},indent=2))
if __name__=='__main__':main()
