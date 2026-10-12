import sys,json,hashlib,gc,time,argparse
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[2];S=H.parent/'screen';R=P/'goal_top5_20260919/tabpfn35';sys.path.insert(0,str(R));import runtime35
import numpy as np,torch
from tabpfn import TabPFNClassifier
CK=R/'tabpfn-v3.5-fast-20260909.safetensors';EXP='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(p):
 st=time.monotonic();d=np.load(p,allow_pickle=False);torch.cuda.reset_peak_memory_stats();m=TabPFNClassifier(model_path=str(CK),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=35935,n_preprocessing_jobs=1);m.fit(d['X_train'],d['y_train']);q=np.concatenate([m.predict_proba(d['X_held'][i:i+256]) for i in range(0,len(d['X_held']),256)]);assert np.array_equal(m.classes_,np.arange(4)) and q.shape==(len(d['X_held']),4) and np.isfinite(q).all() and (q>=0).all() and (q<=1).all() and np.allclose(q.sum(1),1,atol=1e-5) and torch.cuda.max_memory_reserved()<3*1024**3 and time.monotonic()-st<300;return q
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');x=ap.parse_args();st=time.monotonic();man=H/'MANIFEST.json';rel=H/'GPU_RELEASE.json';assert rel.exists() and json.loads(rel.read_text())['manifest_sha256']==sha(man) and sha(CK)==EXP and torch.cuda.is_available();mm=json.loads(man.read_text());assert all(sha(Path(k))==v for k,v in mm['sources'].items()) and all(sha(H/k)==v for k,v in mm['artifacts'].items());cr=json.loads((H/'CPU_RECEIPT.json').read_text());assert cr['manifest_sha256']==sha(man)
 for z in cr['folds']:assert sha(H/f'fast129_context_fold{z["outer"]}.npz')==z['context_sha256']
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
 if x.replay:
  assert (H/'FAST129_RECEIPT.json').exists() and not(H/'REPLAY_RECEIPT.json').exists() and not(H/'fast129_replay_fold0.npz').exists();fr=json.loads((H/'FAST129_RECEIPT.json').read_text());assert sha(H/'fast129_predictions.npz')==fr['predictions_sha256'];q=one(H/'fast129_context_fold0.npz');np.savez_compressed(H/'fast129_replay_fold0.npz',p=q);base=np.load(H/'fast129_predictions.npz');assert np.array_equal(q,base['fold0']) and time.monotonic()-st<300;(H/'REPLAY_RECEIPT.json').write_text(json.dumps({'manifest_sha256':sha(man),'predictions_sha256':fr['predictions_sha256'],'replay_sha256':sha(H/'fast129_replay_fold0.npz'),'exact':True,'seconds':time.monotonic()-st,'peak_reserved_bytes':int(torch.cuda.max_memory_reserved())},indent=2)+'\n');return
 assert not(H/'FAST129_RECEIPT.json').exists() and not(H/'fast129_predictions.npz').exists();a={};peaks=[]
 for f in [0,1]:a[f'fold{f}']=one(H/f'fast129_context_fold{f}.npz');peaks.append(int(torch.cuda.max_memory_reserved()));gc.collect();torch.cuda.empty_cache();assert time.monotonic()-st<600
 np.savez_compressed(H/'fast129_predictions.npz',**a);(H/'FAST129_RECEIPT.json').write_text(json.dumps({'manifest_sha256':sha(man),'predictions_sha256':sha(H/'fast129_predictions.npz'),'seconds':time.monotonic()-st,'peak_reserved_bytes':max(peaks)},indent=2)+'\n')
if __name__=='__main__':main()
