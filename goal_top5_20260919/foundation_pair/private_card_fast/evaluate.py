import site;site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent/'screen'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ap(y,s,k):
 y=np.asarray(y,bool);z=np.argsort(k,kind='stable');o=z[np.argsort(-s[z],kind='stable')];q=y[o];return float((np.cumsum(q)/(np.arange(len(q))+1))[q].mean()) if q.any() else 0.
def met(y,s,route,k):return {'pair_ap':ap(y>0,s,k),'behavior_map':float(np.mean([ap(y==c,s*(route==c),k) for c in [1,2,3]]))}
def rank2(x):
 import scipy.stats as st
 return np.rint(2*st.rankdata(x,method='average')).astype(np.int64)
def main():
 assert not(H/'RESULTS.json').exists();m=json.loads((H/'MANIFEST.json').read_text());assert all(sha(Path(k))==v for k,v in m['sources'].items()) and all(sha(H/k)==v for k,v in m['artifacts'].items());r=json.loads((H/'FAST129_RECEIPT.json').read_text());rr=json.loads((H/'REPLAY_RECEIPT.json').read_text());cr=json.loads((H/'CPU_RECEIPT.json').read_text());assert r['manifest_sha256']==rr['manifest_sha256']==cr['manifest_sha256']==sha(H/'MANIFEST.json') and rr['exact'] and rr['predictions_sha256']==r['predictions_sha256'] and sha(H/'fast129_predictions.npz')==r['predictions_sha256'] and sha(H/'fast129_replay_fold0.npz')==rr['replay_sha256'];new=np.load(H/'fast129_predictions.npz',allow_pickle=False);rep=np.load(H/'fast129_replay_fold0.npz',allow_pickle=False);old=np.load(S/'pf35_predictions.npz',allow_pickle=False);assert set(new.files)=={'fold0','fold1'} and set(rep.files)=={'p'};contexts=[np.load(H/f'fast129_context_fold{f}.npz',allow_pickle=False) for f in [0,1]]
 for f in [0,1]:
  q=np.asarray(new[f'fold{f}']);assert q.shape==(len(contexts[f]['pair_id']),4) and np.isfinite(q).all() and (q>=0).all() and (q<=1).all() and np.allclose(q.sum(1),1,atol=1e-5)
 rp=np.asarray(rep['p']);assert rp.shape==new['fold0'].shape and np.isfinite(rp).all() and np.array_equal(new['fold0'],rp)
 ys=[];ks=[];ts=[];fs=[];op=[];npred=[]
 for f in [0,1]:
  assert sha(H/f'fast129_context_fold{f}.npz')==cr['folds'][f]['context_sha256'];e=np.load(S/f'evaluator_fold{f}.npz',allow_pickle=False);assert len(np.unique(e['pair_id']))==len(e['pair_id']) and np.array_equal(e['pair_id'],np.sort(e['pair_id'])) and np.array_equal(e['pair_id'],contexts[f]['pair_id']);ys.append(e['y']);ks.append(e['pair_id']);ts.append(e['table_id']);fs.append(np.full(len(e['y']),f));op.append(old[f'known_fold{f}']);npred.append(new[f'fold{f}'])
 y=np.concatenate(ys);k=np.concatenate(ks);t=np.concatenate(ts);ff=np.concatenate(fs);a=np.concatenate(op).astype(np.float64);b=np.concatenate(npred).astype(np.float64);assert np.isfinite(b).all() and np.allclose(b.sum(1),1,atol=1e-5) and abs(ap([1,0,1],np.array([.5,.5,.2]),np.array(['b','a','c']))-7/12)<1e-12;ro=1.0-a[:,0];rn=1.0-b[:,0];route=1+np.argmax(a[:,1:],1);cand=4*rank2(ro)+rank2(rn);base=met(y,ro,route,k);aug=met(y,cand,route,k)
 def joint(z):return .7*z['pair_ap']+.1*z['behavior_map']
 folds={str(f):joint(met(y[ff==f],cand[ff==f],route[ff==f],k[ff==f]))-joint(met(y[ff==f],ro[ff==f],route[ff==f],k[ff==f])) for f in [0,1]};tabs=sorted(np.unique(t));ixs=[np.flatnonzero(t==q) for q in tabs];rng=np.random.default_rng(35935);boot=[]
 for _ in range(1000):
  ix=np.concatenate([ixs[j] for j in rng.integers(0,len(tabs),len(tabs))]);boot.append(joint(met(y[ix],cand[ix],route[ix],k[ix]))-joint(met(y[ix],ro[ix],route[ix],k[ix])))
 fam={str(c):{'original':ap(y==c,ro,k),'candidate':ap(y==c,cand,k)} for c in [1,2,3]};ll={'original':float(-np.log(np.maximum(a[np.arange(len(y)),y],1e-12)).mean()),'augmented':float(-np.log(np.maximum(b[np.arange(len(y)),y],1e-12)).mean())};res={'baseline':base,'candidate':aug,'delta_joint':joint(aug)-joint(base),'fold_deltas':folds,'bootstrap95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'risk_correlation':float(np.corrcoef(ro,rn)[0,1]),'family_ap':fam,'multiclass_logloss':ll,'gate':joint(aug)>joint(base) and all(v>0 for v in folds.values())};(H/'RESULTS.json').write_text(json.dumps(res,indent=2)+'\n')
if __name__=='__main__':main()
