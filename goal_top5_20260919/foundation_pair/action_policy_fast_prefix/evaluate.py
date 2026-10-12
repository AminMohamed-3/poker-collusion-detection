import site;site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
H=Path(__file__).resolve().parent;S=H.parent/'screen';T=H.parent/'tabldm_screen'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ap(y,s,k):
 z=np.argsort(k,kind='stable');o=z[np.argsort(-np.asarray(s)[z],kind='stable')];q=np.asarray(y,bool)[o];return float((np.cumsum(q)/(np.arange(len(q))+1))[q].mean()) if q.any() else 0.
def rank2(x):return np.rint(2*rankdata(x,method='average')).astype(np.int64)
def met(y,s,r,k):
 p=ap(y>0,s,k);b=float(np.mean([ap(y==c,s*(r==c),k) for c in [1,2,3]]));return {'pair_ap':p,'behavior_map':b,'joint':.7*p+.1*b}
def main():
 assert not(H/'PREFIX_RESULTS.json').exists();m=json.loads((H/'GPU_MANIFEST.json').read_text());assert all(sha(Path(k))==v for k,v in m['sources'].items());gr=json.loads((H/'PREFIX_GPU_RECEIPT.json').read_text());rr=json.loads((H/'PREFIX_REPLAY_RECEIPT.json').read_text());assert gr['manifest_sha256']==rr['manifest_sha256']==sha(H/'GPU_MANIFEST.json') and gr['full_integrity_exact'] and rr['exact'] and sha(H/'PREFIX_PREDICTIONS.npz')==gr['predictions_sha256']==rr['predictions_sha256'] and sha(H/'PREFIX_REPLAY.npz')==rr['sha256'];new=np.load(H/'PREFIX_PREDICTIONS.npz');rep=np.load(H/'PREFIX_REPLAY.npz');assert np.array_equal(rep['prefix'],new['prefix_fold0']) and np.array_equal(rep['full'],new['full_fold0']);old=np.load(S/'pf35_predictions.npz');sh=np.load(T/'prefix_shared_hands.npz');ys=[];ks=[];ts=[];fs=[];aa=[];bb=[];hh=[]
 for f in [0,1]:
  e=np.load(S/f'evaluator_fold{f}_prefix60.npz');c=np.load(H/f'context_scored_prefix_fold{f}.npz');assert np.array_equal(e['pair_id'],c['pair_id']) and np.array_equal(e['pair_id'],sh[f'pair_id_fold{f}']);a=np.asarray(old[f'prefix60_fold{f}'],dtype=np.float64);b=np.asarray(new[f'prefix_fold{f}'],dtype=np.float64);assert a.shape==b.shape==(len(e['y']),4) and np.isfinite(b).all() and np.allclose(b.sum(1),1,atol=1e-5);ys.append(e['y']);ks.append(e['pair_id']);ts.append(e['table_id']);fs.append(np.full(len(e['y']),f));aa.append(a);bb.append(b);hh.append(sh[f'shared_hands_fold{f}'])
 y,k,t,f,a,b,h=map(np.concatenate,(ys,ks,ts,fs,aa,bb,hh));route=1+np.argmax(a[:,1:],1);ro=1-a[:,0];rn=1-b[:,0];cand=4*rank2(ro)+rank2(rn);res={}
 for name,sel in [('eligible664',h>=38),('all813',np.ones(len(y),bool))]:
  base=met(y[sel],ro[sel],route[sel],k[sel]);ca=met(y[sel],cand[sel],route[sel],k[sel]);folds={str(z):met(y[sel&(f==z)],cand[sel&(f==z)],route[sel&(f==z)],k[sel&(f==z)])['joint']-met(y[sel&(f==z)],ro[sel&(f==z)],route[sel&(f==z)],k[sel&(f==z)])['joint'] for z in [0,1]};cell={'n':int(sel.sum()),'baseline':base,'candidate':ca,'delta_joint':ca['joint']-base['joint'],'fold_deltas':folds}
  if name=='eligible664':
   yy,kk,tt,rr,oo,cc=y[sel],k[sel],t[sel],route[sel],ro[sel],cand[sel];tabs=sorted(np.unique(tt));ixs=[np.flatnonzero(tt==x) for x in tabs];rng=np.random.default_rng(35935);ds=[]
   for _ in range(1000):
    ix=np.concatenate([ixs[j] for j in rng.integers(0,len(tabs),len(tabs))]);ds.append(met(yy[ix],cc[ix],rr[ix],kk[ix])['joint']-met(yy[ix],oo[ix],rr[ix],kk[ix])['joint'])
   cell['bootstrap95']=[float(np.quantile(ds,.025)),float(np.quantile(ds,.975))]
  res[name]=cell
 saved=json.loads((T/'quality_results.json').read_text())['surfaces'];assert res['eligible664']['baseline']==saved['prefix_eligible']['pooled']['original'] and res['all813']['baseline']==saved['prefix_all_secondary']['pooled']['original'];res['rank_population']='global doubled ranks over all 813, then select eligible664; prospectively differs from older within-664 TabLDM candidate reranking';res['gate']=res['eligible664']['delta_joint']>0 and all(v>0 for v in res['eligible664']['fold_deltas'].values());(H/'PREFIX_RESULTS.json').write_text(json.dumps(res,indent=2)+'\n')
if __name__=='__main__':main()
