import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import site,json,hashlib
from pathlib import Path
site.addsitedir(r'C:\Data\Projects\Money Making\work\poker\.venv\Lib\site-packages')
import numpy as np
from scipy.stats import rankdata,spearmanr
OUT=Path(__file__).resolve().parent;SCREEN=OUT.parent/'screen'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def r2(x):
 a=rankdata(x,method='average');z=np.rint(2*a).astype(np.int64);assert np.allclose(z/2,a);return z
def ap(y,s,k):
 b=np.argsort(k,kind='stable');o=b[np.argsort(-np.asarray(s)[b],kind='stable')];q=np.asarray(y,dtype=float)[o]
 return 0. if q.sum()==0 else float(np.sum(np.cumsum(q)/np.arange(1,len(q)+1)*q)/q.sum())
def met(y,s,route,k):
 pa=ap(y>0,s,k);by={str(c):ap(y==c,s*(route==c),k) for c in (1,2,3)};bm=float(np.mean(list(by.values())));return {'rows':len(y),'positives':int((y>0).sum()),'pair_ap':pa,'behavior_map':bm,'by_family_ap':by,'joint':.7*pa+.1*bm}
def load_ev(suf):
 z=[]
 for f in (0,1):
  a=np.load(SCREEN/f'evaluator_fold{f}{suf}.npz',allow_pickle=False);z.append({k:a[k] for k in a.files}|{'fold':np.full(len(a['y']),f)})
 return {k:np.concatenate([a[k] for a in z]) for k in z[0]}
def probs(path,prefix):
 a=np.load(path,allow_pickle=False);return np.concatenate([a[f'{prefix}_fold{f}'] for f in (0,1)])
def assess(ev,old,bag):
 ro=1-old[:,0];rb=1-bag[:,0];num=4*r2(ro)+r2(rb);oldroute=1+np.argmax(old[:,1:],axis=1);bagroute=1+np.argmax(bag[:,1:],axis=1)
 base=met(ev['y'],ro,oldroute,ev['pair_id']);cand=met(ev['y'],num,oldroute,ev['pair_id']);stand=met(ev['y'],rb,bagroute,ev['pair_id']);folds={}
 for f in (0,1):
  m=ev['fold']==f;b=met(ev['y'][m],ro[m],oldroute[m],ev['pair_id'][m]);c=met(ev['y'][m],num[m],oldroute[m],ev['pair_id'][m]);folds[str(f)]={'baseline':b,'candidate':c,'joint_delta':c['joint']-b['joint'],'rank_scope':'pooled ranks, subset after ranking'}
 return ro,rb,num,oldroute,base,cand,stand,folds
def main():
 assert not (OUT/'results.json').exists();p=np.load(OUT/'balanced_predictions.npz',allow_pickle=False);re=np.load(OUT/'replay_fold0_bag0.npz',allow_pickle=False);assert np.array_equal(p['known_fold0_bag0'],re['known']) and np.array_equal(p['prefix60_fold0_bag0'],re['prefix60'])
 ev=load_ev('');old=probs(SCREEN/'pf35_predictions.npz','known');bag=probs(OUT/'balanced_predictions.npz','known');ro,rb,num,route,base,cand,stand,folds=assess(ev,old,bag);delta=cand['joint']-base['joint']
 pe=load_ev('_prefix60');po=probs(SCREEN/'pf35_predictions.npz','prefix60');pb=probs(OUT/'balanced_predictions.npz','prefix60');_,_,_,_,pbase,pcand,pstand,pfolds=assess(pe,po,pb);pdelta=pcand['joint']-pbase['joint']
 rng=np.random.default_rng(35935);tables=np.unique(ev['table_id']);ds=[]
 for z in range(1000):
  chosen=rng.choice(tables,len(tables),replace=True);chunks=[np.flatnonzero(ev['table_id']==t) for t in chosen];ii=np.concatenate(chunks);keys=np.concatenate([np.char.add(f'{j:04d}:',ev['pair_id'][q]) for j,q in enumerate(chunks)]);b=met(ev['y'][ii],ro[ii],route[ii],keys);c=met(ev['y'][ii],num[ii],route[ii],keys);ds.append(c['joint']-b['joint'])
 fg=[folds[str(f)]['joint_delta'] for f in (0,1)];gate=bool(delta>0 and all(x>0 for x in fg) and pdelta>=0)
 out={'scope':'single frozen table-balanced normal-context bag screen','convention':{'fusion':'exact integer numerator 4*r2_original+r2_bagged over pooled rows','ap':'pair_id ascending then stable descending score','primary_route':'immutable original Fast route'},'hashes':{'protocol':sha(OUT/'PROTOCOL.md'),'prep':sha(OUT/'prep_receipt.json'),'predictions':sha(OUT/'balanced_predictions.npz'),'original_predictions':sha(SCREEN/'pf35_predictions.npz'),'replay':sha(OUT/'replay_fold0_bag0.npz')},'known':{'original':base,'balanced_bag_standalone_own_route':stand,'primary':cand,'joint_delta':delta,'folds':folds},'prefix60':{'original':pbase,'balanced_bag_standalone_own_route':pstand,'primary':pcand,'joint_delta':pdelta,'folds':pfolds},'correlation':{'risk_pearson':float(np.corrcoef(ro,rb)[0,1]),'risk_spearman':float(spearmanr(ro,rb).statistic)},'table_bootstrap_1000_joint_delta':{'mean':float(np.mean(ds)),'ci95':[float(np.quantile(ds,.025)),float(np.quantile(ds,.975))],'p_above_zero':float(np.mean(np.asarray(ds)>0))},'gate':{'definition':'known pooled and both-fold joint deltas >0, prefix pooled joint delta >=0; CI descriptive','known_pooled_delta':delta,'known_fold_deltas':fg,'prefix_pooled_delta':pdelta,'pass':gate},'fresh_replay_exact':True}
 (OUT/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['gate'],indent=2))
if __name__=='__main__':main()
