import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import json,hashlib,site
from pathlib import Path
site.addsitedir(r'C:\Data\Projects\Money Making\work\poker\.venv\Lib\site-packages')
import numpy as np
from scipy.stats import rankdata,spearmanr
OUT=Path(__file__).resolve().parent;SCREEN=OUT.parent/'screen'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rank2(x):
 a=rankdata(x,method='average');r=np.rint(2*a).astype(np.int64);assert np.allclose(r/2,a);return r
def ap(y,s,keys):
 base=np.argsort(keys,kind='stable');o=base[np.argsort(-np.asarray(s)[base],kind='stable')];yy=np.asarray(y,dtype=float)[o]
 return 0. if yy.sum()==0 else float(np.sum(np.cumsum(yy)/np.arange(1,len(yy)+1)*yy)/yy.sum())
def metrics(y,risk,route,keys):
 pa=ap(y>0,risk,keys);by={str(c):ap(y==c,risk*(route==c),keys) for c in (1,2,3)};bm=float(np.mean(list(by.values())))
 return {'rows':len(y),'positives':int((y>0).sum()),'pair_ap':pa,'behavior_map_fixed_original_route':bm,'by_family_ap':by,'joint_point7_pair_point1_behavior':.7*pa+.1*bm}
def load(prefix):
 ev=[]
 for f in (0,1):
  z=np.load(OUT/f'evaluator_fold{f}{prefix}.npz',allow_pickle=False);ev.append({k:z[k] for k in z.files}|{'fold':np.full(len(z['y']),f)})
 return {k:np.concatenate([v[k] for v in ev]) for k in ev[0]}
def pred(file,prefix):
 z=np.load(file,allow_pickle=False);return np.concatenate([z[f'{prefix}_fold{f}'] for f in (0,1)])
def assess(ev,po,pc):
 ro=1-po[:,0];rc=1-pc[:,0];n=len(ro);blend=(4*rank2(ro)+rank2(rc))/(10*n);route=1+np.argmax(po[:,1:],axis=1)
 base=metrics(ev['y'],ro,route,ev['pair_id']);cand=metrics(ev['y'],blend,route,ev['pair_id'])
 folds={}
 for f in (0,1):
  m=ev['fold']==f;b=metrics(ev['y'][m],ro[m],route[m],ev['pair_id'][m]);c=metrics(ev['y'][m],blend[m],route[m],ev['pair_id'][m]);folds[str(f)]={'baseline':b,'candidate':c,'joint_delta':c['joint_point7_pair_point1_behavior']-b['joint_point7_pair_point1_behavior'],'rank_scope':'pooled ranks, subset after ranking'}
 return ro,rc,blend,route,base,cand,folds
def main():
 assert not (OUT/'results.json').exists(),'refuse overwrite';new=np.load(OUT/'complementary_predictions.npz',allow_pickle=False);old=np.load(SCREEN/'pf35_predictions.npz',allow_pickle=False)
 replay=np.load(OUT/'replay_fold0.npz',allow_pickle=False)['p'];assert np.array_equal(replay,new['known_fold0'])
 ev=load('');po=pred(SCREEN/'pf35_predictions.npz','known');pc=pred(OUT/'complementary_predictions.npz','known');assert len(po)==len(ev['y'])
 ro,rc,blend,route,b,c,folds=assess(ev,po,pc);delta=c['joint_point7_pair_point1_behavior']-b['joint_point7_pair_point1_behavior']
 pev=load('_prefix60');ppo=pred(SCREEN/'pf35_predictions.npz','prefix60');ppc=pred(OUT/'complementary_predictions.npz','prefix60');_,_,_,_,pb,pcand,pfolds=assess(pev,ppo,ppc)
 rng=np.random.default_rng(35935);tables=np.unique(ev['table_id']);d=[]
 for _ in range(1000):
  chosen=rng.choice(tables,len(tables),replace=True);ii=np.concatenate([np.flatnonzero(ev['table_id']==t) for t in chosen]);keys=np.asarray([f'{j:04d}:{ev["pair_id"][i]}' for j,t in enumerate(chosen) for i in np.flatnonzero(ev['table_id']==t)],dtype='U')
  mb=metrics(ev['y'][ii],ro[ii],route[ii],keys);mc=metrics(ev['y'][ii],blend[ii],route[ii],keys);d.append(mc['joint_point7_pair_point1_behavior']-mb['joint_point7_pair_point1_behavior'])
 fg=[folds[str(f)]['joint_delta'] for f in (0,1)];gate=bool(delta>0 and all(x>0 for x in fg))
 out={'scope':'curated known-label complementary raw-view screen; no STEP272 or public-gain claim','numerical_convention':{'rank':'twice-average ranks globally, (4*r2_original+r2_complementary)/(10*N)','ap':'pair_id ascending then stable descending score'},'hashes':{'protocol':sha(OUT/'PROTOCOL.md'),'prep_receipt':sha(OUT/'prep_receipt.json'),'predictions':sha(OUT/'complementary_predictions.npz'),'original_predictions':sha(SCREEN/'pf35_predictions.npz'),'replay':sha(OUT/'replay_fold0.npz')},'known':{'baseline_original_fast':b,'candidate_80original_20complementary':c,'joint_delta':delta,'folds':folds},'prefix60':{'baseline_original_fast':pb,'candidate_80original_20complementary':pcand,'joint_delta':pcand['joint_point7_pair_point1_behavior']-pb['joint_point7_pair_point1_behavior'],'folds':pfolds},'correlation':{'risk_pearson':float(np.corrcoef(ro,rc)[0,1]),'risk_spearman':float(spearmanr(ro,rc).statistic),'p_none_pearson':float(np.corrcoef(po[:,0],pc[:,0])[0,1])},'table_bootstrap_1000_joint_delta':{'mean':float(np.mean(d)),'ci95':[float(np.quantile(d,.025)),float(np.quantile(d,.975))],'p_above_zero':float(np.mean(np.asarray(d)>0))},'gate':{'definition':'pooled and both fold joint deltas > 0; CI reported only','pooled_delta':delta,'fold_deltas':fg,'pass':gate},'fresh_replay_exact':True}
 (OUT/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['gate'],indent=2))
if __name__=='__main__':main()
