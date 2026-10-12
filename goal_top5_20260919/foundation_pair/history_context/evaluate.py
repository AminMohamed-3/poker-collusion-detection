import site,json,hashlib
from pathlib import Path
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import numpy as np
from scipy.stats import rankdata
ROOT=Path(r'C:/Data/Projects/Money Making');OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/history_context';SCR=ROOT/'work/poker/goal_top5_20260919/foundation_pair/screen'
def ap(y,s,k):
 o=np.argsort(k,kind='stable'); y=np.asarray(y)[o];s=np.asarray(s)[o];o=np.argsort(-s,kind='stable');y=y[o];n=y.sum();pos=np.flatnonzero(y)+1;return float((np.cumsum(y)[y.astype(bool)]/pos).sum()/n) if n else 0.
def r2(x):return np.rint(2*rankdata(x,method='average')).astype(np.int64)
def blend(a,b):return (4*r2(a)+r2(b)).astype(np.int64) # denominator irrelevant to ordering
def met(y,s,route,k):
 p=ap(y>0,s,k); bm=np.mean([ap(y==c,np.where(route==c,s,0),k) for c in (1,2,3)]);return {'pair_ap':p,'behavior_map':float(bm),'joint':float(.7*p+.1*bm)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (OUT/'quality_results.json').exists(); assert (OUT/'replay_receipt.json').exists()
 freeze=json.loads((OUT/'freeze.json').read_text());
 for n,h in freeze['files'].items():assert sha(OUT/n)==h,(n,'hash mismatch')
 manifest=json.loads((OUT/'prep_manifest.json').read_text());assert sha(SCR/'pf35_predictions.npz')==manifest['source_hashes']['original_predictions']
 rr=json.loads((OUT/'replay_receipt.json').read_text());assert rr['exact'] is True
 assert (OUT/'predictions.npz').exists() and (OUT/'runtime.json').exists()
 replay=np.load(OUT/'replay.npz')['p'];predcheck=np.load(OUT/'predictions.npz');assert np.array_equal(replay,predcheck['prefix_full_fold0'])
 new=np.load(OUT/'predictions.npz'); old=np.load(SCR/'pf35_predictions.npz'); ef=np.load(OUT/'evaluator.npz'); ep=np.load(OUT/'evaluator_prefix.npz'); result={'scope':'reused curated prospective history-view diagnostic','known':{},'prefix_eligible':{},'prefix_all_secondary':{}}
 for scope,e,select in [('known',ef,np.ones(len(ef['y']),bool)),('prefix_eligible',ep,ep['shared_hands']>=38),('prefix_all_secondary',ep,np.ones(len(ep['y']),bool))]:
  keys=e['pair_id'];y=e['y'];fold=e['fold']; op=np.zeros((len(y),4));apx=np.zeros_like(op);dup=np.zeros_like(op)
  for f in (0,1):
   m=fold==f
   if scope=='known': oo=old[f'known_fold{f}']; suffix='full'
   else: oo=old[f'prefix60_fold{f}']; suffix='prefix'
   assert len(oo)==m.sum();op[m]=oo;apx[m]=new[f'prefix_{suffix}_fold{f}'];dup[m]=new[f'duplicate_{suffix}_fold{f}']
  keys=keys[select];y=y[select];fold=fold[select];op=op[select];apx=apx[select];dup=dup[select];route=1+np.argmax(op[:,1:],1); base=1-op[:,0]; cand=blend(base,1-apx[:,0]);ctrl=blend(base,1-dup[:,0]); cell={'n':len(y),'class_counts':np.bincount(y,minlength=4).tolist(),'pooled':{'original':met(y,base,route,keys),'candidate':met(y,cand,route,keys),'duplicate_control':met(y,ctrl,route,keys)},'folds':{}}
  for f in (0,1):
   m=fold==f;cell['folds'][str(f)]={'original':met(y[m],base[m],route[m],keys[m]),'candidate':met(y[m],cand[m],route[m],keys[m]),'duplicate_control':met(y[m],ctrl[m],route[m],keys[m])}
  result[scope]=cell
  if scope=='known': known_arrays=(base,cand,ctrl,route)
  elif scope=='prefix_eligible': prefix_arrays=(base,cand,ctrl,route,select)
 # fixed 1000 table bootstraps, preserving the table clustering of each surface.
 for scope,e,arrays in [('known',ef,known_arrays),('prefix_eligible',ep,prefix_arrays)]:
  if scope=='prefix_eligible': base,cand,ctrl,route,select=arrays;e={k:e[k][select] for k in e.files}
  else: base,cand,ctrl,route=arrays
  rng=np.random.default_rng(35935);tabs=np.unique(e['table_id']);do=[];dc=[]
  for _ in range(1000):
   take=rng.choice(tabs,len(tabs),replace=True);ix=np.concatenate([np.flatnonzero(e['table_id']==t) for t in take]);mc=met(e['y'][ix],cand[ix],route[ix],e['pair_id'][ix])['joint'];do.append(mc-met(e['y'][ix],base[ix],route[ix],e['pair_id'][ix])['joint']);dc.append(mc-met(e['y'][ix],ctrl[ix],route[ix],e['pair_id'][ix])['joint'])
  result[scope]['bootstrap']={'candidate_minus_original':{'mean':float(np.mean(do)),'ci95':np.quantile(do,[.025,.975]).tolist()},'candidate_minus_duplicate':{'mean':float(np.mean(dc)),'ci95':np.quantile(dc,[.025,.975]).tolist()}}
 jp=result['prefix_eligible']['pooled']; result['gate']={'prefix_vs_original_pooled':jp['candidate']['joint']>jp['original']['joint'],'prefix_vs_original_both_folds':all(result['prefix_eligible']['folds'][str(f)]['candidate']['joint']>result['prefix_eligible']['folds'][str(f)]['original']['joint'] for f in (0,1)),'prefix_vs_duplicate_pooled':jp['candidate']['joint']>jp['duplicate_control']['joint'],'known_noninferior':result['known']['pooled']['candidate']['joint']>=result['known']['pooled']['original']['joint']};result['advance']=all(result['gate'].values());(OUT/'quality_results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
