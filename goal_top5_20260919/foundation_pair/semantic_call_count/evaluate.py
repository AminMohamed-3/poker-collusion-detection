import site,json,hashlib
from pathlib import Path
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import numpy as np,polars as pl
from scipy.stats import rankdata
ROOT=Path(r'C:/Data/Projects/Money Making');OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/semantic_call_count';SCR=ROOT/'work/poker/goal_top5_20260919/foundation_pair/screen'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ap(y,s,k):
 o=np.argsort(k,kind='stable');y=np.asarray(y)[o];s=np.asarray(s)[o];o=np.argsort(-s,kind='stable');y=y[o];n=y.sum();pos=np.flatnonzero(y)+1;return float((np.cumsum(y)[y.astype(bool)]/pos).sum()/n) if n else 0.
def r2(x):return np.rint(2*rankdata(x,method='average')).astype(np.int64)
def blend(a,b):return 4*r2(a)+r2(b)
def met(y,s,route,k):
 p=ap(y>0,s,k);b=float(np.mean([ap(y==c,np.where(route==c,s,0),k) for c in (1,2,3)]));return {'pair_ap':p,'behavior_map':b,'joint':.7*p+.1*b}
def surface(prefix=False):
 old=np.load(SCR/'pf35_predictions.npz');new=np.load(OUT/'predictions.npz');ys=[];ks=[];ts=[];fs=[];op=[];npred=[];sh=[]
 pc=pl.read_parquet(OUT/'corrected_prefix.parquet').select('pair_id','shared_hands');pm=dict(zip(pc['pair_id'],pc['shared_hands']))
 for f in (0,1):
  e=np.load(SCR/f'evaluator_fold{f}_prefix60.npz' if prefix else SCR/f'evaluator_fold{f}.npz');key=e['pair_id'];ys.append(e['y']);ks.append(key);ts.append(e['table_id']);fs.append(np.full(len(key),f));op.append(old[f'prefix60_fold{f}' if prefix else f'known_fold{f}']);npred.append(new[f'prefix_fold{f}' if prefix else f'full_fold{f}']);sh.append(np.asarray([pm[x] for x in key]) if prefix else np.full(len(key),999))
 return tuple(map(np.concatenate,(ys,ks,ts,fs,op,npred,sh)))
def main():
 assert not (OUT/'quality_results.json').exists();freeze=json.loads((OUT/'freeze.json').read_text());[(_ for _ in ()).throw(AssertionError((n,'hash'))) for n,h in freeze['files'].items() if sha(OUT/n)!=h]
 ext={'pf35_predictions':SCR/'pf35_predictions.npz',**{f'evaluator_fold{f}':SCR/f'evaluator_fold{f}.npz' for f in (0,1)},**{f'evaluator_fold{f}_prefix60':SCR/f'evaluator_fold{f}_prefix60.npz' for f in (0,1)}}
 for n,pth in ext.items():assert sha(pth)==freeze['external'][n],(n,'external hash')
 rr=json.loads((OUT/'replay_receipt.json').read_text());assert rr['exact'];p=np.load(OUT/'predictions.npz');r=np.load(OUT/'replay.npz');assert np.array_equal(r['full'],p['full_fold0']) and np.array_equal(r['prefix'],p['prefix_fold0'])
 result={'scope':'reused curated semantic-call-count alternative; original literal feature protected','surfaces':{}}
 for name,prefix,elig in [('known',False,False),('prefix_eligible',True,True),('prefix_all_secondary',True,False)]:
  y,k,t,f,o,n,sh=surface(prefix);sel=(sh>=38) if elig else np.ones(len(y),bool);y,k,t,f,o,n=y[sel],k[sel],t[sel],f[sel],o[sel],n[sel];route=1+np.argmax(o[:,1:],1);base=1-o[:,0];cand=blend(base,1-n[:,0]);cell={'n':len(y),'class_counts':np.bincount(y,minlength=4).tolist(),'pooled':{'original':met(y,base,route,k),'candidate':met(y,cand,route,k)},'folds':{}}
  for z in (0,1):m=f==z;cell['folds'][str(z)]={'original':met(y[m],base[m],route[m],k[m]),'candidate':met(y[m],cand[m],route[m],k[m])}
  rng=np.random.default_rng(35935);tabs=np.unique(t);delta=[]
  for _ in range(1000):
   take=rng.choice(tabs,len(tabs),replace=True);ix=np.concatenate([np.flatnonzero(t==x) for x in take]);delta.append(met(y[ix],cand[ix],route[ix],k[ix])['joint']-met(y[ix],base[ix],route[ix],k[ix])['joint'])
  cell['table_bootstrap_candidate_minus_original']={'mean':float(np.mean(delta)),'ci95':np.quantile(delta,[.025,.975]).tolist()};result['surfaces'][name]=cell
 kn=result['surfaces']['known'];pr=result['surfaces']['prefix_eligible'];result['gate']={'known_pooled_positive':kn['pooled']['candidate']['joint']>kn['pooled']['original']['joint'],'known_both_folds_positive':all(kn['folds'][str(z)]['candidate']['joint']>kn['folds'][str(z)]['original']['joint'] for z in (0,1)),'eligible_prefix_nonnegative':pr['pooled']['candidate']['joint']>=pr['pooled']['original']['joint']};result['advance']=all(result['gate'].values());(OUT/'quality_results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
