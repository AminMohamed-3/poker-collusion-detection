import site,json,hashlib
from pathlib import Path
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import numpy as np
from scipy.stats import rankdata
ROOT=Path(r'C:/Data/Projects/Money Making');OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/tabldm_enhanced_screen';BASE=ROOT/'work/poker/goal_top5_20260919/foundation_pair/tabldm_screen';SCR=ROOT/'work/poker/goal_top5_20260919/foundation_pair/screen'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank2(x):
 r=np.rint(2*rankdata(x,method='average')).astype(np.int64);assert np.allclose(r/2,rankdata(x,method='average'));return r
def order(s,k):
 z=np.argsort(k,kind='stable');return z[np.argsort(-np.asarray(s)[z],kind='stable')]
def rap(y,o,m):
 yy=np.repeat(np.asarray(y,dtype=float)[o],np.asarray(m,dtype=np.int64)[o]);return 0. if not yy.sum() else float(np.sum((np.cumsum(yy)/np.arange(1,len(yy)+1))*yy)/yy.sum())
def ap(y,s,k):
 o=order(s,k);yy=np.asarray(y,dtype=float)[o];return 0. if not yy.sum() else float(np.sum((np.cumsum(yy)/np.arange(1,len(yy)+1))*yy)/yy.sum())
def met(y,s,route,k):
 p=ap(y>0,s,k);b=float(np.mean([ap(y==c,np.where(route==c,s,0),k) for c in (1,2,3)]));return {'pair_ap':p,'behavior_map':b,'joint':.7*p+.1*b}
def prep(y,s,route,k):return {'p':order(s,k),**{str(c):order(np.where(route==c,s,0),k) for c in (1,2,3)}}
def rmet(y,m,o):
 p=rap(y>0,o['p'],m);b=float(np.mean([rap(y==c,o[str(c)],m) for c in (1,2,3)]));return .7*p+.1*b
def parity():
 y=np.array([1,0,2,0,3,0]);s=np.array([.5,.5,.2,.2,.8,.2]);k=np.array(['P3','P1','P6','P4','P2','P5']);t=np.array([0,0,1,1,2,2]);route=np.array([1,2,2,3,3,1]);chosen=np.array([2,0,2,1,0]);m=np.bincount(chosen,minlength=3)[t];ix=np.concatenate([np.flatnonzero(t==z) for z in chosen]);assert met(y[ix],s[ix],route[ix],k[ix])['joint']==rmet(y,m,prep(y,s,route,k))
def load(prefix):
 old=np.load(SCR/'pf35_predictions.npz');base_tab=np.load(BASE/'predictions.npz');new=np.load(OUT/'predictions.npz');ys=[];ks=[];ts=[];fs=[];op=[];bpd=[];npd=[];sh=[]
 shared=np.load(OUT/'prefix_shared_hands.npz') if prefix else None
 for f in (0,1):
  e=np.load(SCR/f'evaluator_fold{f}_prefix60.npz' if prefix else SCR/f'evaluator_fold{f}.npz');ctx=np.load(OUT/f'context_fold{f}_prefix60.npz' if prefix else OUT/f'context_fold{f}.npz');assert np.array_equal(ctx['pair_id'],e['pair_id']) and len(np.unique(e['pair_id']))==len(e['pair_id']) and np.array_equal(e['pair_id'],np.sort(e['pair_id']));ys.append(e['y']);ks.append(e['pair_id']);ts.append(e['table_id']);fs.append(np.full(len(e['y']),f));oo=np.asarray(old[f'prefix60_fold{f}' if prefix else f'known_fold{f}'],dtype=np.float64);nn=np.asarray(new[f'prefix_fold{f}' if prefix else f'full_fold{f}'],dtype=np.float64);assert oo.shape==nn.shape==(len(e['y']),4) and np.isfinite(nn).all() and (nn>=0).all() and (nn<=1).all() and np.allclose(nn.sum(1),1,atol=2e-6);op.append(oo);bb=np.asarray(base_tab[f'prefix_fold{f}' if prefix else f'full_fold{f}'],dtype=np.float64);assert bb.shape==nn.shape and np.isfinite(bb).all() and np.allclose(bb.sum(1),1,atol=2e-6);bpd.append(bb);npd.append(nn);assert not prefix or np.array_equal(shared[f'pair_id_fold{f}'],e['pair_id']);sh.append(shared[f'shared_hands_fold{f}'] if prefix else np.full(len(e['y']),999))
 return tuple(map(np.concatenate,(ys,ks,ts,fs,op,bpd,npd,sh)))
def main():
 assert not (OUT/'quality_results.json').exists();assert (OUT/'RUNNER_FROZEN').exists();fr=json.loads((OUT/'freeze.json').read_text());
 for n,h in fr['files'].items():assert sha(OUT/n)==h,(n,'hash')
 ext={'pf35_predictions':SCR/'pf35_predictions.npz','base_tabldm_predictions':BASE/'predictions.npz','base_quality_results':BASE/'quality_results.json',**{f'evaluator_fold{f}':SCR/f'evaluator_fold{f}.npz' for f in (0,1)},**{f'evaluator_fold{f}_prefix60':SCR/f'evaluator_fold{f}_prefix60.npz' for f in (0,1)}}
 for n,p in ext.items():assert sha(p)==fr['external'][n],(n,'external hash')
 rr=json.loads((OUT/'replay_receipt.json').read_text());rt=json.loads((OUT/'runtime.json').read_text());sr=json.loads((OUT/'supervisor_receipt.json').read_text());assert sr['status']=='PASS' and sr['wall_seconds']<1800;assert rt['enhance_candidates'] is True and rt['n_estimators']==4 and rt['batch_size']==1 and rt['seed']==35935 and rt['n_jobs']==2 and rt['query_chunk']==256;assert len(rt['fits'])==2 and all(x['candidate_counts']=={'main':4,'adaptive_plus':16,'gaussian_rank':16} for x in rt['fits']);assert rr['candidate_counts']=={'main':4,'adaptive_plus':16,'gaussian_rank':16};assert rr['exact_full'] and rr['exact_prefix'] and rr['predictions_sha256']==rt['predictions_sha256']==sha(OUT/'predictions.npz') and rr['replay_sha256']==sha(OUT/'replay.npz');p=np.load(OUT/'predictions.npz');assert set(p.files)=={'full_fold0','prefix_fold0','full_fold1','prefix_fold1'};r=np.load(OUT/'replay.npz');assert np.array_equal(r['full'],p['full_fold0']) and np.array_equal(r['prefix'],p['prefix_fold0']);parity();res={'scope':'prospective enhanced-default TabLDM versus base-TabLDM matched screen on reused curated surfaces','surfaces':{}}
 for name,pfx,eligible in [('known',False,False),('prefix_eligible',True,True),('prefix_all_secondary',True,False)]:
  y,k,t,f,o,b,n,sh=load(pfx);sel=(sh>=38) if eligible else np.ones(len(y),bool);y,k,t,f,o,b,n=y[sel],k[sel],t[sel],f[sel],o[sel],b[sel],n[sel];route=1+np.argmax(o[:,1:],1);fast=1.0-o[:,0];base_tab_risk=1.0-b[:,0];enh_risk=1.0-n[:,0];base_cand=rank2(fast)+rank2(base_tab_risk);enh_cand=rank2(fast)+rank2(enh_risk);cell={'n':len(y),'pooled':{'fast_plus_base_tabldm':met(y,base_cand,route,k),'fast_plus_enhanced_tabldm':met(y,enh_cand,route,k),'enhanced_tabldm_standalone_old_route':met(y,enh_risk,route,k)},'folds':{}}
  for z in (0,1):q=f==z;cell['folds'][str(z)]={'fast_plus_base_tabldm':met(y[q],base_cand[q],route[q],k[q]),'fast_plus_enhanced_tabldm':met(y[q],enh_cand[q],route[q],k[q])}
  if name!='prefix_all_secondary':
   rng=np.random.default_rng(35935);tabs,inv=np.unique(t,return_inverse=True);ob,oc=prep(y,base_cand,route,k),prep(y,enh_cand,route,k);ds=[]
   for _ in range(1000):cnt=np.bincount(np.searchsorted(tabs,rng.choice(tabs,len(tabs),replace=True)),minlength=len(tabs));m=cnt[inv];ds.append(rmet(y,m,oc)-rmet(y,m,ob))
   cell['table_bootstrap']={'mean':float(np.mean(ds)),'ci95':np.quantile(ds,[.025,.975]).tolist()}
  res['surfaces'][name]=cell
 saved=json.loads((BASE/'quality_results.json').read_text());
 for surface in ('known','prefix_eligible','prefix_all_secondary'):
  assert res['surfaces'][surface]['pooled']['fast_plus_base_tabldm']==saved['surfaces'][surface]['pooled']['candidate_equal_rank'],(surface,'base parity')
 kn=res['surfaces']['known'];pr=res['surfaces']['prefix_eligible'];res['base_screen_exact_parity']=True;res['gate']={'known_pooled_positive':kn['pooled']['fast_plus_enhanced_tabldm']['joint']>kn['pooled']['fast_plus_base_tabldm']['joint'],'known_both_folds_positive':all(kn['folds'][str(z)]['fast_plus_enhanced_tabldm']['joint']>kn['folds'][str(z)]['fast_plus_base_tabldm']['joint'] for z in (0,1)),'eligible_prefix_nonnegative':pr['pooled']['fast_plus_enhanced_tabldm']['joint']>=pr['pooled']['fast_plus_base_tabldm']['joint']};res['advance']=all(res['gate'].values());(OUT/'quality_results.json').write_text(json.dumps(res,indent=2)+'\n')
if __name__=='__main__':main()


