import os,site,json,hashlib
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import numpy as np
from scipy.stats import rankdata,spearmanr
OUT=Path(__file__).resolve().parent;SCR=OUT.parent/'screen'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ap(y,s,k):
 o=np.argsort(k,kind='stable');o=o[np.argsort(-np.asarray(s)[o],kind='stable')];y=np.asarray(y,dtype=float)[o];return 0. if not y.sum() else float(np.sum(np.cumsum(y)/np.arange(1,len(y)+1)*y)/y.sum())
def r2(x):a=rankdata(x,method='average');return np.rint(2*a).astype(np.int64)
def met(y,s,r,k):p=ap(y>0,s,k);b=float(np.mean([ap(y==c,s*(r==c),k) for c in (1,2,3)]));return {'pair_ap':p,'behavior_map':b,'joint':.7*p+.1*b}
def evals():
 a=[]
 for f in (0,1):
  z=np.load(OUT/f'evaluator_fold{f}.npz');a.append({k:z[k] for k in z.files}|{'fold':np.full(len(z['y']),f)})
 return {k:np.concatenate([z[k] for z in a]) for k in a[0]}
def score(e,old,binary,sel):
 y=e['y'][sel];k=e['pair_id'][sel];fold=e['fold'][sel];old=old[sel];binary=binary[sel];risk=1-old[:,0];route=1+np.argmax(old[:,1:],1);cand=4*r2(risk)+r2(binary);out={'pooled':{'original':met(y,risk,route,k),'candidate':met(y,cand,route,k),'standalone_binary_old_route':met(y,binary,route,k)},'folds':{}}
 for f in (0,1):
  m=fold==f;out['folds'][str(f)]={'original':met(y[m],risk[m],route[m],k[m]),'candidate':met(y[m],cand[m],route[m],k[m])}
 return out,(y,k,fold,e['table_id'][sel],risk,cand,route)
def boot(v):
 y,k,fold,t,r,c,route=v;rng=np.random.default_rng(35935);tabs=np.unique(t);d=[]
 for _ in range(1000):
  take=rng.choice(tabs,len(tabs),replace=True);ix=np.concatenate([np.flatnonzero(t==z) for z in take]);d.append(met(y[ix],c[ix],route[ix],k[ix])['joint']-met(y[ix],r[ix],route[ix],k[ix])['joint'])
 return {'mean':float(np.mean(d)),'ci95':np.quantile(d,[.025,.975]).tolist()}
def main():
 assert not (OUT/'results.json').exists();m=json.loads((OUT/'prep_manifest.json').read_text());release=json.loads((OUT/'GPU_RELEASE.json').read_text());assert release['prep_manifest_sha256']==sha(OUT/'prep_manifest.json')
 for n,h in m['artifact_hashes'].items():assert sha(OUT/n)==h,(n,'hash mismatch')
 assert sha(SCR/'pf35_predictions.npz')==m['original_predictions_sha256'];p=np.load(OUT/'predictions.npz');re=np.load(OUT/'replay.npz');assert np.array_equal(p['known_fold0'],re['known']) and np.array_equal(p['prefix_fold0'],re['prefix']);old=np.load(SCR/'pf35_predictions.npz');e=evals();op=np.concatenate([old[f'known_fold{f}'] for f in (0,1)]);bp=np.concatenate([p[f'known_fold{f}'] for f in (0,1)]);known,kv=score(e,op,bp,np.ones(len(e['y']),bool));ep=np.load(OUT/'evaluator_prefix.npz');epo=np.concatenate([old[f'prefix60_fold{f}'] for f in (0,1)]);epb=np.concatenate([p[f'prefix_fold{f}'] for f in (0,1)]);eligible=ep['shared_hands']>=38;assert eligible.sum()==664;pe,pv=score(ep,epo,epb,eligible);pa,_=score(ep,epo,epb,np.ones(len(ep['y']),bool));kd=known['pooled']['candidate']['joint']-known['pooled']['original']['joint'];fd=[known['folds'][str(f)]['candidate']['joint']-known['folds'][str(f)]['original']['joint'] for f in (0,1)];pd=pe['pooled']['candidate']['joint']-pe['pooled']['original']['joint'];out={'known':known,'prefix_eligible_664':pe,'prefix_all_813_secondary':pa,'bootstrap':{'known':boot(kv),'prefix_eligible':boot(pv)},'correlation':float(spearmanr(1-op[:,0],bp).statistic),'gate':{'known_pooled_delta':kd,'known_fold_deltas':fd,'eligible_prefix_delta':pd,'pass':bool(kd>0 and all(x>0 for x in fd) and pd>=0)},'hashes':{'protocol':sha(OUT/'PROTOCOL.md'),'prep':sha(OUT/'prep_manifest.json'),'predictions':sha(OUT/'predictions.npz')},'replay_exact':True};(OUT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
