import hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];GP=HERE.parent/'global_pu';TG=HERE.parent/'tabldm_global_pu';DEST=HERE/'quality_results.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def r2(x):r=np.rint(2*rankdata(x,method='average')).astype(np.int64);assert np.allclose(r/2,rankdata(x,method='average'));return r
def ap(y,s,k):
 b=np.argsort(k,kind='stable');o=b[np.argsort(-np.asarray(s)[b],kind='stable')];z=np.asarray(y,float)[o];return 0. if not z.sum() else float(np.sum(np.cumsum(z)/np.arange(1,len(z)+1)*z)/z.sum())
def score(y,risk,route,keys):
 p=ap(y>0,risk,keys);by={str(c):ap(y==c,risk*(route==c),keys) for c in (1,2,3)};b=float(np.mean(list(by.values())));return {'pair_ap':p,'behavior_map':b,'joint':.7*p+.1*b,'by_family_ap':by}
def order(s,k):b=np.argsort(k,kind='stable');return b[np.argsort(-np.asarray(s)[b],kind='stable')]
def rap(y,m,o):z=np.repeat(np.asarray(y,float)[o],np.asarray(m,np.int64)[o]);return 0. if not z.sum() else float(np.sum(np.cumsum(z)/np.arange(1,len(z)+1)*z)/z.sum())
def rscore(y,m,orders):
 p=rap(y>0,m,orders['p']);b=float(np.mean([rap(y==c,m,orders[str(c)]) for c in (1,2,3)]));return {'pair_ap':p,'behavior_map':b,'joint':.7*p+.1*b}
def main():
 assert os.environ.get('ROOT_QUALITY')=='tabldm-equal50-pu-v1','root quality release absent';assert not DEST.exists();man=json.loads((HERE/'MANIFEST.json').read_text());assert all(sha(HERE/n)==h for n,h in man['code_files'].items());assert all(sha(ROOT/n)==h for n,h in man['inputs'].items());pred=np.load(TG/'predictions.npz',allow_pickle=False);assert json.loads((TG/'runtime.json').read_text())['predictions_sha256']==sha(TG/'predictions.npz');rr=json.loads((TG/'replay_receipt.json').read_text());assert rr['replay_sha256']==sha(TG/'replay.npz');rp=np.load(TG/'replay.npz',allow_pickle=False);assert np.array_equal(rp['pair_id'],pred['pair_fold1'][:1024]) and np.array_equal(rp['table_id'],pred['table_fold1'][:1024]) and np.array_equal(rp['prob'],pred['prob_fold1'][:1024])
 rows=[]
 for f in (1,2):
  ev=np.load(GP/f'evaluator_fold{f}.npz',allow_pickle=False);assert np.array_equal(pred[f'pair_fold{f}'],ev['pair_id']) and np.array_equal(pred[f'table_fold{f}'],ev['table_id']);rows.append({**{k:ev[k] for k in ('pair_id','table_id','known','label','family')},'tab':pred[f'prob_fold{f}'],'fold':np.full(len(ev['pair_id']),f)})
 x={k:np.concatenate([z[k] for z in rows]) for k in rows[0]};_,ix=np.unique(x['pair_id'],return_index=True);ix=np.sort(ix);x={k:v[ix] for k,v in x.items()};assert len(x['pair_id'])==50761 and int(x['known'].sum())==664
 base=np.load(GP/'final2_baseline.npz',allow_pickle=False);bi={v:i for i,v in enumerate(base['pair_id'])};q=np.array([bi[v] for v in x['pair_id']]);assert np.array_equal(base['table_id'][q],x['table_id']);b=base['risk_v11_final2'][q]
 fast=[]
 for f in (1,2):
  ps=[]
  for p in sorted((GP/'returned/outputs').glob(f'pred_fold{f}_*.npz')):ps.append(np.load(p,allow_pickle=False))
  fk=np.concatenate([z['pair_id'] for z in ps]);ft=np.concatenate([z['table_id'] for z in ps]);assert np.array_equal(fk,pred[f'pair_fold{f}']) and np.array_equal(ft,pred[f'table_fold{f}']);fast.append(np.concatenate([z['probabilities'] for z in ps]))
 allfast=np.concatenate(fast)[ix].astype(np.float64,copy=False);tab=x['tab'].astype(np.float64,copy=False);assert np.isfinite(tab).all() and (tab>=0).all() and (tab<=1).all() and np.allclose(tab.sum(1),1,atol=2e-6);assert np.isfinite(allfast).all() and (allfast>=0).all() and (allfast<=1).all() and np.allclose(allfast.sum(1),1,atol=2e-6)
 routes=np.load(GP/'predicted_routes.npz',allow_pickle=False);ri={v:i for i,v in enumerate(routes['pair_id'])};rq=np.array([ri[v] for v in x['pair_id']]);route=routes['predicted_class'][rq];assert np.array_equal(routes['table_id'][rq],x['table_id'])
 rf=1-allfast[:,0];rt=1-tab[:,0];rb=r2(b);rfast=r2(rf);equal=rfast+r2(rt);baseline=4*rb+r2(equal);candidate=rb+r2(equal);out={'scope':'adaptive historical v11-prefix + STEP163-FINAL2 surrogate after public TabLDM gain; PU unknowns are not negatives; not exact incumbent','baseline':'fixed 80/20 historical-base/equal-Fast-TabLDM source composition from tabldm_global_pu','candidate':'single fixed 50/50 historical-base/equal-Fast-TabLDM source composition','route_source':'unchanged byte-exact historical prefix60 predicted_class','weights_tested':[.5],'prior_fast_only_50_status':'closed failure; not reinterpreted','pu':{},'curated_664':{},'folds':{},'bootstrap':{}}
 y=x['family'];keys=x['pair_id']
 for name,m in [('pu',np.ones(len(keys),bool)),('curated_664',x['known'])]:out[name]={'rows':int(m.sum()),'baseline80_equalFT':score(y[m],baseline[m],route[m],keys[m]),'candidate50_equalFT':score(y[m],candidate[m],route[m],keys[m])}
 prior=json.loads((TG/'quality_results.json').read_text())
 for surface in ('pu','curated_664'):
  for metric in ('pair_ap','behavior_map','joint'):
   assert out[surface]['baseline80_equalFT'][metric]==prior[surface]['candidate80_equalFT'][metric]
 out['baseline80_saved_parity']={'pu':True,'curated_664':True}
 for f in (1,2):
  m=x['fold']==f
  out['folds'][str(f)]={'rows':int(m.sum()),'rank_scope':'global before subset',
   'baseline80_equalFT':score(y[m],baseline[m],route[m],keys[m]),
   'candidate50_equalFT':score(y[m],candidate[m],route[m],keys[m])}
  for metric in ('pair_ap','behavior_map','joint'):
   assert out['folds'][str(f)]['baseline80_equalFT'][metric]==prior['folds'][str(f)]['candidate80_equalFT'][metric]
 tables,inv=np.unique(x['table_id'],return_inverse=True);orders={}
 for n,s in [('a',baseline),('b',candidate)]:
  orders[n]={'p':order(s,keys),**{str(c):order(s*(route==c),keys) for c in (1,2,3)}}
 rng=np.random.default_rng(35935);vals={k:[] for k in ('pair_ap','behavior_map','joint')}
 for _ in range(1000):
  chosen=rng.choice(tables,len(tables),replace=True);cnt=np.bincount(np.searchsorted(tables,chosen),minlength=len(tables));m=cnt[inv]
  a=rscore(y,m,orders['a']);c=rscore(y,m,orders['b'])
  for k in vals:vals[k].append(c[k]-a[k])
 out['bootstrap']={k:{'mean':float(np.mean(v)),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))]} for k,v in vals.items()}
 pg=out['pu']['candidate50_equalFT']['joint']-out['pu']['baseline80_equalFT']['joint']
 fg=[out['folds'][str(f)]['candidate50_equalFT']['joint']-out['folds'][str(f)]['baseline80_equalFT']['joint'] for f in (1,2)]
 out['diagnostic']={'pooled_joint_gain':pg,'fold_joint_gains':fg,'both_folds_positive':bool(all(v>0 for v in fg)),'automatic_advancement':False}
 out['csv_built']=False;out['upload_performed']=False
 DEST.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
