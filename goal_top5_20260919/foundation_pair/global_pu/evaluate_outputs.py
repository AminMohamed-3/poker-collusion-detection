"""Local-only evaluator. Never copy this file's adjacent evaluator NPZs to the VM."""
import hashlib,json, os
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import numpy as np
from scipy.stats import rankdata
O=Path(__file__).resolve().parent; OUT=O/'vm_bundle/outputs'; KS=(138,508,909,2540); ROOT=O.parents[2]; OFFICIAL=ROOT/'Codex Resume/research/official_metric_20260912/slash-poker-competition-metric.ipynb';PARITY=ROOT/'Codex Resume/research/astra_collaboration_20260919/rank_metric_parity.json'
def rank2(x):
 # Twice the average rank is integral, preserving exact mathematical ties.
 r=np.rint(2*rankdata(x,method='average')).astype(np.int64);assert np.allclose(r/2,rankdata(x,method='average'));return r
def official_ap(y,s,keys):
 base=np.argsort(keys,kind='stable');order=base[np.argsort(-np.asarray(s)[base],kind='stable')];yy=np.asarray(y,dtype=float)[order]
 if not yy.sum():return 0.0
 return float(np.sum((np.cumsum(yy)/np.arange(1,len(yy)+1))*yy)/yy.sum())
def routed(yclass,risk,route,keys):
 by={str(c):official_ap(yclass==c,risk*(route==c),keys) for c in (1,2,3)};return {'behavior_map':float(np.mean(list(by.values()))),'by_family_ap':by}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_fold(f):
 ev=np.load(O/f'evaluator_fold{f}.npz',allow_pickle=False); pieces=[]
 for n in range(1000):
  p=OUT/f'pred_fold{f}_{n:03}.npz'
  if not p.exists():break
  pieces.append(np.load(p,allow_pickle=False))
 assert pieces; keys=np.concatenate([x['pair_id'] for x in pieces]);tables=np.concatenate([x['table_id'] for x in pieces]);probs=np.concatenate([x['probabilities'] for x in pieces]);assert np.array_equal(keys,ev['pair_id']) and np.array_equal(tables,ev['table_id']) and probs.shape==(len(keys),4)
 return {'pair':keys,'table':tables,'prob':probs,'known':ev['known'],'label':ev['label'],'family':ev['family'],'fold':np.full(len(keys),f)}
def metrics(y,s,keys):return {'ap':official_ap(y,s,keys),**{f'top{k}_confirmed':int(y[np.lexsort((keys,-s))[:k]].sum()) for k in KS}}
def main():
 verified=O/'returned_verified.json';assert verified.exists(), 'Run verify_returned.py before any label-bearing evaluation';assert not (O/'global_pu_results.json').exists(), 'Refusing to overwrite or reread an existing quality result'
 fs=[load_fold(f) for f in (1,2)]; allx={k:np.concatenate([z[k] for z in fs]) for k in fs[0]}; _,keep=np.unique(allx['pair'],return_index=True);keep=np.sort(keep)
 x={k:v[keep] for k,v in allx.items()}; base=np.load(O/'final2_baseline.npz',allow_pickle=False);bi={v:i for i,v in enumerate(base['pair_id'])};ix=np.array([bi[v] for v in x['pair']]);assert np.array_equal(base['table_id'][ix],x['table'])
 b=base['risk_v11_final2'][ix];pf=1-x['prob'][:,0];n=len(b);r2b=rank2(b);r2f=rank2(pf);fusion=(4*r2b+r2f)/(10*n); ypos=(x['label']==1); known=x['known']; yknown=(x['label'][known]==1)
 routes=np.load(O/'predicted_routes.npz',allow_pickle=False);ri={v:i for i,v in enumerate(routes['pair_id'])};rix=np.array([ri[v] for v in x['pair']]);assert np.array_equal(routes['table_id'][rix],x['table']);route=routes['predicted_class'][rix]
 assert sha(OFFICIAL)=='3cb11be5c999ada91aa91002f18aabc0c1c48548f1b6aedf3939f31fcd511c5c'
 out={'scope':{'canonical_rows':len(b),'known_rows':int(known.sum()),'known_positive':int(ypos.sum()),'unlabeled':int((~known).sum()),'warning':'Known-positive AP and routed BehaviorMAP over the full population are PU diagnostics, not estimates or bounds on true AP. Unlabeled rows are unknown. All primary AP uses official pair_id-first stable ordering.'},'numerical_convention':{'ap':'sort pair_id ascending, then stable mergesort on descending score; precision at each positive','rank_fusion':'r2=round(2*average_rank); score=(4*r2_baseline+r2_fast)/(10*N), preserving mathematical ties exactly','source_proof':'official_metric_20260912/slash-poker-competition-metric.ipynb plus score_allpairs_fold.py average_precision','ordinary_float_warning':'rank_metric_parity.json found 39/88 mathematical ties split by ordinary float arithmetic on a 1000-row synthetic case'},'input_hashes':{'returned_verified':sha(verified),'final2_baseline':sha(O/'final2_baseline.npz'),'evaluator_fold1':sha(O/'evaluator_fold1.npz'),'evaluator_fold2':sha(O/'evaluator_fold2.npz'),'predicted_routes':sha(O/'predicted_routes.npz'),'predicted_routes_receipt':sha(O/'predicted_routes_receipt.json'),'official_metric_notebook':sha(OFFICIAL),'rank_metric_parity':sha(PARITY)},'pu':{'baseline':metrics(ypos,b,x['pair']),'fast':metrics(ypos,pf,x['pair']),'fusion_80_20':metrics(ypos,fusion,x['pair'])},'source_fold_pu':{},'curated_664':{'coverage':'weak; confirmed rows only','baseline_ap':official_ap(yknown,b[known],x['pair'][known]),'fast_ap':official_ap(yknown,pf[known],x['pair'][known]),'fusion_ap':official_ap(yknown,fusion[known],x['pair'][known])},'family_pu':{}}
 for scope,m in [('pu_all_unknown_as_background',np.ones(len(b),dtype=bool)),('curated_664',known)]:
  yc=x['family'][m];keys=x['pair'][m];rb=routed(yc,b[m],route[m],keys);rf=routed(yc,fusion[m],route[m],keys);pb=official_ap(yc>0,b[m],keys);pfn=official_ap(yc>0,fusion[m],keys);out.setdefault('fixed_predicted_route_diagnostics',{})[scope]={'rows':int(m.sum()),'route_source':'immutable prefix60 predicted_class; canonical fold1-first','baseline':{'pair_ap':pb,**rb,'point7_pair_plus_point1_behavior':.7*pb+.1*rb['behavior_map']},'fusion_80_20':{'pair_ap':pfn,**rf,'point7_pair_plus_point1_behavior':.7*pfn+.1*rf['behavior_map']},'warning':'Predicted routes only; no oracle family routing and no class-probability blend.'}
 # Preserve global ranks, then subset by canonical fold-1-first source. Never rerank within fold.
 for f in (1,2):
  m=x['fold']==f;out['source_fold_pu'][str(f)]={'rows':int(m.sum()),'known_positive':int(ypos[m].sum()),'baseline':metrics(ypos[m],b[m],x['pair'][m]),'fast':metrics(ypos[m],pf[m],x['pair'][m]),'fusion_80_20':metrics(ypos[m],fusion[m],x['pair'][m]),'fusion_minus_baseline_ap':official_ap(ypos[m],fusion[m],x['pair'][m])-official_ap(ypos[m],b[m],x['pair'][m]),'rank_scope':'global canonical ranks, subset after ranking'}
 for c,name in ((1,'directed_transfer'),(2,'soft_play'),(3,'coordinated_isolation')):
  famfusion=(4*r2b+rank2(x['prob'][:,c]))/(10*n);out['family_pu'][name]={'baseline_ap':official_ap(x['family']==c,b,x['pair']),'fast_ap':official_ap(x['family']==c,x['prob'][:,c],x['pair']),'fusion_ap':official_ap(x['family']==c,famfusion,x['pair']),'secondary_oracle_family_ovr':True}
 # Table bootstrap holds the already-frozen global ranks fixed; it estimates this surface only.
 rng=np.random.default_rng(35935);tables=np.unique(x['table']);d=[]
 for _ in range(1000):
  chosen=rng.choice(tables,len(tables),replace=True);ii=np.concatenate([np.flatnonzero(x['table']==t) for t in chosen]);d.append(official_ap(ypos[ii],fusion[ii],x['pair'][ii])-official_ap(ypos[ii],b[ii],x['pair'][ii]))
 out['table_bootstrap_gain']={'mean':float(np.mean(d)),'ci95':[float(np.quantile(d,.025)),float(np.quantile(d,.975))]}
 out['primary_gate']={'definition':'pooled fusion AP gain >0, both canonical-source-fold AP gains >0, table-bootstrap lower 95% bound >0','pooled_gain':out['pu']['fusion_80_20']['ap']-out['pu']['baseline']['ap'],'fold_gains':[out['source_fold_pu'][str(f)]['fusion_minus_baseline_ap'] for f in (1,2)],'bootstrap_lower':out['table_bootstrap_gain']['ci95'][0]};out['primary_gate']['pass']=bool(out['primary_gate']['pooled_gain']>0 and all(v>0 for v in out['primary_gate']['fold_gains']) and out['primary_gate']['bootstrap_lower']>0)
 for k in KS:
  ib=set(np.lexsort((x['pair'],-b))[:k]);ip=set(np.lexsort((x['pair'],-fusion))[:k]);out[f'top{k}_change']={'known_positive_gained':sorted(x['pair'][list((ip-ib)&set(np.flatnonzero(ypos)))].tolist()),'known_positive_lost':sorted(x['pair'][list((ib-ip)&set(np.flatnonzero(ypos)))].tolist()),'unlabeled_promotions':int(sum(not known[i] for i in ip-ib))}
 (O/'global_pu_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['pu'],indent=2))
if __name__=='__main__':main()
