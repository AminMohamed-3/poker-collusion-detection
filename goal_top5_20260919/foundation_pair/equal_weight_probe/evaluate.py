"""One guarded adaptive 50/50 diagnostic. Produces JSON only; never a submission."""
import argparse,hashlib,json,os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import numpy as np
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent; GP=HERE.parent/'global_pu'; ROOT=HERE.parents[2]; OUT=GP/'vm_bundle/outputs'; HASH_MANIFEST=HERE/'INPUT_HASHES.json'; EXPECTED_HASH_MANIFEST='2b54437d87f5fe293f4810492043fd9549aab030429788bca131d5e3f4035c6a'
OFFICIAL=ROOT/'Codex Resume/research/official_metric_20260912/slash-poker-competition-metric.ipynb'; PARITY=ROOT/'Codex Resume/research/astra_collaboration_20260919/rank_metric_parity.json'
EXPECTED={'returned_verified':'44eccaf5862ed57dd3096eb18fdc26a5494cd6555b62bb0ae477e8fe66582b9e','final2_baseline':'0f114be7048ca1219656ecb75d7c1fc1a1134ea4aec89e0c4c4c3b03bf5ecec9','evaluator_fold1':'2fbe8bb3994704a0d520b0fd38c2269b9f65e0ab7050111f890497bb97942bc0','evaluator_fold2':'758461e7110717d7a3dbf2ded719b0c7918490deacf06b73310dd3f622863e71','predicted_routes':'15e99848a2603eb2af7ad09ffe473d6711f776a063bf34db88e3e7ae69c6d8a3','predicted_routes_receipt':'c443946ce992d42af3b4996752519665ebfde7f5795a4a4383a4f2568cbea092','official_metric_notebook':'3cb11be5c999ada91aa91002f18aabc0c1c48548f1b6aedf3939f31fcd511c5c','rank_metric_parity':'2184d76c821c8cfdd7eb5ccf2a4214769890ec8c4c8b95996e4250c056cc2281','source_evaluator':'b71fa3af0dde64784acfbbef3e101fb588bc446543de6376dda9fda1695c0ed3'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rank2(x):
 r=np.rint(2*rankdata(x,method='average')).astype(np.int64);assert np.allclose(r/2,rankdata(x,method='average'));return r
def ap(y,s,keys):
 base=np.argsort(keys,kind='stable');o=base[np.argsort(-np.asarray(s)[base],kind='stable')];yy=np.asarray(y,dtype=float)[o]
 return 0. if not yy.sum() else float(np.sum((np.cumsum(yy)/np.arange(1,len(yy)+1))*yy)/yy.sum())
def fixed_order(s,keys):
 base=np.argsort(keys,kind='stable');return base[np.argsort(-np.asarray(s)[base],kind='stable')]
def repeated_ap(y,order,multiplicity):
 yy=np.repeat(np.asarray(y,dtype=float)[order],np.asarray(multiplicity,dtype=np.int64)[order]);return 0. if not yy.sum() else float(np.sum((np.cumsum(yy)/np.arange(1,len(yy)+1))*yy)/yy.sum())
def ordered_metric(yclass,risk,route,keys):
 return {'pair':fixed_order(risk,keys),**{str(c):fixed_order(risk*(route==c),keys) for c in (1,2,3)}}
def repeated_score(yclass,multiplicity,orders):
 p=repeated_ap(yclass>0,orders['pair'],multiplicity);by={c:repeated_ap(yclass==int(c),orders[c],multiplicity) for c in ('1','2','3')};b=float(np.mean(list(by.values())));return {'pair_ap':p,'behavior_map':b,'joint_point7P_point1B':.7*p+.1*b}
def assert_bootstrap_equivalence():
 # Shuffled keys, positive copy counts >1, tied scores, masked-route zero ties,
 # an absent behavior class and an all-negative draw exercise P, B and joint.
 keys=np.array(['P3','P1','P6','P4','P2','P5']);tables=np.array(['A','A','B','B','C','C']);route=np.array([1,2,2,3,3,1]);risk=np.array([.5,.5,.2,.2,.8,.2])
 cases=[
  (np.array([1,0,2,0,3,0]),np.array(['C','A','C','B','A'])),
  (np.array([1,0,2,0,0,0]),np.array(['B','B','A','B'])), # class 3 absent
  (np.zeros(6,dtype=int),np.array(['C','C','A','C'])) # all negative
 ]
 for y,chosen in cases:
  ii=np.concatenate([np.flatnonzero(tables==t) for t in chosen]);mult=np.array([sum(t==v for t in chosen) for v in tables]);generic=score(y[ii],risk[ii],route[ii],keys[ii]);fast=repeated_score(y,mult,ordered_metric(y,risk,route,keys));assert all(generic[k]==fast[k] for k in ('pair_ap','behavior_map','joint_point7P_point1B'))
def score(yclass,risk,route,keys):
 p=ap(yclass>0,risk,keys); by={str(c):ap(yclass==c,risk*(route==c),keys) for c in (1,2,3)};b=float(np.mean(list(by.values())));return {'pair_ap':p,'behavior_map':b,'joint_point7P_point1B':.7*p+.1*b,'by_family_ap':by}
def load_fold(f):
 ev=np.load(GP/f'evaluator_fold{f}.npz',allow_pickle=False);parts=[]
 for n in range(1000):
  p=OUT/f'pred_fold{f}_{n:03}.npz'
  if not p.exists():break
  parts.append(np.load(p,allow_pickle=False))
 assert parts;keys=np.concatenate([z['pair_id'] for z in parts]);tables=np.concatenate([z['table_id'] for z in parts]);prob=np.concatenate([z['probabilities'] for z in parts]).astype(np.float64,copy=False);assert prob.dtype==np.float64;assert np.array_equal(keys,ev['pair_id']) and np.array_equal(tables,ev['table_id']) and prob.shape==(len(keys),4);assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all() and np.allclose(prob.sum(1),1,atol=1e-5)
 return {'pair':keys,'table':tables,'prob':prob,'known':ev['known'],'label':ev['label'],'family':ev['family'],'fold':np.full(len(keys),f)}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--run-quality',action='store_true');a=pa.parse_args();dest=HERE/'probe_results.json'
 assert a.run_quality and os.environ.get('ROOT_ADVANCE')=='equal-weight-probe-50-50','Root advance token and --run-quality are both required'
 assert not dest.exists(),'Refusing to overwrite an existing probe result'
 assert_bootstrap_equivalence()
 files={'returned_verified':GP/'returned_verified.json','final2_baseline':GP/'final2_baseline.npz','evaluator_fold1':GP/'evaluator_fold1.npz','evaluator_fold2':GP/'evaluator_fold2.npz','predicted_routes':GP/'predicted_routes.npz','predicted_routes_receipt':GP/'predicted_routes_receipt.json','official_metric_notebook':OFFICIAL,'rank_metric_parity':PARITY,'source_evaluator':GP/'evaluate_outputs.py'}
 hashes={k:sha(v) for k,v in files.items()};assert hashes==EXPECTED;assert sha(HASH_MANIFEST)==EXPECTED_HASH_MANIFEST
 hm=json.loads(HASH_MANIFEST.read_text());assert len(hm['prediction_shards'])==16
 for rel,h in {**hm['prediction_shards'],**hm['runtime_and_replay']}.items():assert sha(GP/rel)==h,rel
 fs=[load_fold(f) for f in (1,2)];allx={k:np.concatenate([z[k] for z in fs]) for k in fs[0]};_,keep=np.unique(allx['pair'],return_index=True);keep=np.sort(keep);x={k:v[keep] for k,v in allx.items()};assert len(x['pair'])==50761 and len(np.unique(x['pair']))==50761 and int(x['known'].sum())==664
 base=np.load(GP/'final2_baseline.npz',allow_pickle=False);assert len(np.unique(base['pair_id']))==len(base['pair_id']);bi={v:i for i,v in enumerate(base['pair_id'])};ix=np.array([bi[v] for v in x['pair']]);assert np.array_equal(base['table_id'][ix],x['table']);b=base['risk_v11_final2'][ix];fast=1-x['prob'][:,0];r2b=rank2(b);r2f=rank2(fast);current=4*r2b+r2f;equal=r2b+r2f
 routes=np.load(GP/'predicted_routes.npz',allow_pickle=False);assert len(np.unique(routes['pair_id']))==len(routes['pair_id']);ri={v:i for i,v in enumerate(routes['pair_id'])};rix=np.array([ri[v] for v in x['pair']]);assert np.array_equal(routes['table_id'][rix],x['table']);route=routes['predicted_class'][rix]
 out={'adaptive_disclosure':'50/50 selected after observing the fixed 80/20 public +0.00361; reused PU/curated surfaces are not independent confirmation','comparator_scope':'current 80/20 recipe evaluated on historical v11-prefix plus STEP163-FINAL2 surrogate; not the exact public .91772 system','weights_tested':[.5],'input_hashes':hashes,'prediction_input_hash_manifest':sha(HASH_MANIFEST),'numerical_convention':{'current20_numerator':'4*r2base+r2fast','equal50_numerator':'r2base+r2fast','r2':'twice scipy average rank, integral ties','probabilities':'explicit float64 before 1-p_none, matching public builder','official_ap':'pair_id ascending then stable descending score'},'pu':{},'curated_664':{},'table_bootstrap':{}}
 for scope,m in [('pu',np.ones(len(b),dtype=bool)),('curated_664',x['known'])]:
  out[scope]['pooled']={'current80_20':score(x['family'][m],current[m],route[m],x['pair'][m]),'equal50_50':score(x['family'][m],equal[m],route[m],x['pair'][m])}
  out[scope]['folds']={}
  for f in (1,2):
   q=m&(x['fold']==f);out[scope]['folds'][str(f)]={'current80_20':score(x['family'][q],current[q],route[q],x['pair'][q]),'equal50_50':score(x['family'][q],equal[q],route[q],x['pair'][q]),'rank_scope':'global ranks, subset after ranking'}
 saved=json.loads((GP/'global_pu_results.json').read_text())['fixed_predicted_route_diagnostics']['pu_all_unknown_as_background']['fusion_80_20'];par=out['pu']['pooled']['current80_20'];assert par['pair_ap']==saved['pair_ap'] and par['behavior_map']==saved['behavior_map'] and par['joint_point7P_point1B']==saved['point7_pair_plus_point1_behavior']
 out['current80_saved_parity']={'exact':True,'pair_ap':par['pair_ap'],'behavior_map':par['behavior_map'],'joint':par['joint_point7P_point1B']}
 rng=np.random.default_rng(19143);tables,table_inverse=np.unique(x['table'],return_inverse=True);orders={'current80_20':ordered_metric(x['family'],current,route,x['pair']),'equal50_50':ordered_metric(x['family'],equal,route,x['pair'])};vals={k:[] for k in ('pair_ap','behavior_map','joint_point7P_point1B')}
 for _ in range(1000):
  chosen=rng.choice(tables,len(tables),replace=True);counts=np.bincount(np.searchsorted(tables,chosen),minlength=len(tables));multiplicity=counts[table_inverse];a0=repeated_score(x['family'],multiplicity,orders['current80_20']);a1=repeated_score(x['family'],multiplicity,orders['equal50_50'])
  for k in vals:vals[k].append(a1[k]-a0[k])
 out['table_bootstrap']={k:{'mean':float(np.mean(v)),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))]} for k,v in vals.items()}
 jp=out['pu']['pooled'];folds=out['pu']['folds'];passed=jp['equal50_50']['joint_point7P_point1B']>jp['current80_20']['joint_point7P_point1B'] and all(folds[str(f)]['equal50_50']['joint_point7P_point1B']>=folds[str(f)]['current80_20']['joint_point7P_point1B'] for f in (1,2))
 out['advance_gate']={'definition':'PU pooled joint 50/50 > current80/20 and both PU-fold joint 50/50 >= current80/20','pass':bool(passed),'action_if_fail':'No CSV and no automatic exception','action_if_pass':'Root weighs the remaining slot; no automatic build or upload'};out['csv_built']=False
 dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['advance_gate'],indent=2))
if __name__=='__main__':main()
