import os,sys,json,hashlib
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
O=Path(__file__).resolve().parent;R=O.parent.parent.parent;S=R/'goal_top5_20260919/foundation_pair/standard';F=S.parent/'screen'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 assert not p.exists(),str(p);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
files=[S/'known_results.json',S/'replay_check.json',S/'evaluate_standard.py',F/'pf35_predictions.npz',F/'tree_oof_predictions.npz',F/'tree_stress_predictions.npz',Path(__file__)]
for f in (0,1):
 for suffix in ['', '_prefix60']:
  files += [S/f'predictions_fold{f}{suffix}.npz', F/f'evaluator_fold{f}{suffix}.npz', F/f'context_fold{f}{suffix}.npz']
plan={'utc':datetime.now(timezone.utc).isoformat(),'purpose':'Official stable AP correction of completed Standard experiment; preserve all original risk arrays/formulas and gates; point estimates only, no model, new arm, rescue or bootstrap search.','sources':{str(p):sha(p) for p in files},'official_alignment':'Pair ID lexical order then stable descending risk. Duplicate prefix rows retain original relative ordering only for descriptive repeated-row diagnostic.','quality_reuse':'Previously evaluated known/prefix labels read once more solely for objective parity correction.'}
write(O/'standard_metric_correction_plan.json',plan)
assert json.loads((S/'replay_check.json').read_text())['exact']
fast=np.load(F/'pf35_predictions.npz',allow_pickle=False);tree=np.load(F/'tree_oof_predictions.npz',allow_pickle=False);ts=np.load(F/'tree_stress_predictions.npz',allow_pickle=False)
def frame(f,prefix=False):
 suffix='_prefix60' if prefix else '';ev=np.load(F/f'evaluator_fold{f}{suffix}.npz',allow_pickle=False);ct=np.load(F/f'context_fold{f}{suffix}.npz',allow_pickle=False);st=np.load(S/f'predictions_fold{f}{suffix}.npz',allow_pickle=False)
 keys=ev['pair_id'];assert np.array_equal(keys,ct['pair_id']) and np.array_equal(keys,st['pair_id'])
 if prefix:lgb=ts[f'prefix60_fold{f}'];pf=fast[f'prefix60_fold{f}']
 else:
  m=tree['fold']==f;assert np.array_equal(keys,tree['pair_id'][m]);lgb=tree['lgb128'][m];pf=fast[f'known_fold{f}']
 for x in [lgb,pf,st['probabilities']]:assert x.shape==(len(keys),4) and np.isfinite(x).all() and ((x>=0)&(x<=1)).all() and np.allclose(x.sum(1),1,atol=1e-5)
 return {'keys':keys,'tables':ev['table_id'],'y':ev['y'],'lgb':lgb,'fast':pf,'standard':st['probabilities']}
def ap(y,s,keys,official):
 if not official:return float(average_precision_score(y,s))
 if not y.sum():return 0.
 # Official first sorts pair IDs, then stable sorts risk.
 bykey=np.argsort(keys,kind='stable');order=bykey[np.argsort(-s[bykey],kind='stable')];hits=y[order]
 return float(np.sum(np.cumsum(hits)/np.arange(1,len(hits)+1)*hits)/hits.sum())
def metrics(d,risk,route,official):
 y=d['y'];k=d['keys'];a=ap(y>0,risk,k,official);b=float(np.mean([ap(y==c,np.where(route==c,risk,0.),k,official) for c in (1,2,3)]))
 return {'n':len(y),'positives':int((y>0).sum()),'pair_ap':a,'behavior_map_predicted':b,'pair_plus_behavior':.7*a+.1*b}
def score(d,official):
 risks={n:1-d[n][:,0] for n in ['standard','fast','lgb']};routes={n:1+np.argmax(d[n][:,1:],axis=1) for n in risks};rank=lambda a:rankdata(a,method='average')/len(a)
 # Preserve the original completed experiment's literal floating arithmetic exactly.
 prim=.8*rank(risks['fast'])+.2*rank(risks['standard']);diagn=.8*rank(risks['lgb'])+.2*rank(risks['standard'])
 out={'Fast':metrics(d,risks['fast'],routes['fast'],official),'Standard_own_route':metrics(d,risks['standard'],routes['standard'],official),'LGB128':metrics(d,risks['lgb'],routes['lgb'],official),'primary_80Fast_20Standard_Fast_route':metrics(d,prim,routes['fast'],official),'diagnostic_80LGB_20Standard_LGB_route':metrics(d,diagn,routes['lgb'],official)}
 for name,a,b in [('primary_delta_vs_Fast','primary_80Fast_20Standard_Fast_route','Fast'),('lgb_analogue_delta_vs_LGB128','diagnostic_80LGB_20Standard_LGB_route','LGB128')]:out[name]={k:out[a][k]-out[b][k] for k in ['pair_ap','behavior_map_predicted','pair_plus_behavior']}
 return out
known=[frame(f) for f in (0,1)];prefix=[frame(f,True) for f in (0,1)];concat=lambda fs:{k:np.concatenate([f[k] for f in fs]) for k in fs[0]}
scopes={'pooled_known':concat(known),'pooled_prefix60':concat(prefix),**{f'known_fold{f}':known[f] for f in (0,1)},**{f'prefix60_fold{f}':prefix[f] for f in (0,1)}}
old=json.loads((S/'known_results.json').read_text());oldsc={'pooled_known':old['pooled_known'],'pooled_prefix60':old['pooled_prefix60'],**{f'known_fold{f}':old['known_folds'][str(f)] for f in (0,1)},**{f'prefix60_fold{f}':old['prefix60_folds'][str(f)] for f in (0,1)}}
out={};maxerr=0.
for name,d in scopes.items():
 sk=score(d,False);of=score(d,True)
 for arm in sk:
  for metric in sk[arm]:maxerr=max(maxerr,abs(sk[arm][metric]-oldsc[name][arm][metric]))
 out[name]={'official':of,'previous_primary_delta':oldsc[name]['primary_delta_vs_Fast'],'unique_pairs':len(np.unique(d['keys'])),'rows':len(d['keys'])}
assert maxerr<1e-12,maxerr
passes=[out[k]['official']['primary_delta_vs_Fast']['pair_plus_behavior']>0 for k in ['pooled_known','known_fold0','known_fold1']]
result={'utc':datetime.now(timezone.utc).isoformat(),'scopes':out,'legacy_replay_max_abs_error':maxerr,'old_gate':old['advance_to_lofo'],'corrected_gate':all(passes),'fixed_gate_conditions':passes,'new_fits':0,'metric_correction_quality_reads':1,'original_predictions_and_blend_arithmetic_unchanged':True,'no_new_LOFO_requested':True,'bootstrap_not_recomputed':'User requested point estimates and gate effect; advance gate uses pooled and both fold point estimates only.','prefix_limit':'If duplicated pair IDs exist, pooled prefix values describe the original repeated-row stress instrument, not a legal unique-pair competition submission.'}
for p,h in plan['sources'].items():assert sha(Path(p))==h,p
write(O/'standard_metric_correction.json',result)
print(json.dumps({'legacy_replay_max_abs_error':maxerr,'old_gate':result['old_gate'],'corrected_gate':result['corrected_gate'],'primary_deltas':{k:{'old':v['previous_primary_delta'],'official':v['official']['primary_delta_vs_Fast'],'rows':v['rows'],'unique_pairs':v['unique_pairs']} for k,v in out.items()}},indent=2))
