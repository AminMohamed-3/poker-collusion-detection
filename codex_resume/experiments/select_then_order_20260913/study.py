"""Frozen zero-fit selection/order composition on saved training-only T and LightGBM."""
import os
os.environ['POLARS_MAX_THREADS']='2'
import sys,json,hashlib,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import polars as pl
O=Path(__file__).resolve().parent;HERE=O.parents[1];ROOT=HERE.parent;CV=ROOT/'Claude Version';PARENT=HERE/'experiments/tabpfn3_relative_20260912'
sys.path.insert(0,str(CV/'src'))
from evidence_aug.frame import training_relevance
from evidence_aug.step5_evaluate import ap5,half,TIE_SEED
ARMS={'T':('T','T'),'lgbm':('lgbm','lgbm'),'T_select_L_order':('T','lgbm'),'L_select_T_order':('lgbm','T'),'stage1_only':('oof','oof')}
CANDIDATES=['T_select_L_order','L_select_T_order']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,j):
 assert not p.exists(),p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
def prepare():
 scores=PARENT/'new-scores.parquet';assert sha(scores)==read(PARENT/'predictions-frozen.json')['scores_sha256']
 pf=read(PARENT/'preflight.json');assert not set(pf['audit_pools'])&set(pf['excluded_old_outer_pools']) and len(pf['audit_pools'])==104
 source=[Path(__file__),scores,PARENT/'predictions-frozen.json',PARENT/'preflight.json',PARENT/'per-pair-maps.json',ROOT/'data/development_evidence.csv',CV/'src/evidence_aug/frame.py',CV/'src/evidence_aug/step5_evaluate.py']
 protected=[HERE/'outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv',CV/'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv',HERE/'outputs_other_broad_exploratory_20260913/submission_other_conf065_risk005.csv']
 write(O/'plan.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'source_surface':'common65-reference pool-isolated T selection audit,162pairs104pools3saltappearances; no oldouter metrics','new_fits':0,'arms':ARMS,'two_prespecified_candidates':CANDIDATES,'reference':'T alone on identical savedscore rows, exact per-pair metric parity required','rank_rule':'Generate original931207ties after sorting each whole salt by pair_id,hand_id; choose5 using selector score/tie and sort only those5 using orderer score/same tie. Never change selected set.','research_lead_gate':{'mean_delta_strictly_positive':True,'two_candidate_adjusted_ci97_5_lower_strictly_positive':True,'every_salt_nonnegative':True,'both_fixed_pool_halves_nonnegative':True,'each_family_floor':-.005,'negative_stage1_control_must_fail':True},'bootstrap':{'draws':5000,'seed':827311,'unit':'originalpool; one shared draw for allsalts/arms','ci95_quantiles':[.025,.975],'two_candidate_ci97_5_quantiles':[.0125,.9875]},'no_universal_2pct_rejection_bar':True,'target_size_milestone_separate_from_small_gain_evidence':True,'diagnostics_only':['own-five ideal ordering','selected listed-hand counts'],'prediction_freeze_before_truth':True,'retrospective_context':'Suggested after observing strongT and failedsimpleblends; reuseddevelopmentaudit. Not independentconfirmation. No learner/family/weight search after outcomes.','sources':{str(p):sha(p) for p in source},'protected':{str(p):sha(p) for p in protected}})
 print('PLAN_FROZEN',flush=True)
def check():
 p=read(O/'plan.json')
 for n,h in {**p['sources'],**p['protected']}.items():assert sha(n)==h,n
 return p

def predict():
 p=check();assert not (O/'predictions-frozen.json').exists();d=pl.read_parquet(PARENT/'new-scores.parquet').select('pair_id','hand_id','table_id','behavior_family','T','lgbm','oof','salt','fold')
 assert d.height==19325*3;rows=[]
 for salt in range(3):
  frame=d.filter(pl.col('salt')==salt).sort('pair_id','hand_id');assert frame.height==19325
  frame=frame.with_columns(pl.Series('tie',np.random.default_rng(TIE_SEED).random(frame.height)))
  assert frame.select('pair_id','hand_id').n_unique()==frame.height
  for (pair,),group in frame.group_by('pair_id'):
   assert group.height>=5;table=group['table_id'][0];fam=group['behavior_family'][0]
   top={name:group.sort([name,'tie'],descending=[True,False]).head(5) for name in ['T','lgbm','oof']}
   for arm,(select,order) in ARMS.items():
    selected=top[select];ordered=selected.sort([order,'tie'],descending=[True,False])
    assert set(selected['hand_id'])==set(ordered['hand_id']) and ordered.height==5
    rows.append({'pair_id':pair,'table_id':table,'behavior_family':fam,'salt':salt,'arm':arm,'hands':ordered['hand_id'].to_list()})
  print(json.dumps({'salt_frozen':salt,'predicted_lists':len(rows)}),flush=True)
 out=pl.DataFrame(rows).sort('salt','pair_id','arm');assert out.height==162*3*5
 path=O/'rankings.parquet';out.write_parquet(path);check()
 write(O/'predictions-frozen.json',{'plan_sha256':sha(O/'plan.json'),'rankings_sha256':sha(path),'lists':out.height,'new_model_fits':0,'no_evidence_truth_read':True})

def evaluate():
 p=check();f=read(O/'predictions-frozen.json');assert f['plan_sha256']==sha(O/'plan.json') and f['rankings_sha256']==sha(O/'rankings.parquet')
 write(O/'evaluation-started.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'predictions_frozen_sha256':sha(O/'predictions-frozen.json'),'invocation':1})
 pf=read(PARENT/'preflight.json');e=training_relevance(pf['audit_pools']);truth={k[0]:set(g['hand_id']) for k,g in e.group_by('pair_id')};d=pl.read_parquet(O/'rankings.parquet')
 fams=dict(d.select('pair_id','behavior_family').unique().iter_rows());ptab=dict(d.select('pair_id','table_id').unique().iter_rows());pairs=sorted(fams);pools=sorted(set(ptab.values()));assert len(pairs)==162 and len(pools)==104 and set(pairs)==set(truth)
 maps={a:[{} for _ in range(3)] for a in ARMS};diagnostics={a:[{} for _ in range(3)] for a in ['T','lgbm']}
 for row in d.iter_rows(named=True):
  pair=row['pair_id'];arm=row['arm'];salt=row['salt'];hands=row['hands'];maps[arm][salt][pair]=ap5(hands,truth[pair])
  if arm in diagnostics:
   n=sum(h in truth[pair] for h in hands);diagnostics[arm][salt][pair]={'hits':n,'ideal_map':n/min(5,len(truth[pair]))}
 avg={a:{q:float(np.mean([m[q] for m in v])) for q in pairs} for a,v in maps.items()}
 old=read(PARENT/'per-pair-maps.json')['maps'];parity={a:all(np.isclose(avg[a][q],old['oof' if a=='stage1_only' else a][q],rtol=0,atol=1e-15) for q in pairs) for a in ['T','lgbm','stage1_only']};assert all(parity.values())
 ix=np.random.default_rng(827311).integers(0,len(pools),size=(5000,len(pools)));counts=np.array([sum(ptab[q]==pool for q in pairs) for pool in pools]);comparisons={}
 for arm in [*CANDIDATES,'lgbm','stage1_only']:
  delta={q:avg[arm][q]-avg['T'][q] for q in pairs};sums=np.array([sum(delta[q] for q in pairs if ptab[q]==pool) for pool in pools]);vals=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
  z={'mean_delta':float(np.mean(list(delta.values()))),'ci95':np.quantile(vals,[.025,.975]).tolist(),'ci97_5_two_candidate_adjusted':np.quantile(vals,[.0125,.9875]).tolist(),'salt_deltas':[float(np.mean([maps[arm][s][q]-maps['T'][s][q] for q in pairs])) for s in range(3)],'family_deltas':{fam:float(np.mean([delta[q] for q in pairs if fams[q]==fam])) for fam in sorted(set(fams.values()))},'half_deltas':{str(h):float(np.mean([delta[q] for q in pairs if half(ptab[q])==h])) for h in [0,1]},'pairs_better':sum(v>0 for v in delta.values()),'pairs_worse':sum(v<0 for v in delta.values())}
  z['checks']={'positive_mean':z['mean_delta']>0,'adjusted_lower_positive':z['ci97_5_two_candidate_adjusted'][0]>0,'all_salts_nonnegative':min(z['salt_deltas'])>=0,'both_halves_nonnegative':min(z['half_deltas'].values())>=0,'family_retention':min(z['family_deltas'].values())>=-.005};z['passes_lead_gate']=all(z['checks'].values());comparisons[arm]=z
 diag={a:{'own_MAP':float(np.mean(list(avg[a].values()))),'own_five_ideal_order_MAP':float(np.mean([v[q]['ideal_map'] for v in diagnostics[a] for q in pairs])),'mean_selected_listed_hands':float(np.mean([v[q]['hits'] for v in diagnostics[a] for q in pairs]))} for a in diagnostics}
 result={'mean_MAP':{a:float(np.mean(list(v.values()))) for a,v in avg.items()},'comparisons_vs_T':comparisons,'selection_diagnostics':diag,'baseline_per_pair_parity':parity,'negative_control_rejected':not comparisons['stage1_only']['passes_lead_gate'],'qualified_candidates':[a for a in CANDIDATES if comparisons[a]['passes_lead_gate'] and not comparisons['stage1_only']['passes_lead_gate']],'new_fits':0,'old_outer_evidence_read':False,'evaluation_invocations':1,'reused_training_audit':True,'pairs':162,'pools':104,'appearances':486,'protected_unchanged':True}
 check();write(O/'results.json',result);write(O/'per-pair-maps.json',{'maps':avg,'maps_by_salt':maps,'pair_to_pool':ptab,'pair_to_family':fams});print(json.dumps(result),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','predict','evaluate']);q=a.parse_args();globals()[q.mode]()
