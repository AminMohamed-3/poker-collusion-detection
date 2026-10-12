"""Pool-separated unknown-family surrogate for four class-specific novelty experts."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,platform,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import polars as pl
import joblib,lightgbm as lgb,sklearn
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score,average_precision_score
O=Path(__file__).resolve().parent;HERE=O.parents[1];ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
from robust_pair_experiment import MODEL_PARAMS,assert_disjoint
FAMILIES=['none','directed_transfer','soft_play','coordinated_isolation']
EXCLUDED={'pair_id','table_id','player_1','player_2','label','label_status','behavior_family','hand_id'}
COV_PARAMS={'n_quantiles_max':256,'random_state':7329}

class CorrelatedNovelty:
 def __init__(self,n_quantiles_max=256,random_state=7329):
  self.n_quantiles_max=n_quantiles_max;self.random_state=random_state
 def fit(self,x):
  self.quantiles=QuantileTransformer(n_quantiles=min(self.n_quantiles_max,len(x)),output_distribution='normal',subsample=None,random_state=self.random_state).fit(x)
  self.covariance=LedoitWolf().fit(self.quantiles.transform(x));return self
 def score_samples(self,x):return -self.covariance.mahalanobis(self.quantiles.transform(x))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,j):
 assert not p.exists(),p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
def frame():
 labels=pl.read_csv(ROOT/'data/development_labels.csv')
 assert labels.height==1860 and labels['pair_id'].n_unique()==1860
 assert labels.filter(~(((pl.col('label')==0)&(pl.col('label_status')=='confirmed_non_target'))|((pl.col('label')==1)&(pl.col('label_status')=='confirmed_target')))).height==0
 x=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').join(labels,on='pair_id',validate='1:1').sort('pair_id')
 assert x.height==1860 and x['shared_hands'].min()>=38
 cols=[c for c in x.columns if c not in EXCLUDED];assert len(cols)==653 and all(x.schema[c].is_numeric() for c in cols)
 pools=sorted(x['table_id'].unique(),key=lambda s:hashlib.sha256(('open-set-7329/'+s).encode()).hexdigest())
 fold={p:i%5 for i,p in enumerate(pools)}
 return x.with_columns(pl.col('table_id').replace_strict(fold).alias('fold')),cols

def prepare():
 x,cols=frame();sources=[Path(__file__),ROOT/'src/robust_pair_experiment.py',ROOT/'outputs_v2/pairs_development.parquet',ROOT/'data/development_labels.csv']
 protected=[HERE/'outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv',ROOT/'Claude Version/outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv',HERE/'outputs_other_broad_exploratory_20260913/submission_other_conf065_risk005.csv']
 counts={}
 for f in range(5):
  fit=x.filter(~pl.col('fold').is_in([f,(f+1)%5]));cal=x.filter(pl.col('fold')==(f+1)%5);held=x.filter(pl.col('fold')==f)
  for a,b in [(fit,cal),(fit,held),(cal,held)]:assert_disjoint(a,b)
  counts[str(f)]={n:d.group_by('behavior_family').len().to_dicts() for n,d in [('fit',fit),('calibration',cal),('held',held)]}
  assert all(cal.filter(pl.col('behavior_family')==fam).height>=10 for fam in FAMILIES)
 p={'created_utc':datetime.now(timezone.utc).isoformat(),'question':'Does joint-distribution novelty repair poor normal-expert tail recall from the frozen IsolationForest version?', 'parent_study':'open_set_four_experts_20260913', 'adaptive_reason':'Prior all-rejection detected46/148DT,6/132SP,4/92CI; normal-only caught49/148,7/132,7/92. Most misses precede family vetoes. Replace only class inlier model with correlation-sensitive covariance; all splits/features/calibration thresholds/controls unchanged.','primary':'all_experts_reject','diagnostics':['normal_only','all_reject_and_generic_coordination','closed_set_confidence','blind_id_rank'],'features':cols,'anomaly_params':COV_PARAMS,'algorithm':'class-specific empirical normal marginals plus LedoitWolf shrinkage covariance','lgbm_params':MODEL_PARAMS,'fold_assignment':'sha256(open-set-7329/pool) sorted then roundrobin5; whole pools, held=f, calibration=f+1, fitting=other3','split_counts':counts,'class_pvalues':'(1 + count calibration anomaly >= query anomaly)/(n_class_calibration+1); empirical within-class ranks, NOT a probability of cheating and NOT a pool-exchangeable coverage guarantee','thresholds':{'none':.01,'positive_inlier_class':.10,'generic_coordination_normal_calibration_tail':.05},'surrogate':'Each disclosed family absent from every consulted expert, imputer and generic-classifier fitting/calibration; excluded-family expert is never used in that scenario. Experts are independently fitted perclass, so other models can be safely reused.','fit_budget':{'class_covariance_experts':20,'multiclass_generic_and_confidence_controls':20},'quality_lead_gates':{'minimum_each_omitted_family_recall':.20,'maximum_none_false_alarm_rate':.01,'maximum_any_disclosed_class_false_rejection_rate':.10,'minimum_each_omitted_precision_at_hypothetical1pct_unknown':.50,'negative_control_must_not_pass':True},'gate_scope':'Research lead, not automatic deployment or true fourth-class qualification; actual unknown prevalence and mechanism unobserved.','evaluation':'Predictions for all5folds/allscenarios frozen before metric invocation; all1860knownlabels eligible>=38; eachpair held once; pool bootstrap2000 draws for intervals; no hidden/public evaluation data read','normal_population':'Only explicitly confirmed_non_target labels, never all unlisted pairs','prior_exposure':'These development labels/splits have prior research exposure; this is a new preregistered surrogate audit, not a pristine holdout.','sources':{str(p):sha(p) for p in sources+[O.parent/'open_set_four_experts_20260913/results.json',O.parent/'open_set_four_experts_20260913/predictions-frozen.json']},'protected':{str(p):sha(p) for p in protected},'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'lightgbm':lgb.__version__,'numpy':np.__version__},'no_test_data_fits_or_pseudo_labels':True,'no_submission_actions':True}
 write(O/'plan.json',p);print(json.dumps({'prepared':True,'fold_class_counts':counts}),flush=True)

def check():
 p=read(O/'plan.json')
 for n,h in {**p['sources'],**p['protected']}.items():assert sha(n)==h,n
 assert p['anomaly_params']==COV_PARAMS and p['lgbm_params']==MODEL_PARAMS
 return p

def impute_fit(x):
 a=x.copy();a[~np.isfinite(a)]=np.nan
 med=np.array([np.nanmedian(a[:,i]) if np.any(np.isfinite(a[:,i])) else 0. for i in range(a.shape[1])]);return med,np.where(np.isfinite(a),a,med)
def transform(x,med):return np.where(np.isfinite(x),x,med)
def tail_p(cal,query):
 s=np.sort(cal);return (1+len(s)-np.searchsorted(s,query,side='left'))/(len(s)+1)

def fit():
 p=check();assert not (O/'predictions-frozen.json').exists();x,cols=frame();assert cols==p['features'];started=time.monotonic();outputs=[];proof=[];models=0
 np.testing.assert_array_equal(tail_p(np.array([1.,2.,2.,3.]),np.array([0.,2.,4.])),np.array([1.,.8,.2]))
 for f in range(5):
  tr=x.filter(~pl.col('fold').is_in([f,(f+1)%5]));ca=x.filter(pl.col('fold')==(f+1)%5);te=x.filter(pl.col('fold')==f)
  xt=te.select(cols).to_numpy().astype(np.float64);experts={};saved={}
  for fam in FAMILIES:
   train=tr.filter(pl.col('behavior_family')==fam);cal=ca.filter(pl.col('behavior_family')==fam)
   med,xx=impute_fit(train.select(cols).to_numpy().astype(np.float64));model=CorrelatedNovelty(**COV_PARAMS).fit(xx);models+=1
   cal_anom=-model.score_samples(transform(cal.select(cols).to_numpy().astype(np.float64),med));query_anom=-model.score_samples(transform(xt,med));pv=tail_p(cal_anom,query_anom);experts[fam]=pv
   path=O/'models'/f'fold{f}-{fam}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump({'model':model,'medians':med,'calibration_anomaly_scores':cal_anom,'feature_names':cols,'fit_pair_ids':train['pair_id'].to_list(),'calibration_pair_ids':cal['pair_id'].to_list()},path)
   saved[fam]={'path':str(path.relative_to(O)),'sha256':sha(path),'fit_pairs':train.height,'calibration_pairs':cal.height,'minimum_possible_pvalue':1/(cal.height+1)}
   replay=joblib.load(path);np.testing.assert_array_equal(query_anom,-replay['model'].score_samples(transform(xt,replay['medians'])))
  for omitted in [None,*FAMILIES[1:]]:
   known=[fam for fam in FAMILIES if fam!=omitted];included=tr.filter(pl.col('behavior_family').is_in(known));cal=ca.filter(pl.col('behavior_family').is_in(known))
   assert omitted not in set(included['behavior_family']) and omitted not in set(cal['behavior_family'])
   med,xx=impute_fit(included.select(cols).to_numpy().astype(np.float64));yy=np.array([FAMILIES.index(s) for s in included['behavior_family']]);generic=lgb.LGBMClassifier(**MODEL_PARAMS).fit(xx,yy);models+=1
   probs=generic.predict_proba(transform(xt,med));assert generic.classes_[0]==0
   negcal=cal.filter(pl.col('label')==0);negative_risk=1-generic.predict_proba(transform(negcal.select(cols).to_numpy().astype(np.float64),med))[:,0]
   risk=1-probs[:,0];risk_p=tail_p(negative_risk,risk)
   scaled=np.column_stack([experts[fam]/(.01 if fam=='none' else .10) for fam in known]);all_score=-np.max(scaled,axis=1);gated=-np.maximum(-all_score,risk_p/.05)
   cond=probs[:,1:]/np.maximum(probs[:,1:].sum(axis=1,keepdims=True),1e-15);closed=risk*(1-np.max(cond,axis=1))
   blind=np.array([int(hashlib.sha256(('blind-99173/'+s).encode()).hexdigest()[:13],16)/16**13 for s in te['pair_id']])
   out=te.select('pair_id','table_id','fold','behavior_family','label').with_columns(pl.lit(omitted or 'all_known').alias('omitted'),pl.Series('all_experts_reject',all_score),pl.Series('normal_only',-experts['none']/.01),pl.Series('all_reject_and_generic_coordination',gated),pl.Series('closed_set_confidence',closed),pl.Series('blind_id_rank',blind),pl.Series('generic_risk',risk),*[pl.Series('inlier_p_'+fam,experts[fam]) for fam in known])
   outputs.append(out)
   path=O/'models'/f'fold{f}-generic-{omitted or "all_known"}.joblib';joblib.dump({'model':generic,'medians':med,'known_classes':known,'negative_calibration_risk':negative_risk,'fit_pair_ids':included['pair_id'].to_list(),'calibration_pair_ids':cal['pair_id'].to_list(),'features':cols},path)
   proof.append({'fold':f,'omitted':omitted,'known_experts':{fam:saved[fam] for fam in known},'generic_path':str(path.relative_to(O)),'generic_sha256':sha(path),'fit_pools':sorted(included['table_id'].unique()),'calibration_pools':sorted(cal['table_id'].unique()),'held_pools':sorted(te['table_id'].unique()),'omitted_rows_in_fitting_or_calibration':0})
  print(json.dumps({'fold_complete':f,'fits_so_far':models,'seconds':time.monotonic()-started}),flush=True)
 pred=pl.concat(outputs,how='diagonal_relaxed');assert pred.height==4*1860
 pp=O/'predictions.parquet';pred.write_parquet(pp);write(O/'fit-proof.json',{'runs':proof,'fitted_models':models,'seconds':time.monotonic()-started,'class_expert_saved_replay_exact':True})
 check();assert models==40
 write(O/'predictions-frozen.json',{'plan_sha256':sha(O/'plan.json'),'predictions_sha256':sha(pp),'fit_proof_sha256':sha(O/'fit-proof.json'),'all_predictions_before_metrics':True})

def ci_pool_rate(mask,flag,groups,draws):
 pool_index={s:i for i,s in enumerate(np.unique(groups))};idx=np.array([pool_index[s] for s in groups]);den=np.bincount(idx,weights=mask.astype(float),minlength=len(pool_index));num=np.bincount(idx,weights=(mask&flag).astype(float),minlength=len(pool_index))
 n=draws@num;d=draws@den;return [float(v) for v in np.quantile(n[d>0]/d[d>0],[.025,.975])]

def evaluate():
 p=check();f=read(O/'predictions-frozen.json');assert f['plan_sha256']==sha(O/'plan.json') and f['predictions_sha256']==sha(O/'predictions.parquet') and f['fit_proof_sha256']==sha(O/'fit-proof.json')
 write(O/'evaluation-started.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'predictions_frozen_sha256':sha(O/'predictions-frozen.json')})
 d=pl.read_parquet(O/'predictions.parquet');result={'scenarios':{},'gates':{},'actual_fourth_type_truth_available':False,'metric_use':'new surrogate audit on reused development data, not public-score estimate'}
 arms=['all_experts_reject','normal_only','all_reject_and_generic_coordination','closed_set_confidence','blind_id_rank']
 for omitted in ['all_known',*FAMILIES[1:]]:
  sub=d.filter(pl.col('omitted')==omitted).sort('pair_id');assert sub.height==sub['pair_id'].n_unique()==1860
  truth=sub['behavior_family'].to_numpy();groups=sub['table_id'].to_numpy();normal=truth=='none';unknown=truth==omitted;known=(~normal)&(~unknown);pools=np.unique(groups)
  rng=np.random.default_rng(44319);draws=rng.multinomial(len(pools),np.full(len(pools),1/len(pools)),size=2000)
  rows={}
  for arm in arms:
   scores=sub[arm].to_numpy();flag=scores>=(-1. if arm not in ['closed_set_confidence','blind_id_rank'] else (.99 if arm=='blind_id_rank' else float('inf')))
   item={'normal_false_alarms':int((normal&flag).sum()),'normal_count':int(normal.sum()),'normal_false_alarm_rate':float(flag[normal].mean()),'normal_false_alarm_ci95_pool':ci_pool_rate(normal,flag,groups,draws),'known_family_rejections':{fam:{'flagged':int(((truth==fam)&flag).sum()),'count':int((truth==fam).sum()),'rate':float(flag[truth==fam].mean())} for fam in FAMILIES[1:] if fam!=omitted},'flags':int(flag.sum())}
   if omitted!='all_known':
    neg=~unknown;sel=unknown|normal
    recall=float(flag[unknown].mean());fpr=float(flag[neg].mean());fpr_normal=float(flag[normal].mean())
    item.update({'unknown_count':int(unknown.sum()),'unknown_detected':int((unknown&flag).sum()),'unknown_recall':recall,'unknown_recall_ci95_pool':ci_pool_rate(unknown,flag,groups,draws),'unknown_vs_all_known_auc':float(roc_auc_score(unknown,scores)),'unknown_vs_all_known_ap':float(average_precision_score(unknown,scores)),'unknown_vs_normal_auc':float(roc_auc_score(unknown[sel],scores[sel])),'unknown_vs_normal_ap':float(average_precision_score(unknown[sel],scores[sel])),'known_mixture_false_alarm_rate':fpr,'precision_if_unknown_prevalence_1pct':.01*recall/(.01*recall+.99*fpr) if recall+fpr else None,'precision_if_unknown_prevalence_01pct':.001*recall/(.001*recall+.999*fpr) if recall+fpr else None,'precision_if_1pct_unknown_1pct_known_cheating_98pct_normal':.01*recall/(.01*recall+.98*fpr_normal+.01*float(flag[known].mean())) if recall+fpr else None,'by_fold':{str(k):{'unknown_count':int((unknown&(sub['fold'].to_numpy()==k)).sum()),'unknown_recall':float(flag[unknown&(sub['fold'].to_numpy()==k)].mean())} for k in range(5)}})
   if arm=='closed_set_confidence':item['threshold_note']='Ranking-only control; no threshold or emitted unknown labels.'
   rows[arm]=item
  result['scenarios'][omitted]=rows
 for arm in ['all_experts_reject','normal_only','all_reject_and_generic_coordination','blind_id_rank']:
  unknown=[result['scenarios'][fam][arm] for fam in FAMILIES[1:]];allknown=result['scenarios']['all_known'][arm]
  gates={'each_surrogate_recall_ge20pct':all(z['unknown_recall']>=.2 for z in unknown),'none_false_alarm_le1pct':all(z['normal_false_alarm_rate']<=.01 for z in unknown+[allknown]),'all_disclosed_false_rejection_le10pct':all(v['rate']<=.1 for z in unknown+[allknown] for v in z['known_family_rejections'].values()),'each_precision_if1pct_unknown_ge50pct':all(z['precision_if_unknown_prevalence_1pct'] is not None and z['precision_if_unknown_prevalence_1pct']>=.5 for z in unknown)}
  result['gates'][arm]={'checks':gates,'pass':all(gates.values())}
 result['primary_pass']=result['gates']['all_experts_reject']['pass'] and not result['gates']['blind_id_rank']['pass'];result['no_automatic_alternative_promotion']=True
 result['protected_unchanged']=True;check();write(O/'results.json',result)
 print(json.dumps({'primary_pass':result['primary_pass'],'gate_results':result['gates'],'primary_by_omission':{fam:result['scenarios'][fam]['all_experts_reject'] for fam in FAMILIES[1:]}}),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','fit','evaluate']);q=a.parse_args();globals()[q.mode]()
