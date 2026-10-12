import os,sys,json,hashlib,argparse
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,polars as pl
from sklearn.metrics import average_precision_score
OUT=Path(__file__).resolve().parent;CR=OUT.parent.parent;R=CR.parent;CV=R/'Claude Version'
sys.path.insert(0,str(CV/'src'));sys.path.insert(0,str(CR/'src'))
from evidence_aug.step104_C_typed_decoder_build import chronology,align,decode
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step5_evaluate import half
K=['pair_id','hand_id'];SP='soft_play';DT='directed_transfer';CI='coordinated_isolation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
def check():
 plan=json.loads((OUT/'plan.json').read_text())
 for path,h in plan['sources'].items():assert sha(Path(path))==h,path
 return plan

def prepare():
 plan=check();assert not (OUT/'predictions-frozen.json').exists()
 sc=pl.read_parquet(CV/'outputs_step251_20260917/audit_scores_step251.parquet').sort(K)
 audit=pl.read_parquet(CV/'experiments/label_process_20260911/audit-oof.parquet',columns=K+['table_id','behavior_family']).sort(K)
 assert sc.select(K).equals(audit.select(K)) and audit.height==19325 and audit['pair_id'].n_unique()==162
 frame=chronology(audit.with_columns(pl.Series('score',sc['v1_DTfp3'].to_numpy())))
 assert frame.select(K).equals(sc.select(K))
 fam=frame['behavior_family'].to_numpy();base=sc['v1_DTfp3'].to_numpy();top=base>5
 assert top.sum()==3240
 attrs=pl.read_parquet(CV/'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet')
 typed=align(frame,attrs)&(fam!=CI)
 keys=pl.read_parquet(CV/'experiments/step254_mechanical_tensor_20260917/hand_oof_step254.parquet',columns=K+['mechplus_p_any','mechplus_p_DT','mechplus_p_SP','mechplus_p_CI'])
 registered=np.load(CV/'experiments/step254_mechanical_tensor_20260917/oof_mechplus.npy')
 g4=np.load(CV/'experiments/step266_fold_average_calibration_20260918/oof_G4.npy')
 assert registered.shape==g4.shape==(4,keys.height)
 for i,n in enumerate(['any','DT','SP','CI']):assert np.allclose(registered[i],keys[f'mechplus_p_{n}'].to_numpy(),atol=1e-7)
 assert np.array_equal(g4[0],registered[0]) and np.array_equal(g4[3],registered[3])
 ev=keys.select(K).with_columns(pl.Series('evt_DT',g4[1]),pl.Series('evt_SP',g4[2]))
 aligned=frame.select(K).join(ev,on=K,how='left',validate='1:1',maintain_order='left')
 assert aligned.select(K).equals(frame.select(K)) and aligned['evt_DT'].null_count()==0 and aligned['evt_SP'].null_count()==0
 current=base.copy()
 for family,column in [(DT,'evt_DT'),(SP,'evt_SP')]:
  mask=top&(fam==family);e=np.clip(aligned[column].to_numpy()[mask].astype(np.float64),1e-9,1-1e-9);odds=e/(1-e)*10.;pe=odds/(1+odds)
  current[mask]=10.+np.clip(base[mask]-10.,1e-9,1.)**.75*np.clip(pe,1e-9,1-1e-9)**.25
 rule=np.load(CV/'experiments/step277_generator_rule_20260919/induced_rule_audit.npy')
 assert rule.shape==base.shape and np.isfinite(rule).all() and np.all((rule>=0)&(rule<=1))
 def add_rule(scores,w):
  new=scores.copy()
  if w:
   m=top&(fam==SP);new[m]=10.+np.clip(scores[m]-10.,1e-9,1.)**(1-w)*np.clip(rule[m],1e-9,1.)**w
  return new
 raw={'bare_control':base,'bare_rule_control':add_rule(base,.3),'current_reference':current,'current_plus_rule':add_rule(current,.3)}
 assert np.array_equal(add_rule(current,0),current)
 pred=frame.select(K+['table_id','behavior_family']).with_columns(pl.Series('is_top20',top))
 for name,s in raw.items():
  assert np.isfinite(s).all()
  decoded=decode(frame.with_columns(pl.Series('score',s)),typed)
  pred=pred.with_columns(pl.Series(name,decoded),pl.Series(name+'_predecode',s))
 assert np.array_equal(pred['current_reference'].to_numpy()[fam!=SP],pred['current_plus_rule'].to_numpy()[fam!=SP])
 assert np.array_equal(decode(frame.with_columns(pl.Series('score',add_rule(current,0))),typed),pred['current_reference'].to_numpy())
 p=OUT/'predictions.parquet';assert not p.exists();pred.write_parquet(p)
 write(OUT/'predictions-frozen.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':sha(p),'plan_sha256':sha(OUT/'plan.json'),'rows':pred.height,'pairs':162,'top20_rows':3240,'neutral_w0_bit_exact':True,'non_SP_predictions_identical':True,'fits':0,'quality_reads':0,'scope':'Auditable OOF production-form composition; G4 train-size differs from deployed five-member ensemble.'})
 print(json.dumps({'predictions_frozen':True,'rows':pred.height,'quality_evaluations':0}))

def evaluate():
 plan=check();freeze=json.loads((OUT/'predictions-frozen.json').read_text());assert freeze['plan_sha256']==sha(OUT/'plan.json') and freeze['sha256']==sha(OUT/'predictions.parquet')
 assert not (OUT/'evaluation-started.json').exists()
 write(OUT/'evaluation-started.json',{'utc':datetime.now(timezone.utc).isoformat(),'single_fixed_comparison':True})
 d=pl.read_parquet(OUT/'predictions.parquet')
 truth=pl.read_parquet(CV/'experiments/label_process_20260911/audit-oof.parquet',columns=K+['relevance']).sort(K)
 assert d.select(K).equals(truth.select(K));d=d.with_columns(truth['relevance'])
 truths={k[0]:set(g['hand_id']) for k,g in d.filter(pl.col('relevance')==1).group_by('pair_id')}
 fams=dict(d.select('pair_id','behavior_family').unique().iter_rows());pools=dict(d.select('pair_id','table_id').unique().iter_rows());qids=sorted(truths)
 names=['bare_control','bare_rule_control','current_reference','current_plus_rule'];scores={};rawscores={}
 for n in names:
  _,scores[n]=summarize(d.select(K+['table_id',pl.col(n).alias('score')]),truths,fams)
  _,rawscores[n]=summarize(d.select(K+['table_id',pl.col(n+'_predecode').alias('score')]),truths,fams)
 means={n:float(np.mean([scores[n][q] for q in qids])) for n in names}
 old=json.loads((CV/'experiments/step277_generator_rule_20260919/gen_honest.json').read_text())['arms']['soft_play|w=0.30']
 oldg=json.loads((CV/'experiments/step266_fold_average_calibration_20260918/fusion_check266.json').read_text())['G4:COMP_DT25_SP25']
 controls={'bare_baseline':abs(means['bare_control']-.7337397119341563)<1e-12,'bare_rule_parent':abs(means['bare_rule_control']-old['map5'])<5.01e-7,'G4_parent':abs(means['current_reference']-means['bare_control']-oldg['delta'])<5.01e-7}
 if not all(controls.values()):
  write(OUT/'control-failure.json',{'controls':controls,'baseline_means':{n:means[n] for n in names[:-1]}})
  raise AssertionError('Parent replay failed; do not interpret the candidate')
 delta=np.array([scores['current_plus_rule'][q]-scores['current_reference'][q] for q in qids]);bare=np.array([scores['bare_rule_control'][q]-scores['bare_control'][q] for q in qids])
 tabs=sorted(set(pools.values()));ss=np.array([sum(delta[i] for i,q in enumerate(qids) if pools[q]==t) for t in tabs]);ns=np.array([sum(pools[q]==t for q in qids) for t in tabs]);ix=np.random.default_rng(20260919).integers(0,len(tabs),(5000,len(tabs)));bs=ss[ix].sum(1)/ns[ix].sum(1)
 ci=np.quantile(bs,[.025,.975]).tolist();hd={str(h):float(np.mean([delta[i] for i,q in enumerate(qids) if half(pools[q])==h])) for h in [0,1]};fd={f:float(np.mean([delta[i] for i,q in enumerate(qids) if fams[q]==f])) for f in [DT,SP,CI]}
 rankmetrics={}
 for n in ['current_reference','current_plus_rule']:
  num=den=0.;aps=[]
  for key,g in d.filter(pl.col('is_top20')&(pl.col('behavior_family')==SP)).group_by('pair_id'):
   y=g['relevance'].to_numpy();p=g[n+'_predecode'].to_numpy();pos=p[y==1];neg=p[y==0]
   num+=float((pos[:,None]>neg[None,:]).sum()+.5*(pos[:,None]==neg[None,:]).sum());den+=len(pos)*len(neg)
   aps.append(average_precision_score(y,p))
  rankmetrics[n]={'SP_pairwise_weighted_AUC':num/den,'SP_mean_within_pair_AP':float(np.mean(aps)),'raw_all_query_MAP5':float(np.mean([rawscores[n][q] for q in qids]))}
 conditions={'positive_delta':bool(delta.mean()>0),'pool_CI95_lower_positive':ci[0]>0,'both_halves_nonnegative':min(hd.values())>=0,'SP_within_pair_AUC_nonnegative':rankmetrics['current_plus_rule']['SP_pairwise_weighted_AUC']>=rankmetrics['current_reference']['SP_pairwise_weighted_AUC'],'SP_within_pair_AP_nonnegative':rankmetrics['current_plus_rule']['SP_mean_within_pair_AP']>=rankmetrics['current_reference']['SP_mean_within_pair_AP']}
 report={'utc':datetime.now(timezone.utc).isoformat(),'means':means,'marginal_rule_gain_bare':float(bare.mean()),'marginal_rule_gain_current':float(delta.mean()),'interaction_difference':float((delta-bare).mean()),'CI95_pool_bootstrap':ci,'bootstrap_positive_fraction':float((bs>0).mean()),'halves':hd,'family_deltas':fd,'ranking_diagnostics':rankmetrics,'up_down_unchanged':[int((delta>0).sum()),int((delta<0).sum()),int((delta==0).sum())],'controls':controls,'gate_conditions':conditions,'development_screen_pass':all(conditions.values()),'verdict':'FIXED_COMPOSITION_DEVELOPMENT_PASS' if all(conditions.values()) else 'FIXED_COMPOSITION_NOT_ESTABLISHED','new_fits':0,'quality_evaluation_invocations':1,'audit_pairs':162,'audit_pools':len(tabs),'caveats':plan['caveats'],'no_submission_prepared':True,'no_weight_or_family_rescue':True}
 per=pl.DataFrame({'pair_id':qids,'table_id':[pools[q] for q in qids],'family':[fams[q] for q in qids],**{n:[scores[n][q] for q in qids] for n in names}});per.write_parquet(OUT/'per-query.parquet')
 write(OUT/'results.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate']);args=p.parse_args();{'prepare':prepare,'evaluate':evaluate}[args.mode]()
