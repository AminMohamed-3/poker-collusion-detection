"""Additional pool-disjoint and larger-audit fits for the admitted bucket blend."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse
from datetime import datetime,timezone
import json
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,TRAIN_VIEWS,HELDOUT_VIEW,MODEL_PARAMS,LABELS,assert_disjoint
from prepare_reserved_tabicl import data_views
from prepare_attention_reranker import expanded_views
from evaluate_conditional_confirmation import verified_policy
from diverse_next_day import locations
from conditional_bucket_features import bucket_features,BOUNDARIES
from run_conditional_policy_screen import hand_features
from evaluate_mil_pairs import scorecard

OUT=ROOT/'reports/conditional-bucket-experiment'
SOURCE=ROOT/'reports/conditional-action-experiment'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def freeze():
    assert not (OUT/'confirmation-plan.json').exists()
    omitted=json.loads((OUT/'omitted-results.json').read_text()); assert omitted['decisions']['bucket_equal_blend']['passed_omitted_gate']
    write(OUT/'confirmation-plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),'parent_plan_sha256':sha(OUT/'plan.json'),'omitted_results_sha256':sha(OUT/'omitted-results.json'),'sources':{p:sha(ROOT/'src'/p) for p in ['confirm_bucket_known.py','confirm_bucket_coherent.py']},'candidate':'bucket_equal_blend only','jobs':['fold1','fold2','audit'],'gates':{'known_mean_delta_min':-.0005,'known_worst_delta_min':-.003,'known_mean_sparse_delta_min':-.01,'stress_mean_delta_min':0.,'stress_worst_delta_min':-.003},'coherent':'Six held-history checks, 65% and40% across fold0/1/2. Same fitted weights and rebuilt phase features. No hyperparameter choices.'})
    print(json.dumps({'confirmation_plan_sha256':sha(OUT/'confirmation-plan.json')}))


def build_reference(job,actions,policy,dest,plan):
    ca,summary=verified_policy(policy); pieces=[]; audit=[]
    for name,original in zip(['inner0','inner1','inner2','held'],summary['completed']):
        trainpools=original['training_pools']; scorepools=original['scored_pools']; assert not set(trainpools)&set(scorepools)
        training=actions.filter((pl.col('label')==0)&pl.col('table_id').is_in(trainpools)); assert set(training['label_status'])=={'confirmed_non_target'}
        training=training.with_columns(pl.Series('sample_order',np.random.default_rng(55193).random(training.height))).sort('pair_id','sample_order').group_by('pair_id',maintain_order=True).head(128)
        if training.height>100000: training=training.sample(n=100000,seed=55193)
        model=lgb.LGBMClassifier(**plan['parameters']).fit(matrix(training,plan['inputs']),np.digitize(training['partner_strength'].to_numpy(),BOUNDARIES)); assert list(model.classes_)==[0,1,2,3,4]
        reference=joblib.load(policy/(name+'-models.joblib')); scored=actions.filter(pl.col('table_id').is_in(scorepools)); x=matrix(scored,plan['inputs']); ap=reference['classifier'].predict_proba(x,num_threads=2)
        control=hand_features(scored,ap,reference['regressor'].predict(x,num_threads=2)).sort('pair_id','hand_id'); previous=pl.read_parquet(policy/(name+'.parquet')).sort('pair_id','hand_id')
        assert control.select('pair_id','hand_id').equals(previous.select('pair_id','hand_id')); caf=[c for c in control.columns if c.startswith('ca_')]; np.testing.assert_array_equal(matrix(control,caf),matrix(previous,caf))
        frame=bucket_features(scored,ap,model.predict_proba(x,num_threads=2)); frame.write_parquet(dest/(name+'.parquet')); pieces.append(frame)
        joblib.dump({'model':model,'inputs':plan['inputs'],'boundaries':BOUNDARIES},dest/(name+'.joblib'))
        audit.append({'name':name,'training_pools':trainpools,'scored_pools':scorepools,'original_reference_sha256':sha(policy/(name+'-models.joblib')),'model_sha256':sha(dest/(name+'.joblib')),'feature_sha256':sha(dest/(name+'.parquet')),'original_control_exact':True})
        print(json.dumps({'job':job,'reference_complete':name}),flush=True)
    cb=pl.concat(pieces).sort('pair_id','hand_id'); assert cb.select(pl.struct('pair_id','hand_id').n_unique()).item()==cb.height==223749
    cb.write_parquet(dest/'hand-features.parquet'); write(dest/'summary.json',{'parent_plan_sha256':sha(OUT/'plan.json'),'completed':audit,'features_sha256':sha(dest/'hand-features.parquet')})
    return ca,cb,summary


def run(job):
    frozen=json.loads((OUT/'confirmation-plan.json').read_text()); assert all(sha(ROOT/'src'/p)==h for p,h in frozen['sources'].items()); assert job in frozen['jobs']
    dest=OUT/'confirmation'/job; dest.mkdir(parents=True,exist_ok=False); policyout=dest/'policy'; policyout.mkdir()
    plan=json.loads((OUT/'plan.json').read_text()); heldpath,policy,modeldir,_=locations(job)
    labels=pl.read_csv(ROOT/'data/development_labels.csv'); actions=pl.read_parquet(sorted((SOURCE/'action-cache').glob('*.parquet'))).join(labels.select('pair_id','label','label_status'),on='pair_id',validate='m:1').sort('pair_id','hand_id','direction','action_no')
    ca,cb,summary=build_reference(job,actions,policy,policyout,plan); del actions
    raw,full,baseviews=data_views(); caf=[c for c in ca.columns if c.startswith('ca_')]; cbf=[c for c in cb.columns if c.startswith('cb_')]; full,views=expanded_views(raw,full,baseviews,ca,caf); full,views=expanded_views(raw,full,views,cb,cbf)
    pools=summary['completed'][-1]['scored_pools']; keys=pl.read_csv(heldpath); train=full.filter(~pl.col('table_id').is_in(pools)); held=full.join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id'); assert_disjoint(train,held)
    training=pl.concat([train]+[views[k].join(train.select('pair_id'),on='pair_id',how='semi') for k in TRAIN_VIEWS]); base=joblib.load(modeldir/'conditional-pair.joblib'); names=base['features']+[c+s for s in ['_mean','_std','_max','_q95'] for c in cbf]
    model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(training,names),np.array([LABELS[f] for f in training['behavior_family']]),sample_weight=np.full(training.height,1/3)); joblib.dump({'model':model,'features':names},dest/'pair.joblib'); rows=[]; pred=[]
    for history,pf in [('full',held),('shortened',views[HELDOUT_VIEW].join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id'))]:
        p=base['model'].predict_proba(matrix(pf,base['features']),num_threads=2); control=1-p[:,0]; roles=base['model'].classes_[1:][np.argmax(p[:,1:],axis=1)]; y=np.array([LABELS[f] for f in pf['behavior_family']]); baseline=scorecard(y,control,roles)
        old=pl.read_csv(ROOT/'reports/diverse-next-day'/job/'pair-predictions.csv').filter((pl.col('history')==history)&(pl.col('candidate')=='baseline')).sort('pair_id'); assert pf['pair_id'].to_list()==old['pair_id'].to_list(); np.testing.assert_array_equal(control,old['risk'].to_numpy())
        risk=.5*(control+1-model.predict_proba(matrix(pf,names),num_threads=2)[:,0]); result=scorecard(y,risk,roles)
        rows.append({'job':job,'history':history,'candidate':'bucket_equal_blend',**result,'delta':{k:result[k]-baseline[k] for k in result}})
        pred.extend({'pair_id':pid,'history':history,'baseline':float(a),'bucket_equal_blend':float(b)} for pid,a,b in zip(pf['pair_id'],control,risk))
    pl.DataFrame(pred).write_csv(dest/'pair-predictions.csv'); write(dest/'metrics.json',{'plan_sha256':sha(OUT/'confirmation-plan.json'),'metrics':rows,'model_sha256':sha(dest/'pair.joblib'),'all_controls_exact':True}); print(json.dumps({'job_complete':job,'metrics':rows}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); p.add_argument('--job'); args=p.parse_args(); freeze() if args.freeze else run(args.job)
