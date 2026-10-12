"""Cross-fitted categorical partner-card residuals; descriptive, not causal tests."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse
from datetime import datetime,timezone
import json
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix
from run_conditional_policy_screen import hand_features

OUT=ROOT/'reports/conditional-bucket-experiment'
SOURCE=ROOT/'reports/conditional-action-experiment'
BOUNDARIES=[.2,.4,.6,.8]
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def bucket_features(frame,action_probability,bucket_probability):
    action=frame['action_class'].to_numpy(); bucket=np.digitize(frame['partner_strength'].to_numpy(),BOUNDARIES)
    ar=np.clip((np.eye(4)[action]-action_probability)/np.sqrt(action_probability*(1-action_probability)+.05),-4,4)
    br=np.clip((np.eye(5)[bucket]-bucket_probability)/np.sqrt(bucket_probability*(1-bucket_probability)+.05),-4,4)
    unseen=((frame['street_index'].to_numpy()==0)&(frame['partner_prior_actions'].to_numpy()==0)).astype(np.float32)
    values={}
    for a in range(4):
        for b in range(5):
            v=ar[:,a]*br[:,b]; values[f'cb_cross_{a}_{b}']=v; values[f'cb_unseen_{a}_{b}']=v*unseen
    data=frame.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(k,v.astype(np.float32)) for k,v in values.items()])
    return data.group_by('pair_id','hand_id','table_id').agg(*[pl.col(k).mean().alias(k+'_mean') for k in values],*[pl.col(k).abs().max().alias(k+'_absmax') for k in values]).sort('pair_id','hand_id')


def freeze():
    OUT.mkdir(exist_ok=False)
    old=json.loads((SOURCE/'policy-plan.json').read_text())
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'sources':{p:sha(ROOT/'src'/p) for p in ['conditional_bucket_features.py','evaluate_conditional_buckets.py']},
        'hypothesis':'The existing action/card residual product sees primarily conditional mean dependence. Five fixed private-strength categories may capture nonlinear action dependence hidden by the continuous mean.',
        'basis_source':'Shah and Peters, Generalised Covariance Measure; Scheidegger et al., Weighted GCM. This is a predictive feature adaptation, not a calibrated hypothesis test or causal evidence.',
        'normal_model':'Five-class partner-strength reference from confirmed-none actions only; same37 own/public pre-action inputs, fixed .2/.4/.6/.8 boundaries, same100-tree parameters and sample cap. Reuse verified normal-action probabilities. Three pool-cross-fitted training references and outer held reference.',
        'parameters':{**old['parameters'],'n_jobs':2},'inputs':old['inputs'],'boundaries':BOUNDARIES,
        'features':'20 products of standardized action/category residuals plus20 preflop-before-partner-action gated products. Per-hand mean/absolute maximum=80. Pair adds320 summaries; rank adds80 raw+80 within-pair contexts.',
        'risk_recipes':['bucket_risk','bucket_equal_blend'],'risk_screen_gate':{'mean_pair_behavior_delta_min':-.0005,'worst_pair_behavior_delta_min':-.003,'mean_sparse_ap_delta_min':-.01},
        'risk_confirmation':'Known means >=-.0005,worst>=-.003; sparse known mean>=-.01; omitted mean sparse gain>=.01,worst>=-.03; coherent/audit mean weighted gain>=0,worst>=-.003.',
        'evidence_recipes':['bucket_evidence'],'evidence_screen_gate':{'full_delta_min':.005,'shortened_delta_min':-.005,'worst_family_delta_min':-.025},
        'evidence_confirmation':'Known pooled full/short gains>=.005, every family>=-.025; omitted mean>=0,worst>=-.025; coherent/audit mean>=0,worst>=-.02.',
        'progression':'Bounded fold0 extraction and pair/ranker fits first; only passing endpoints proceed. Freeze all settings before new outcomes. All splits reused; no public-score fitting or subgroup rescue.',
        'protected':json.loads((ROOT/'reports/mil-pair-experiment/plan.json').read_text())['protected']}
    write(OUT/'plan.json',plan); print(json.dumps({'plan_sha256':sha(OUT/'plan.json')}),flush=True)


def build():
    plan=json.loads((OUT/'plan.json').read_text()); sources=json.loads((OUT/'input-path-repair.json').read_text())['sources'] if (OUT/'input-path-repair.json').exists() else plan['sources']
    assert all(sha(ROOT/'src'/p)==h for p,h in sources.items())
    oldplan=json.loads((SOURCE/'policy-plan.json').read_text())
    relocated={'development_labels.csv':ROOT/'data/development_labels.csv','held-pairs.csv':ROOT/'reports/tabicl-screen/held-pairs.csv','run_conditional_policy_screen.py':ROOT/'src/run_conditional_policy_screen.py'}
    assert all(sha(relocated.get(p,SOURCE/p))==h for p,h in oldplan['files'].items())
    labels=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','label','label_status')
    data=pl.read_parquet(sorted((SOURCE/'action-cache').glob('*.parquet'))).join(labels,on='pair_id',validate='m:1').sort('pair_id','hand_id','direction','action_no')
    dest=OUT/'policy-screen'; dest.mkdir(exist_ok=True); completed=[]
    for name in ['inner0','inner1','inner2','held']:
        receiptpath=dest/(name+'.json')
        if receiptpath.exists():
            receipt=json.loads(receiptpath.read_text()); assert receipt['plan_sha256']==sha(OUT/'plan.json') and sha(dest/(name+'.parquet'))==receipt['features_sha256']; completed.append(receipt); continue
        reference=json.loads((SOURCE/'policy-screen'/(name+'.json')).read_text()); trainpools=reference['training_pools']; scorepools=reference['scored_pools']; assert not set(trainpools)&set(scorepools)
        bundlepath=SOURCE/'policy-screen'/(name+'-models.joblib'); assert sha(bundlepath)==reference['model_sha256']; bundle=joblib.load(bundlepath)
        training=data.filter((pl.col('label')==0)&pl.col('table_id').is_in(trainpools)); assert set(training['label_status'])=={'confirmed_non_target'}
        training=training.with_columns(pl.Series('sample_order',np.random.default_rng(55193).random(training.height))).sort('pair_id','sample_order').group_by('pair_id',maintain_order=True).head(128)
        if training.height>100000: training=training.sample(n=100000,seed=55193)
        target=np.digitize(training['partner_strength'].to_numpy(),BOUNDARIES); model=lgb.LGBMClassifier(**plan['parameters']).fit(matrix(training,plan['inputs']),target)
        assert list(model.classes_)==[0,1,2,3,4]
        scored=data.filter(pl.col('table_id').is_in(scorepools)); x=matrix(scored,plan['inputs']); ap=bundle['classifier'].predict_proba(x,num_threads=2); bp=model.predict_proba(x,num_threads=2)
        control=hand_features(scored,ap,bundle['regressor'].predict(x,num_threads=2)).sort('pair_id','hand_id')
        original=pl.read_parquet(SOURCE/'policy-screen'/(name+'.parquet')).sort('pair_id','hand_id')
        assert control.select('pair_id','hand_id').equals(original.select('pair_id','hand_id'))
        np.testing.assert_array_equal(matrix(control,[c for c in control.columns if c.startswith('ca_')]),matrix(original,[c for c in control.columns if c.startswith('ca_')]))
        result=bucket_features(scored,ap,bp); result.write_parquet(dest/(name+'.parquet')); joblib.dump({'model':model,'inputs':plan['inputs'],'boundaries':BOUNDARIES},dest/(name+'.joblib'))
        receipt={'plan_sha256':sha(OUT/'plan.json'),'training_pools':trainpools,'scored_pools':scorepools,'training_actions':training.height,'scored_actions':scored.height,'original_action_control_exact':True,'old_reference_sha256':sha(bundlepath),'model_sha256':sha(dest/(name+'.joblib')),'features_sha256':sha(dest/(name+'.parquet'))}; write(receiptpath,receipt); completed.append(receipt)
        print(json.dumps({'bucket_reference_complete':name,'scored_actions':scored.height}),flush=True)
    features=pl.concat([pl.read_parquet(dest/(n+'.parquet')) for n in ['inner0','inner1','inner2','held']]).sort('pair_id','hand_id')
    assert features.select(pl.struct('pair_id','hand_id').n_unique()).item()==features.height==223749
    features.write_parquet(dest/'hand-features.parquet'); write(dest/'summary.json',{'plan_sha256':sha(OUT/'plan.json'),'completed':completed,'hand_features_sha256':sha(dest/'hand-features.parquet')}); print(json.dumps({'features_complete':True,'rows':features.height}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); args=p.parse_args(); freeze() if args.freeze else build()
