"""Fit the admitted fixed risk blend, preserving every incumbent other field."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import json
from datetime import datetime, timezone
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT, sha, matrix, MODEL_PARAMS, TRAIN_VIEWS, LABELS
from conditional_bucket_features import bucket_features, BOUNDARIES
from run_conditional_policy_screen import hand_features
from prepare_reserved_tabicl import data_views
from prepare_attention_reranker import expanded_views

OUT=ROOT/'outputs_v15_bucket'
EXP=ROOT/'reports/conditional-bucket-experiment'
OLD=ROOT/'outputs_v11_conditional'
SOURCE=ROOT/'outputs_v13_coverage/submission_conditional_coverage.csv'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    admission=json.loads((EXP/'admission.json').read_text()); assert admission['admitted'] and admission['candidate']=='bucket_equal_blend'
    experiment=json.loads((EXP/'plan.json').read_text())
    sources=['fit_bucket_production.py','infer_bucket_production.py','assemble_bucket_production.py','conditional_bucket_features.py','conditional_inference_worker.py','conditional_action_features.py','run_conditional_policy_screen.py','advanced_features.py']
    actionpaths=sorted((ROOT/'reports/conditional-action-experiment/action-cache').glob('*.parquet')); assert len(actionpaths)==20
    oldplan=json.loads((OLD/'policy/plan.json').read_text()); assert {p.name:sha(p) for p in actionpaths}==oldplan['action_cache_sha256']
    identity={'created_at':datetime.now(timezone.utc).isoformat(),'admission_sha256':sha(EXP/'admission.json'),
        'experiment_plan_sha256':sha(EXP/'plan.json'),'sources':{p:sha(ROOT/'src'/p) for p in sources},
        'recipe':'Exactly equal incumbent/bucket risk blend on all evaluation pairs. Preserve v13 behavior and every evidence field. No new ranking or subgroup exceptions.',
        'parameters':experiment['parameters'],'pair_parameters':MODEL_PARAMS,'inputs':experiment['inputs'],'boundaries':BOUNDARIES,
        'labels_sha256':sha(ROOT/'data/development_labels.csv'),'action_cache_sha256':oldplan['action_cache_sha256'],
        'incumbent_policy_plan_sha256':sha(OLD/'policy/plan.json'),'incumbent_pair_model_sha256':sha(OLD/'pair-model.joblib'),
        'incumbent_submission_sha256':sha(SOURCE),'protected':experiment['protected'],
        'development_reference':'Same three pool folds as production v11. Each training action is scored by normal references excluding its pool. Final reference uses all confirmed normal development pairs.',
        'inference':'20 batches; reproduce all 132 incumbent conditional pair aggregates and incumbent risks exactly before accepting new risks. Full CSV membership validation required.'}
    OUT.mkdir(exist_ok=False); write(OUT/'plan.json',identity); dest=OUT/'policy'; dest.mkdir()
    labels=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','label','label_status')
    data=pl.read_parquet(actionpaths).join(labels,on='pair_id',validate='m:1').sort('pair_id','hand_id','direction','action_no')
    receipts=[]
    for name in ['fold0','fold1','fold2','final']:
        ref=json.loads((OLD/'policy'/(name+'.json')).read_text()); oldpath=OLD/'policy'/(name+'-models.joblib')
        assert ref['plan_sha256']==identity['incumbent_policy_plan_sha256'] and sha(oldpath)==ref['model_sha256']
        trainpools=ref['training_pools']; scorepools=ref['scored_pools']; assert not set(trainpools)&set(scorepools)
        training=data.filter((pl.col('label')==0)&pl.col('table_id').is_in(trainpools)); assert set(training['label_status'])=={'confirmed_non_target'}
        training=training.with_columns(pl.Series('sample_order',np.random.default_rng(55193).random(training.height))).sort('pair_id','sample_order').group_by('pair_id',maintain_order=True).head(128)
        if training.height>100000: training=training.sample(n=100000,seed=55193)
        assert training.height==ref['training_actions']
        x=matrix(training,identity['inputs']); y=np.digitize(training['partner_strength'].to_numpy(),BOUNDARIES)
        model=lgb.LGBMClassifier(**identity['parameters']).fit(x,y); assert list(model.classes_)==[0,1,2,3,4]
        path=dest/(name+'.joblib'); joblib.dump({'model':model,'inputs':identity['inputs'],'boundaries':BOUNDARIES},path)
        np.testing.assert_array_equal(model.predict_proba(x[:64]),joblib.load(path)['model'].predict_proba(x[:64]))
        receipt={'plan_sha256':sha(OUT/'plan.json'),'model_sha256':sha(path),'training_pools':trainpools,'scored_pools':scorepools,'training_actions':training.height,'reload_exact':True}
        if scorepools:
            scored=data.filter(pl.col('table_id').is_in(scorepools)); z=matrix(scored,identity['inputs']); old=joblib.load(oldpath)
            ap=old['classifier'].predict_proba(z,num_threads=2)
            control=hand_features(scored,ap,old['regressor'].predict(z,num_threads=2)).sort('pair_id','hand_id')
            original=pl.read_parquet(OLD/'policy'/(name+'.parquet')).sort('pair_id','hand_id')
            assert sha(OLD/'policy'/(name+'.parquet'))==ref['feature_sha256']
            assert control.equals(original)
            result=bucket_features(scored,ap,model.predict_proba(z,num_threads=2)); result.write_parquet(dest/(name+'.parquet'))
            receipt.update(original_features_exact=True,features_sha256=sha(dest/(name+'.parquet')))
        write(dest/(name+'.json'),receipt); receipts.append(receipt); print(json.dumps({'production_reference':name,'training_actions':training.height}),flush=True)
    cb=pl.concat([pl.read_parquet(dest/('fold'+str(i)+'.parquet')) for i in range(3)]).sort('pair_id','hand_id')
    assert cb.height==223749 and cb['pair_id'].n_unique()==1860 and cb.select(pl.struct('pair_id','hand_id').n_unique()).item()==cb.height
    cb.write_parquet(dest/'development-hand-features.parquet'); cbf=[c for c in cb.columns if c.startswith('cb_')]; assert len(cbf)==80
    ca=pl.read_parquet(OLD/'policy/development-hand-features.parquet'); oldsummary=json.loads((OLD/'policy/summary.json').read_text()); assert sha(OLD/'policy/development-hand-features.parquet')==oldsummary['features_sha256']
    raw,full,views=data_views(); full,views=expanded_views(raw,full,views,ca,oldsummary['features']); full,views=expanded_views(raw,full,views,cb,cbf)
    oldpair=joblib.load(OLD/'pair-model.joblib'); features=oldpair['features']+[c+s for s in ['_mean','_std','_max','_q95'] for c in cbf]; assert len(features)==1217
    training=pl.concat([full]+[views[k] for k in TRAIN_VIEWS]); assert training.height==5580
    model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(training,features),np.array([LABELS[f] for f in training['behavior_family']]),sample_weight=np.full(training.height,1/3))
    path=OUT/'pair-model.joblib'; joblib.dump({'model':model,'features':features,'bucket_features':cbf,'events':oldpair['events']},path)
    np.testing.assert_array_equal(model.predict_proba(matrix(training.head(64),features)),joblib.load(path)['model'].predict_proba(matrix(training.head(64),features)))
    write(OUT/'fit.json',{'plan_sha256':sha(OUT/'plan.json'),'pair_model_sha256':sha(path),'policy_receipts':receipts,'development_features_sha256':sha(dest/'development-hand-features.parquet'),'pair_training_rows':5580,'pair_features':1217,'reload_exact':True,'incumbent_development_features_exact':True})
    print(json.dumps({'production_fit_complete':True,'plan_sha256':sha(OUT/'plan.json'),'pair_model_sha256':sha(path)}),flush=True)

if __name__=='__main__': main()
