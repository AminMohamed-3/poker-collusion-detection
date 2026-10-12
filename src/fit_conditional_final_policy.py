"""After confirmation, fit three-fold normal references and a final reference."""
import os
os.environ['POLARS_MAX_THREADS']='4'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='4'
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from run_conditional_policy_screen import hand_features, sha

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/conditional-action-experiment'
OUT=ROOT/'outputs_v11_conditional/policy'


def main():
    confirmation=ROOT/'reports/conditional-action-confirmation/results/summary.json'
    outcome=json.loads(confirmation.read_text())
    assert outcome['selected_pair'] or outcome['advance_evidence'], 'No confirmed candidate; stop'
    sourceplan=json.loads((SOURCE/'policy-plan.json').read_text())
    paths=sorted((SOURCE/'action-cache').glob('*.parquet'))
    assert len(paths)==20
    plan={'source_sha256':sha(__file__),'reference_code_sha256':sha(ROOT/'src/run_conditional_policy_screen.py'),
        'confirmation_sha256':sha(confirmation),'inputs':sourceplan['inputs'],'parameters':sourceplan['parameters'],
        'action_cache_sha256':{p.name:sha(p) for p in paths},'labels_sha256':sha(ROOT/'data/development_labels.csv'),
        'method':'Three pool folds, seed 39117, confirmed-normal pairs only. Each development action is scored by references excluding its pool. Final references use all confirmed normal development pairs for evaluation inference.',
        'sampling':'Seed 55193, at most 128 actions per pair and 100000 total, as frozen in the validation recipe.',
        'selected_pair':outcome['selected_pair'],'advance_evidence':outcome['advance_evidence']}
    OUT.mkdir(parents=True,exist_ok=True)
    planpath=OUT/'plan.json'
    if planpath.exists(): assert json.loads(planpath.read_text())==plan
    else: planpath.write_text(json.dumps(plan,indent=2)+'\n')
    labels=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','label','label_status')
    assert set(labels.filter(pl.col('label')==0)['label_status'])=={'confirmed_non_target'}
    data=pl.read_parquet(paths).join(labels,on='pair_id',validate='m:1').sort('pair_id','hand_id','direction','action_no')
    pools=np.array(sorted(data['table_id'].unique())); np.random.default_rng(39117).shuffle(pools)
    folds=[set(x.tolist()) for x in np.array_split(pools,3)]
    specs=[('fold'+str(i),set(pools)-held,held) for i,held in enumerate(folds)]+[('final',set(pools),set())]
    receipts=[]
    for name,trainpools,scorepools in specs:
        receipt=OUT/(name+'.json'); artifact=OUT/(name+'-models.joblib'); featurepath=OUT/(name+'.parquet')
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['plan_sha256']==sha(planpath) and saved['model_sha256']==sha(artifact)
            if scorepools: assert saved['feature_sha256']==sha(featurepath)
            receipts.append(saved); continue
        assert not artifact.exists() and not featurepath.exists()
        started=time.perf_counter()
        train=data.filter((pl.col('label')==0)&pl.col('table_id').is_in(sorted(trainpools)))
        rng=np.random.default_rng(55193)
        train=train.with_columns(pl.Series('sample_order',rng.random(train.height))).sort('pair_id','sample_order').group_by('pair_id',maintain_order=True).head(128)
        if train.height>100000: train=train.sample(n=100000,seed=55193)
        X=train.select(plan['inputs']).to_numpy().astype(np.float32)
        classifier=lgb.LGBMClassifier(**plan['parameters']).fit(X,train['action_class'].to_numpy())
        regressor=lgb.LGBMRegressor(**plan['parameters']).fit(X,train['partner_strength'].to_numpy())
        assert np.array_equal(classifier.classes_,np.arange(4))
        bundle={'classifier':classifier,'regressor':regressor,'inputs':plan['inputs']}
        joblib.dump(bundle,artifact)
        reloaded=joblib.load(artifact)
        np.testing.assert_array_equal(classifier.predict_proba(X[:32]),reloaded['classifier'].predict_proba(X[:32]))
        np.testing.assert_array_equal(regressor.predict(X[:32]),reloaded['regressor'].predict(X[:32]))
        saved={'plan_sha256':sha(planpath),'model_sha256':sha(artifact),'training_pools':sorted(trainpools),
            'scored_pools':sorted(scorepools),'training_actions':train.height,'reload_exact':True}
        if scorepools:
            scored=data.filter(pl.col('table_id').is_in(sorted(scorepools)))
            assert not set(train['table_id'])&set(scored['table_id'])
            x=scored.select(plan['inputs']).to_numpy().astype(np.float32)
            features=hand_features(scored,classifier.predict_proba(x),regressor.predict(x))
            features.write_parquet(featurepath)
            saved.update(feature_sha256=sha(featurepath),pair_hands=features.height)
        saved['seconds']=time.perf_counter()-started
        receipt.write_text(json.dumps(saved,indent=2)+'\n'); receipts.append(saved)
        print(json.dumps({'job':name,'seconds':saved['seconds'],'training_actions':train.height}),flush=True)
    combined=pl.concat([pl.read_parquet(OUT/(name+'.parquet')) for name,_,held in specs if held]).sort('pair_id','hand_id')
    assert combined.select(pl.struct('pair_id','hand_id').n_unique()).item()==combined.height==223749
    assert combined['pair_id'].n_unique()==1860
    combined.write_parquet(OUT/'development-hand-features.parquet')
    summary={'completed_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(planpath),'receipts':receipts,
        'features_sha256':sha(OUT/'development-hand-features.parquet'),'rows':combined.height,'features':[c for c in combined.columns if c.startswith('ca_')],
        'final_reference_sha256':sha(OUT/'final-models.joblib')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
