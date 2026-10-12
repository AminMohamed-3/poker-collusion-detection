"""Cross-fitted amount-given-action references, using only confirmed normals."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
from pathlib import Path
import json
import time
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha

OUT=ROOT/'reports/conditional-sizing-experiment'
SOURCE=ROOT/'reports/conditional-action-experiment'


def actual_size(frame):
    return np.log1p(np.expm1(frame['observed_log_amount_bb'].to_numpy())/np.maximum(1,np.expm1(frame['log_pot_bb'].to_numpy())))


def predictors(frame,inputs):
    return np.column_stack([frame.select(inputs).to_numpy(),(frame['action_class'].to_numpy()==3).astype(np.float32)]).astype(np.float32)


def sizing_features(frame,predicted_size,partner_prediction):
    eligible=frame['action_class'].to_numpy()>=2
    residual=np.where(eligible,np.clip(actual_size(frame)-predicted_size,-3,3),0)
    partner=np.clip((frame['partner_strength'].to_numpy()-partner_prediction)/.25,-3,3)
    weak=1-frame['own_strength'].to_numpy(); excess=np.maximum(residual,0)
    values={'cs_size_residual':residual,'cs_absolute_size_residual':np.abs(residual),
        'cs_weak_excess_size':excess*weak,'cs_partner_facing_excess_size':excess*frame['facing_partner'].to_numpy(),
        'cs_partner_size_covariance':residual*partner,
        'cs_preflop_size_information':residual*partner*((frame['street_index'].to_numpy()==0)&(frame['partner_prior_actions'].to_numpy()==0)),
        'cs_weak_multiway_size':excess*weak*(frame['players_active'].to_numpy()>=3)}
    assert all(np.isfinite(v).all() for v in values.values())
    assert all(np.all(v[~eligible]==0) for v in values.values())
    data=frame.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(n,v.astype(np.float32)) for n,v in values.items()])
    return data.group_by('pair_id','hand_id','table_id').agg(*[pl.col(c).mean().alias(c+'_mean') for c in values],*[pl.col(c).abs().max().alias(c+'_absmax') for c in values])


def main():
    identity={'source_sha256':sha(__file__),'plan_sha256':sha(OUT/'plan.json'),'action_plan_sha256':sha(SOURCE/'policy-plan.json')}
    if (OUT/'reference-identity.json').exists(): assert json.loads((OUT/'reference-identity.json').read_text())==identity
    else: (OUT/'reference-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    plan=json.loads((SOURCE/'policy-plan.json').read_text())
    labels=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','label','label_status')
    data=pl.read_parquet(sorted((SOURCE/'action-cache').glob('*.parquet'))).join(labels,on='pair_id',validate='m:1').sort('pair_id','hand_id','direction','action_no')
    assert set(data.filter(pl.col('label')==0)['label_status'])=={'confirmed_non_target'}
    dest=OUT/'policy-screen'; dest.mkdir(exist_ok=True)
    pieces=[]
    for name in ['inner0','inner1','inner2','held']:
        reference=json.loads((SOURCE/'policy-screen'/(name+'.json')).read_text())
        receipt=dest/(name+'.json'); featurepath=dest/(name+'.parquet'); modelpath=dest/(name+'-model.joblib')
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['identity']==identity and saved['feature_sha256']==sha(featurepath) and saved['model_sha256']==sha(modelpath)
            pieces.append(pl.read_parquet(featurepath)); continue
        start=time.perf_counter()
        train=data.filter((pl.col('label')==0)&pl.col('table_id').is_in(reference['training_pools']))
        train=train.with_columns(pl.Series('_sample',np.random.default_rng(55193).random(train.height))).sort('pair_id','_sample').group_by('pair_id',maintain_order=True).head(128)
        if train.height>100000: train=train.sample(n=100000,seed=55193)
        train=train.filter(pl.col('action_class')>=2)
        scored=data.filter(pl.col('table_id').is_in(reference['scored_pools']))
        assert not set(train['table_id'])&set(scored['table_id'])
        parameters={**plan['parameters'],'n_jobs':2}
        model=lgb.LGBMRegressor(**parameters).fit(predictors(train,plan['inputs']),actual_size(train))
        oldpath=SOURCE/'policy-screen'/(name+'-models.joblib'); assert sha(oldpath)==reference['model_sha256']
        old=joblib.load(oldpath)
        size=model.predict(predictors(scored,plan['inputs']),num_threads=2)
        strength=old['regressor'].predict(scored.select(plan['inputs']).to_numpy().astype(np.float32),num_threads=2)
        features=sizing_features(scored,size,strength); features.write_parquet(featurepath)
        joblib.dump({'model':model,'inputs':plan['inputs'],'additional_input':'observed_action_is_raise'},modelpath)
        normal=(scored['label'].to_numpy()==0)&(scored['action_class'].to_numpy()>=2)
        saved={'identity':identity,'feature_sha256':sha(featurepath),'model_sha256':sha(modelpath),
            'training_pools':reference['training_pools'],'scored_pools':reference['scored_pools'],
            'training_call_raise_actions':train.height,'normal_scored_log_size_mse':float(np.mean((actual_size(scored)[normal]-size[normal])**2)),
            'seconds':time.perf_counter()-start}
        receipt.write_text(json.dumps(saved,indent=2)+'\n'); pieces.append(features)
        print(json.dumps({k:v for k,v in saved.items() if k not in ['identity','training_pools','scored_pools']}),flush=True)
    combined=pl.concat(pieces).sort('pair_id','hand_id'); assert combined.height==223749
    assert combined.select(pl.struct('pair_id','hand_id').n_unique()).item()==combined.height
    combined.write_parquet(dest/'hand-features.parquet')
    summary={'features':[c for c in combined.columns if c.startswith('cs_')],'features_sha256':sha(dest/'hand-features.parquet'),'identity':identity}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
