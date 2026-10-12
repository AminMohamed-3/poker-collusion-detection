"""Original-training inner-pool check of conditional raise-size dispersion."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import polars as pl
import joblib
from lightgbm import LGBMRegressor
H=Path(__file__).resolve().parent;R=H.parents[1];E=H.parent;OLD=E/'data_expansion';L=OLD/'learned'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pinball(y,p,q):
    r=y-p;return np.maximum(q*r,(q-1)*r)
def main():
    assert not (H/'result.json').exists();plan=json.loads((H/'plan.json').read_text());old=json.loads((R/'reports/conditional-action-experiment/policy-plan.json').read_text());cal=json.loads((OLD/'training-calibration.json').read_text());pools=set(cal['training_pools']);assert len(pools)==297
    paths=sorted((R/'reports/conditional-action-experiment/action-cache').glob('*.parquet'))
    for p in paths:assert sha(p)==old['files'][p.relative_to(R/'reports/conditional-action-experiment').as_posix()]
    fit=json.loads((L/'fit.json').read_text());assert sha(L/'models.joblib')==fit['models_sha256'];bundle=joblib.load(L/'models.joblib');inputs=bundle['inputs']
    labels=pl.read_csv(R/'data/development_labels.csv').filter((pl.col('label')==0)&(pl.col('label_status')=='confirmed_non_target')).select('pair_id')
    f=pl.read_parquet(paths).join(labels,on='pair_id',how='semi').filter(pl.col('table_id').is_in(sorted(pools))).select('table_id','pair_id','hand_id','direction','action_no','action_class','observed_log_amount_bb',*inputs)
    f=f.unique(subset=['hand_id','action_no'],keep='first').with_columns(pl.col('hand_id').hash(seed=50191).alias('_sample')).sort('_sample','action_no').drop('_sample')
    assert f.height==fit['confirmed_negative_actions']
    names=sorted(pools);rng=np.random.default_rng(50191);rng.shuffle(names);parts=np.array_split(names,4);sets=[];base=[]
    for i in range(4):
        train=f.filter(~pl.col('table_id').is_in(parts[i].tolist())).head(150000);test=f.filter(pl.col('table_id').is_in(parts[i].tolist()))
        assert sorted(set(train['table_id']))==fit['families'][i]['training_pools'] and sorted(set(test['table_id']))==fit['families'][i]['inner_scored_pools']
        train=train.filter(pl.col('action_class')==3);test=test.filter(pl.col('action_class')==3);x=test.select(inputs).to_numpy();y=test['observed_log_amount_bb'].to_numpy();median=bundle['models'][i]['regressor'].predict(x,num_threads=2)
        pred=np.column_stack([np.log1p(np.maximum(1,np.expm1(median))*(.8+.4*q)) for q in plan['quantiles']]);coverage=float(np.mean((y>=pred[:,0])&(y<=pred[:,2])))
        base.append({'family':i,'actions':len(y),'coverage10_90':coverage});sets.append((train,test,x,y,pred,median));print('baseline',base[-1],flush=True)
    overall=float(np.average([v['coverage10_90'] for v in base],weights=[v['actions'] for v in base]));admit=overall<.7 and sum(v['coverage10_90']<.7 for v in base)>=3
    (H/'baseline.json').write_text(json.dumps({'overall_coverage':overall,'families':base,'permits_quantile_probe':admit,'training_only':True},indent=2))
    if not admit:
        (H/'result.json').write_text(json.dumps({'passes':False,'reason':'Baseline dispersion does not meet the prespecified undercoverage trigger.','baseline_coverage':overall},indent=2));return
    rows=[];saved=[];scores=[]
    for i,(train,test,x,y,pred,median) in enumerate(sets):
        params=dict(n_estimators=140,learning_rate=.05,num_leaves=[15,23,31,15][i],min_child_samples=80,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=50191+i,deterministic=True,force_col_wise=True)
        mm=[];pp=[]
        for q in [.1,.9]:
            m=LGBMRegressor(**params,objective='quantile',alpha=q).fit(train.select(inputs).to_numpy(),train['observed_log_amount_bb'].to_numpy());mm.append(m.booster_);pp.append(m.predict(x));print('fitted family',i,'quantile',q,flush=True)
        # Quantile crossing repair is prespecified: order the lower, median and upper predictions.
        candidate=np.sort(np.column_stack([pp[0],median,pp[1]]),axis=1);loss0=np.mean([pinball(y,pred[:,j],q) for j,q in enumerate(plan['quantiles'])],axis=0);loss1=np.mean([pinball(y,candidate[:,j],q) for j,q in enumerate(plan['quantiles'])],axis=0)
        for name,take in [('all',np.ones(len(y),dtype=bool)),*[(f'street{j}',test['street_index'].to_numpy()==j) for j in range(4)]]:
            if not take.any():continue
            rows.append({'family':i,'context':name,'actions':int(take.sum()),'baseline_loss':float(loss0[take].mean()),'candidate_loss':float(loss1[take].mean()),'gain':float((loss0-loss1)[take].mean()),'baseline_coverage':float(np.mean((y[take]>=pred[take,0])&(y[take]<=pred[take,2]))),'candidate_coverage':float(np.mean((y[take]>=candidate[take,0])&(y[take]<=candidate[take,2])))})
        np.savez_compressed(H/f'family{i}-oof.npz',y=y,baseline=pred,candidate=candidate,groups=test['table_id'].to_numpy().astype(str));saved.append(mm);scores.append((loss0,loss1,y,candidate))
    a=np.concatenate([s[0] for s in scores]);b=np.concatenate([s[1] for s in scores]);gain=float((a-b).mean());relative=gain/float(a.mean());coverage=float(np.mean(np.concatenate([(s[2]>=s[3][:,0])&(s[2]<=s[3][:,2]) for s in scores])));passed=relative>=.05 and all(v['gain']>0 for v in rows if v['context']=='all') and .7<=coverage<=.9
    joblib.dump({'quantiles':[.1,.9],'models':saved,'inputs':inputs,'median_model_source_sha256':sha(L/'models.joblib')},H/'quantile-models.joblib')
    result={'at':datetime.now(timezone.utc).isoformat(),'passes':passed,'baseline_mean_pinball':float(a.mean()),'candidate_mean_pinball':float(b.mean()),'relative_improvement':relative,'baseline_coverage':overall,'candidate_coverage':coverage,'folds_and_contexts':rows,'training_only':True,'outer_validation_read':False,'plan_sha256':sha(H/'plan.json'),'models_sha256':sha(H/'quantile-models.joblib'),'source_sha256':sha(Path(__file__))};(H/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
