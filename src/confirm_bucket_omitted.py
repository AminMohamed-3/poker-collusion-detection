"""All omitted-family risk checks; failed evidence endpoint remains excluded."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
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
from evaluate_mil_pairs import scorecard

OUT=ROOT/'reports/conditional-bucket-experiment'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    parent=json.loads((OUT/'plan.json').read_text()); screen=json.loads((OUT/'screen/screen.json').read_text())
    assert all(screen['decisions'][r]['initial_nonregression_pass'] for r in parent['risk_recipes'])
    planpath=OUT/'omitted-plan.json'; plan={'created_at':datetime.now(timezone.utc).isoformat(),'parent_plan_sha256':sha(OUT/'plan.json'),'screen_sha256':sha(OUT/'screen/screen.json'),'source_sha256':sha(__file__),'recipes':parent['risk_recipes'],'mean_sparse_delta_min':.01,'worst_sparse_delta_min':-.03,'jobs':['omit_directed_transfer','omit_soft_play','omit_coordinated_isolation'],'evidence_eligible':False}
    assert not planpath.exists(); write(planpath,plan)
    raw,full,baseviews=data_views(); ca,summary=verified_policy(ROOT/'reports/conditional-action-experiment/policy-screen'); cb=pl.read_parquet(OUT/'policy-screen/hand-features.parquet')
    stored=json.loads((OUT/'policy-screen/summary.json').read_text()); assert sha(OUT/'policy-screen/hand-features.parquet')==stored['hand_features_sha256']
    caf=[c for c in ca.columns if c.startswith('ca_')]; cbf=[c for c in cb.columns if c.startswith('cb_')]
    full,views=expanded_views(raw,full,baseviews,ca,caf); full,views=expanded_views(raw,full,views,cb,cbf)
    pools=summary['completed'][-1]['scored_pools']; rows=[]; pred=[]
    for job in plan['jobs']:
        dest=OUT/job; dest.mkdir(exist_ok=False); heldpath,_,modeldir,_=locations(job); keys=pl.read_csv(heldpath)
        train=full.filter(~pl.col('table_id').is_in(pools)&(pl.col('behavior_family')!=job.removeprefix('omit_'))); held=full.join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id'); assert_disjoint(train,held)
        augmented=pl.concat([train]+[views[k].join(train.select('pair_id'),on='pair_id',how='semi') for k in TRAIN_VIEWS])
        baseline=joblib.load(modeldir/'conditional-pair.joblib'); names=baseline['features']+[c+s for s in ['_mean','_std','_max','_q95'] for c in cbf]
        model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(augmented,names),np.array([LABELS[f] for f in augmented['behavior_family']]),sample_weight=np.full(augmented.height,1/3)); joblib.dump({'model':model,'features':names},dest/'pair.joblib')
        for history,pf in [('full',held),('shortened',views[HELDOUT_VIEW].join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id'))]:
            p=baseline['model'].predict_proba(matrix(pf,baseline['features']),num_threads=2); control=1-p[:,0]; roles=baseline['model'].classes_[1:][np.argmax(p[:,1:],axis=1)]; y=np.array([LABELS[f] for f in pf['behavior_family']]); base=scorecard(y,control,roles)
            old=pl.read_csv(ROOT/'reports/diverse-next-day'/job/'pair-predictions.csv').filter((pl.col('history')==history)&(pl.col('candidate')=='baseline')).sort('pair_id'); assert pf['pair_id'].to_list()==old['pair_id'].to_list(); np.testing.assert_array_equal(control,old['risk'].to_numpy())
            risk=1-model.predict_proba(matrix(pf,names),num_threads=2)[:,0]
            for candidate,v in [('baseline',control),('bucket_risk',risk),('bucket_equal_blend',(control+risk)/2)]:
                scores=scorecard(y,v,roles); rows.append({'job':job,'history':history,'candidate':candidate,**scores,'delta':{k:scores[k]-base[k] for k in scores}})
                pred.extend({'pair_id':pid,'job':job,'history':history,'candidate':candidate,'risk':float(value)} for pid,value in zip(pf['pair_id'],v))
        print(json.dumps({'completed':job}),flush=True)
    decisions={}
    for candidate in parent['risk_recipes']:
        values=[r['delta']['sparse_ap'] for r in rows if r['candidate']==candidate]; mean=float(np.mean(values)); worst=min(values); decisions[candidate]={'passed_omitted_gate':mean>=.01 and worst>=-.03,'mean_sparse_delta':mean,'worst_sparse_delta':worst}
    pl.DataFrame(pred).write_csv(OUT/'omitted-predictions.csv'); result={'plan_sha256':sha(planpath),'decisions':decisions,'metrics':rows}; write(OUT/'omitted-results.json',result); print(json.dumps(decisions),flush=True)


if __name__=='__main__': main()
