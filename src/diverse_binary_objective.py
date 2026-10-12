"""Independent, fixed binary-risk objective; preserve incumbent behavior/evidence."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse,json
from datetime import datetime,timezone
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from diverse_next_day import OUT,write,locations,expanded_views,verified_policy
from robust_pair_experiment import ROOT,sha,matrix,TRAIN_VIEWS,HELDOUT_VIEW,MODEL_PARAMS,LABELS,stable_ap,assert_disjoint
from prepare_reserved_tabicl import data_views
from evaluate_conditional_behavior import family_metrics

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); parser.add_argument('--jobs',nargs='+'); args=parser.parse_args()
    if args.freeze:
        path=OUT/'binary-plan.json'; assert not path.exists()
        write(path,{'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
            'hypothesis':'Binary target-versus-confirmed-normal training uses capacity for overall risk, rather than allocating it across three known target families.',
            'candidates':{'binary':'226-tree binary cross-entropy with original full/.7/.4 views, equal view weights and unchanged 897 features.','binary_mixed':'Equal arithmetic risk mean of binary and incumbent conditional multiclass models.'},
            'parameters':MODEL_PARAMS,'parent_plan_sha256':sha(OUT/'plan.json'),'admission':'Exactly the risk gates in the parent plan. No public-score choice, tuned weight or altered family assignment. No weakening after results.',
            'scope':'All seven existing jobs plus rebuilt coherent histories; no new split selected by outcome.'}); return
    plan=json.loads((OUT/'binary-plan.json').read_text()); assert sha(__file__)==plan['source_sha256']
    raw,full,baseviews=data_views()
    for job in args.jobs:
        dest=OUT/job; dest.mkdir(exist_ok=True)
        if (dest/'binary-metrics.json').exists(): continue
        heldpath,policy,modeldir,_=locations(job); keys=pl.read_csv(heldpath); pools=sorted(keys['table_id'].unique())
        ca,_=verified_policy(policy); caf=[c for c in ca.columns if c.startswith('ca_')]
        expanded,views=expanded_views(raw,full,baseviews,ca,caf)
        train=expanded.filter(~pl.col('table_id').is_in(pools))
        if job.startswith('omit_'): train=train.filter(pl.col('behavior_family')!=job.removeprefix('omit_'))
        held=expanded.join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id'); assert_disjoint(train,held)
        trainkeys=train.select('pair_id'); train=pl.concat([train]+[views[k].join(trainkeys,on='pair_id',how='semi') for k in TRAIN_VIEWS])
        incumbent=joblib.load(modeldir/'conditional-pair.joblib'); features=incumbent['features']
        model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(train,features),train['label'].to_numpy(),sample_weight=np.full(train.height,1/3))
        joblib.dump({'model':model,'features':features},dest/'pair-binary.joblib'); rows=[]; predictions=[]
        for history,frame in [('full',held),('shortened',views[HELDOUT_VIEW].join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id'))]:
            x=matrix(frame,features); p=incumbent['model'].predict_proba(x); base=1-p[:,0]; binary=model.predict_proba(x)[:,1]
            roles=incumbent['model'].classes_[1:][np.argmax(p[:,1:],axis=1)]; y=np.array([LABELS[f] for f in frame['behavior_family']]); weights=np.where(y>0,.01/np.mean(y>0),.99/np.mean(y==0))
            for candidate,risk in [('baseline',base),('binary',binary),('binary_mixed',.5*(base+binary))]:
                ap=stable_ap(y>0,risk); bm=family_metrics(y,risk,roles)['behavior_map']
                rows.append({'job':job,'history':history,'candidate':candidate,'pair_ap':ap,'sparse_ap':stable_ap(y>0,risk,weights),'behavior_map':bm,'pair_behavior':.7*ap+.1*bm})
                predictions.extend({'pair_id':p,'history':history,'candidate':candidate,'risk':float(v),'class':int(c)} for p,v,c in zip(frame['pair_id'],risk,roles))
        pl.DataFrame(predictions).write_csv(dest/'binary-predictions.csv'); write(dest/'binary-metrics.json',{'job':job,'plan_sha256':sha(OUT/'binary-plan.json'),'pair_results':rows,'model_sha256':sha(dest/'pair-binary.joblib')}); print(json.dumps({'completed':job}),flush=True)

if __name__=='__main__': main()
