"""Training-only ordinary behavior of players while a named partner is absent."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
H=Path(__file__).resolve().parent;E=H.parent;R=E.parent;N=E/'absent_partner_styles_20260910';U=E/'bet_size_utility_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pick(meta,soft):
    # One deterministic same-pool ordinary pair per positive, where available.
    controls=[]
    for row in soft:
        candidates=[v for v in meta if v['table_id']==row['table_id'] and v['behavior_family']=='none' and v['pair_id'] not in controls]
        if candidates:controls.append(min(candidates,key=lambda v:hashlib.sha256(('absent-style-control-v1:'+v['pair_id']).encode()).hexdigest())['pair_id'])
    return [r['pair_id'] for r in soft]+controls
def extract(data,meta,ids,name):
    rows=[r for r in meta if r['pair_id'] in ids];lab=pl.read_csv(data/'development_labels.csv').filter(pl.col('pair_id').is_in(ids));table=pl.DataFrame(rows).select('pair_id','table_id','behavior_family');lab=lab.select('pair_id','player_1','player_2').join(table,on='pair_id')
    hands=pl.scan_parquet(data/'hands.parquet').filter(pl.col('table_id').is_in(sorted({r['table_id'] for r in rows}))).select('hand_id')
    seats=pl.scan_parquet(data/'seats.parquet').select('hand_id','player_id').join(hands,on='hand_id',how='semi')
    mapping=pl.concat([lab.select('pair_id','table_id','behavior_family',pl.col('player_1').alias('player_id'),pl.col('player_2').alias('partner'),pl.lit(0).alias('direction')),lab.select('pair_id','table_id','behavior_family',pl.col('player_2').alias('player_id'),pl.col('player_1').alias('partner'),pl.lit(1).alias('direction'))])
    alone=seats.join(mapping.lazy(),on='player_id').join(seats.rename({'player_id':'partner'}),on=['hand_id','partner'],how='anti').select('hand_id','player_id','pair_id','table_id','behavior_family','direction')
    a=pl.scan_parquet(data/'actions.parquet').select('hand_id','player_id','action','amount','to_call').join(alone,on=['hand_id','player_id'])
    a=a.with_columns(pl.when(pl.col('action')=='fold').then(0).when(pl.col('action')=='check').then(1).when(pl.col('amount')>pl.col('to_call')).then(3).otherwise(2).alias('kind'))
    actors=a.group_by('pair_id','table_id','behavior_family','direction').agg(pl.len().alias('actions'),*[(pl.col('kind')==i).sum().alias(f'count{i}') for i in range(4)]).collect(engine='streaming');actors=actors.with_columns(*[((pl.col(f'count{i}')+1)/(pl.col('actions')+4)).alias(f'p{i}') for i in range(4)])
    first=actors.filter(pl.col('direction')==0).drop('direction');second=actors.filter(pl.col('direction')==1).drop('direction');pairs=first.join(second,on=['pair_id','table_id','behavior_family'],suffix='_other').filter((pl.col('actions')>=100)&(pl.col('actions_other')>=100));pairs=pairs.with_columns(*[((pl.col(f'p{i}')+pl.col(f'p{i}_other'))/2).alias(f'mean_{i}') for i in range(4)],*[(pl.col(f'p{i}')-pl.col(f'p{i}_other')).abs().alias(f'gap_{i}') for i in range(4)],pl.lit(name).alias('dataset'))
    pairs.write_parquet(N/(name+'-profiles.parquet'));print(name,'pairs',pairs.height,'ordinary actions',int(actors['actions'].sum()),flush=True);return pairs
def main():
    N.mkdir(exist_ok=False);plan={'at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'definition':'Ordinary actions in hands where the named partner was not seated. Original training pools only. Each actor requires100 such actions; Dirichlet1 smoothing. Pair mean and absolute member gap for four action frequencies. No target intervention can occur in these generated hands.','controls':'One fixed hashed confirmed-none pair from the same pool per selected positive, without replacement where available.','source_audit':'Four-fold grouped logistic source classifier on eight profile features, standardization inside fold, class-balanced, seeds19083/27031/35027. Descriptive only; no new admission gate or model selection.','goal':'Distinguish remaining ordinary-player profile coverage from coordinated-action realism. No outer-validation score, filtering or new generator fit.'};(N/'plan.json').write_text(json.dumps(plan,indent=2))
    meta=json.loads((E/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs'];assert len({r['table_id'] for r in meta})==297
    frames={'original':extract(R/'data',meta,pick(meta,[r for r in meta if r['behavior_family']=='soft_play']),'original')};p=json.loads((U/'study-plan.json').read_text());freeze=json.loads((U/'trial/freeze.json').read_text())
    for i,source in enumerate(p['generation_sources']):
        arm=source['arm'];src=Path(source['directory']);m=json.loads((src/'features/metadata.json').read_text())['views'][0];selected=set(freeze['selected_target_pairs'][str(i)]);frames[arm]=extract(src/'features/data',m,pick(m,[r for r in m if r['pair_id'] in selected]),arm)
    names=[f'{p}_{i}' for p in ['mean','gap'] for i in range(4)];out={}
    for behavior in ['soft_play','none']:
        rows=pl.concat([v.select('pair_id','table_id','behavior_family','dataset',*names) for v in frames.values()]).filter(pl.col('behavior_family')==behavior);x=rows.select(names).to_numpy();y=(rows['dataset'].to_numpy()!='original').astype(int);g=rows['table_id'].to_numpy();aucs=[]
        for seed in [19083,27031,35027]:
            pred=np.zeros(len(y))
            for tr,te in StratifiedGroupKFold(4,shuffle=True,random_state=seed).split(x,y,g):
                assert not set(g[tr])&set(g[te]);model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=seed)).fit(x[tr],y[tr]);pred[te]=model.predict_proba(x[te])[:,1]
            aucs.append(float(roc_auc_score(y,pred)))
        out[behavior]={'source_auc_three_splits':aucs,'real_pairs':int((y==0).sum()),'synthetic_pairs':int((y==1).sum()),'real_mean':dict(zip(names,x[y==0].mean(0).tolist())),'synthetic_mean':dict(zip(names,x[y==1].mean(0).tolist())),'real_std':dict(zip(names,x[y==0].std(0).tolist())),'synthetic_std':dict(zip(names,x[y==1].std(0).tolist()))}
    result={'at':datetime.now(timezone.utc).isoformat(),'training_only':True,'outer_labels_or_scores_read':False,'results':out,'not_a_new_gate':True,'plan_sha256':sha(N/'plan.json')};(N/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
