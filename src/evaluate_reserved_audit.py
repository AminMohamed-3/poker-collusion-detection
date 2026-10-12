"""Refit frozen recipes and report every result on the 169-pool audit."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
from datetime import datetime,timezone
import json
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,thin,TRAIN_VIEWS,HELDOUT_VIEW,MODEL_PARAMS,LABELS,stable_ap,assert_disjoint
from prepare_reserved_tabicl import data_views
from evaluate_conditional_action_screen import extra_aggregate,metrics
from evaluate_conditional_confirmation import verified_policy
from evaluate_conditional_behavior import family_metrics
from evidence_capacity_experiment import load_sequence,fit_variant,compare
from relative_evidence_experiment import records,summarize
from rank_evidence import context,family_columns,PARAMS
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ

OUT=ROOT/'reports/conditional-reserved-audit'


def main():
    dest=OUT/'results'; dest.mkdir(exist_ok=False)
    plan=json.loads((OUT/'plan.json').read_text()); pools=plan['held_pools']
    ca,policy=verified_policy(OUT/'remote-returned/policy-screen')
    assert all(r['plan_sha256']==sha(OUT/'policy-package/policy-plan.json') for r in policy['completed'])
    assert set(policy['completed'][-1]['scored_pools'])==set(pools)
    receipt=json.loads((OUT/'gpu-returned/receipt.json').read_text())
    assert receipt['plan_sha256']==sha(OUT/'gpu-package/plan.json') and receipt['predictions_sha256']==sha(OUT/'gpu-returned/predictions.npz')
    identity={'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'plan_sha256':sha(OUT/'plan.json'),
        'policy_summary_sha256':sha(OUT/'remote-returned/policy-screen/summary.json'),'tabicl_receipt_sha256':sha(OUT/'gpu-returned/receipt.json'),
        'gate_application':'Evidence and behavior gates apply separately to full and original pair-thinned 65% histories. Ranker contexts are rebuilt after thinning; pair action references and player baselines retain full observed histories as specified in the plan.'}
    (dest/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    raw,full,baseviews=data_views(); caf=policy['features']; added=[c+s for s in ['_mean','_std','_max','_q95'] for c in caf]
    meta=json.loads((ROOT/'reports/tail-pair-experiment/metrics.json').read_text()); base=meta['base_features']+meta['extra_features']; rolefeatures=meta['base_features']
    combined=raw.select('pair_id','hand_id','table_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    expanded=full.join(extra_aggregate(combined,caf),on=['pair_id','table_id']).sort('pair_id')
    views={k:frame.join(extra_aggregate(thin(combined,*k),caf),on=['pair_id','table_id']).select(expanded.columns).sort('pair_id') for k,frame in baseviews.items()}
    training=pl.concat([expanded.filter(~pl.col('table_id').is_in(pools))]+[views[k].filter(~pl.col('table_id').is_in(pools)) for k in TRAIN_VIEWS])
    heldframes={'full':expanded.filter(pl.col('table_id').is_in(pools)),'shortened':views[HELDOUT_VIEW].filter(pl.col('table_id').is_in(pools))}
    assert_disjoint(training,heldframes['full']); ytrain=np.array([LABELS[f] for f in training['behavior_family']]); weights=np.full(training.height,1/3)
    models={}
    for name,features in [('behavior',rolefeatures),('conditional',base+added)]:
        model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(training,features),ytrain,sample_weight=weights)
        models[name]={'model':model,'features':features}; joblib.dump(models[name],dest/(name+'-pair.joblib'))
    v5=joblib.load(OUT/'v5-control.joblib'); arrays=np.load(OUT/'gpu-package/arrays.npz',allow_pickle=False); tab=np.load(OUT/'gpu-returned/predictions.npz',allow_pickle=False)
    pairpred={}; pairmetrics=[]; pairrows=[]; inverse={v:k for k,v in LABELS.items()}
    for history,frame in heldframes.items():
        assert frame['pair_id'].to_list()==pl.read_csv(OUT/'held-pairs.csv')['pair_id'].to_list()
        control=1-v5['model'].predict_proba(matrix(frame,v5['features']))[:,0]; np.testing.assert_array_equal(control,arrays['control_'+history])
        probability=models['conditional']['model'].predict_proba(matrix(frame,models['conditional']['features']))
        oldprob=models['behavior']['model'].predict_proba(matrix(frame,rolefeatures))
        y=np.array([LABELS[f] for f in frame['behavior_family']]); old=np.argmax(oldprob[:,1:],axis=1)+1; new=np.argmax(probability[:,1:],axis=1)+1
        risks={'v10':.5*(control+tab[history]),'conditional':1-probability[:,0]}
        for candidate,risk in risks.items(): pairmetrics.append({'history':history,'candidate':candidate,**metrics(y,risk)})
        pairpred[history]={'truth':y,'risks':risks,'old':old,'new':new}
        pairrows.extend({'pair_id':p,'history':history,'v10':float(a),'conditional':float(b),'old_class':int(c),'new_class':int(d)} for p,a,b,c,d in zip(frame['pair_id'],risks['v10'],risks['conditional'],old,new))
    pl.DataFrame(pairrows).write_csv(dest/'pair-predictions.csv')
    evidence,current,truths,_,_=load_sequence()
    evidence=context(evidence.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
    evidence=context(evidence.join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1'),caf)
    trainhands=evidence.filter(~pl.col('table_id').is_in(pools)); basefeatures=current+EQ+['context_'+c for c in EQ]
    rankers={}
    for name,features in [('equity',basefeatures),('conditional',basefeatures+caf+['context_'+c for c in caf])]:
        model=fit_variant(trainhands,features,PARAMS); rankers[name]={'model':model,'features':features}; joblib.dump(rankers[name],dest/(name+'-evidence.joblib'))
    variants=[('v10','v10','old','equity'),('v11_risk','conditional','old','equity'),('v11_evidence','v10','old','conditional'),('v11_combined','conditional','old','conditional'),('v12_behavior','conditional','new','conditional')]
    events=[c for c in raw.columns if c.startswith('v2_')]; allrows=[]; results=[]; comparisons=[]
    for history,frame in heldframes.items():
        data=evidence.filter(pl.col('table_id').is_in(pools)); observed=truths
        if history=='shortened':
            keys=thin(raw,*HELDOUT_VIEW).select('pair_id','hand_id')
            data=context(data.join(keys,on=['pair_id','hand_id'],how='semi'),events+SEQ+EQ+caf)
            available={k[0]:set(g['hand_id']) for k,g in data.group_by('pair_id')}
            observed={p:(hands&available[p]) for p,hands in truths.items() if p in available}
            observed={p:h for p,h in observed.items() if h}; data=data.filter(pl.col('pair_id').is_in(list(observed)))
        data=data.sort('pair_id','hand_id'); prediction=pairpred[history]; rows=[]
        for candidate,risk_name,role_name,rank_name in variants:
            assigned=frame.select('pair_id').with_columns(pl.Series('predicted_behavior',[inverse[int(i)] for i in prediction[role_name]]))
            held=family_columns(data.join(assigned,on='pair_id',validate='m:1',maintain_order='left'),'predicted_behavior')
            bundle=rankers[rank_name]; score=bundle['model'].predict(matrix(held,bundle['features']),num_threads=2)
            erows=records(held,score,observed,history,candidate); rows.extend(erows)
            e=summarize(erows); risk=prediction['risks'][risk_name]; pairap=stable_ap(prediction['truth']>0,risk); b=family_metrics(prediction['truth'],risk,prediction[role_name])
            result={'history':history,'candidate':candidate,'pair_ap':pairap,**b,'evidence':e,'composite':.7*pairap+.1*b['behavior_map']+.2*e['map5']}; results.append(result)
            held.select('pair_id','hand_id').with_columns(pl.Series('score',score)).write_parquet(dest/(history+'_'+candidate+'-scores.parquet'))
        allrows.extend(rows)
        comparisons.append({'history':history,'evidence':compare(rows,'v11_evidence',baseline='v10'),'behavior_evidence':compare(rows,'v12_behavior',baseline='v11_combined')})
    pl.DataFrame(allrows).write_csv(dest/'evidence-pairs.csv')
    lookup={(r['history'],r['candidate']):r for r in pairmetrics}; ordinary=[lookup[h,'conditional']['pair_ap']-lookup[h,'v10']['pair_ap'] for h in heldframes]; sparse=lookup['shortened','conditional']['sparse_ap']-lookup['shortened','v10']['sparse_ap']
    lookup={(r['history'],r['candidate']):r for r in results}; behavior=[lookup[h,'v12_behavior']['composite']-lookup[h,'v11_combined']['composite'] for h in heldframes]
    gates={'risk':min(ordinary)>=-.003 and np.mean(ordinary)>0 and sparse>=0,
        'evidence':all(r['evidence']['delta']>=-.005 and min(r['evidence']['by_family_delta'].values())>=-.03 for r in comparisons),
        'behavior':min(behavior)>=-.003}
    payload={'results':results,'pair_metrics':pairmetrics,'comparisons':comparisons,'ordinary_risk_deltas':ordinary,'shortened_sparse_risk_delta':sparse,'behavior_composite_deltas':behavior,
        'gates':{k:bool(v) for k,v in gates.items()},'all_passed':bool(all(gates.values())),'training_pairs':training.height//3,'held_pairs':heldframes['full'].height,'held_pools':len(pools),'v5_control_exact':True,'scope':'Less-reused overlapping historical validation, not untouched data. No tuning.'}
    (dest/'summary.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload),flush=True)

if __name__=='__main__': main()
