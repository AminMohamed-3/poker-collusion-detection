"""Frozen pair/evidence endpoints for richer private-card residual categories."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import json
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,thin,TRAIN_VIEWS,HELDOUT_VIEW,MODEL_PARAMS,LABELS,assert_disjoint
from prepare_reserved_tabicl import data_views
from prepare_attention_reranker import expanded_views
from evaluate_conditional_confirmation import verified_policy
from evidence_capacity_experiment import load_sequence,fit_variant,compare
from rank_evidence import context,family_columns,PARAMS
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from relative_evidence_experiment import records
from evaluate_mil_pairs import scorecard

OUT=ROOT/'reports/conditional-bucket-experiment'
SOURCE=ROOT/'reports/conditional-action-experiment'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    plan=json.loads((OUT/'plan.json').read_text()); sources=json.loads((OUT/'input-path-repair.json').read_text())['sources'] if (OUT/'input-path-repair.json').exists() else plan['sources']
    assert all(sha(ROOT/'src'/p)==h for p,h in sources.items()); dest=OUT/'screen'; dest.mkdir(exist_ok=False)
    summary=json.loads((OUT/'policy-screen/summary.json').read_text()); assert sha(OUT/'policy-screen/hand-features.parquet')==summary['hand_features_sha256']
    for name,r in zip(['inner0','inner1','inner2','held'],summary['completed']):
        assert sha(OUT/'policy-screen'/(name+'.joblib'))==r['model_sha256']; assert not set(r['training_pools'])&set(r['scored_pools'])
    ca,reference=verified_policy(SOURCE/'policy-screen'); cb=pl.read_parquet(OUT/'policy-screen/hand-features.parquet'); caf=[c for c in ca.columns if c.startswith('ca_')]; cbf=[c for c in cb.columns if c.startswith('cb_')]
    assert len(cbf)==80
    raw,full,baseviews=data_views(); full,views=expanded_views(raw,full,baseviews,ca,caf); full,views=expanded_views(raw,full,views,cb,cbf)
    pools=reference['completed'][-1]['scored_pools']; train=full.filter(~pl.col('table_id').is_in(pools)); held=full.filter(pl.col('table_id').is_in(pools)).sort('pair_id'); assert_disjoint(train,held)
    bundle=joblib.load(SOURCE/'screen-results/conditional-pair.joblib'); extra=[c+s for s in ['_mean','_std','_max','_q95'] for c in cbf]; pairnames=bundle['features']+extra; assert len(pairnames)==1217
    training=pl.concat([train]+[views[k].join(train.select('pair_id'),on='pair_id',how='semi') for k in TRAIN_VIEWS]); model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(training,pairnames),np.array([LABELS[f] for f in training['behavior_family']]),sample_weight=np.full(training.height,1/3))
    joblib.dump({'model':model,'features':pairnames},dest/'pair.joblib')
    data,_,_,_,_=load_sequence(); data=context(data.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
    data=context(data.join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1'),caf); data=context(data.join(cb.drop('table_id'),on=['pair_id','hand_id'],validate='1:1'),cbf)
    ranker=joblib.load(SOURCE/'screen-results/conditional-evidence.joblib'); ranknames=ranker['features']+cbf+['context_'+c for c in cbf]; assert len(ranknames)==542
    eranker=fit_variant(data.join(train.select('pair_id'),on='pair_id',how='semi'),ranknames,PARAMS); joblib.dump({'model':eranker,'features':ranknames},dest/'evidence.joblib')
    contexts=[c for c in raw.columns if c.startswith('v2_')]+SEQ+EQ+caf+cbf; rows=[]; preds=[]; erows=[]; comps={}
    for history,pf in [('full',held),('shortened',views[HELDOUT_VIEW].join(held.select('pair_id'),on='pair_id',how='semi').sort('pair_id'))]:
        p=bundle['model'].predict_proba(matrix(pf,bundle['features']),num_threads=2); control=1-p[:,0]; roles=bundle['model'].classes_[1:][np.argmax(p[:,1:],axis=1)]; y=np.array([LABELS[f] for f in pf['behavior_family']])
        old=pl.read_csv(ROOT/'reports/diverse-next-day/fold0/pair-predictions.csv').filter((pl.col('history')==history)&(pl.col('candidate')=='baseline')).sort('pair_id'); assert pf['pair_id'].to_list()==old['pair_id'].to_list(); np.testing.assert_array_equal(control,old['risk'].to_numpy())
        risk=1-model.predict_proba(matrix(pf,pairnames),num_threads=2)[:,0]; baseline=scorecard(y,control,roles)
        for candidate,v in [('baseline',control),('bucket_risk',risk),('bucket_equal_blend',(control+risk)/2)]:
            m=scorecard(y,v,roles); rows.append({'history':history,'candidate':candidate,**m,'delta':{k:m[k]-baseline[k] for k in m}})
            preds.extend({'pair_id':pid,'history':history,'candidate':candidate,'risk':float(value)} for pid,value in zip(pf['pair_id'],v))
        assigned=pf.select('pair_id').with_columns(pl.Series('predicted_behavior',[{v:k for k,v in LABELS.items()}[int(r)] for r in roles]))
        hf=data.join(held.select('pair_id'),on='pair_id',how='semi')
        if history=='shortened': hf=context(thin(hf,*HELDOUT_VIEW),contexts)
        truth={k[0]:set(g.filter(pl.col('relevance')>0)['hand_id']) for k,g in hf.group_by('pair_id') if g['relevance'].sum()>0}
        hf=family_columns(hf.filter(pl.col('pair_id').is_in(list(truth))).join(assigned,on='pair_id',validate='m:1'),'predicted_behavior').sort('pair_id','hand_id')
        scores={'baseline':ranker['model'].predict(matrix(hf,ranker['features']),num_threads=2),'bucket_evidence':eranker.predict(matrix(hf,ranknames),num_threads=2)}
        olde=pl.read_parquet(ROOT/'reports/diverse-next-day/fold0'/(history+'-baseline-evidence.parquet')); assert olde.select('pair_id','hand_id').equals(hf.select('pair_id','hand_id')); np.testing.assert_array_equal(olde['score'].to_numpy(),scores['baseline'])
        er=[]
        for candidate,score in scores.items():
            er+=records(hf,score,truth,history,candidate)
            hf.select('pair_id','hand_id').with_columns(pl.Series('score',score)).write_parquet(dest/(history+'-'+candidate+'.parquet'))
        erows+=er; comps[history]=compare(er,'bucket_evidence','baseline')
    decisions={}; gate=plan['risk_screen_gate']
    for candidate in plan['risk_recipes']:
        values=[r for r in rows if r['candidate']==candidate]; mean=float(np.mean([r['delta']['pair_behavior'] for r in values])); worst=min(r['delta']['pair_behavior'] for r in values); sparse=float(np.mean([r['delta']['sparse_ap'] for r in values]))
        decisions[candidate]={'initial_nonregression_pass':mean>=gate['mean_pair_behavior_delta_min'] and worst>=gate['worst_pair_behavior_delta_min'] and sparse>=gate['mean_sparse_ap_delta_min'],'mean_pair_behavior_delta':mean,'worst_delta':worst,'mean_sparse_delta':sparse}
    eg=plan['evidence_screen_gate']; worst=min([*comps['full']['by_family_delta'].values(),*comps['shortened']['by_family_delta'].values()])
    decisions['bucket_evidence']={'passed_screen':comps['full']['delta']>=eg['full_delta_min'] and comps['shortened']['delta']>=eg['shortened_delta_min'] and worst>=eg['worst_family_delta_min'],'full_delta':comps['full']['delta'],'shortened_delta':comps['shortened']['delta'],'worst_family_delta':worst}
    pl.DataFrame(preds).write_csv(dest/'pair-predictions.csv'); pl.DataFrame(erows).write_csv(dest/'evidence-pairs.csv'); value={'plan_sha256':sha(OUT/'plan.json'),'decisions':decisions,'pair_metrics':rows,'evidence_comparisons':comps,'exact_incumbent_pair_and_evidence_controls':True,'models':{p.name:sha(p) for p in dest.glob('*.joblib')}}; write(dest/'screen.json',value); print(json.dumps(value),flush=True)


if __name__=='__main__': main()
