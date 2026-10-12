"""Nested event classifiers feed learned hand-score summaries to pair detection."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse
from datetime import datetime,timezone
import json
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,thin,TRAIN_VIEWS,HELDOUT_VIEW,MODEL_PARAMS,LABELS,assert_disjoint
from prepare_reserved_tabicl import data_views
from prepare_attention_reranker import expanded_views
from evaluate_conditional_confirmation import verified_policy
from rank_evidence import context
from evaluate_mil_pairs import scorecard

OUT=ROOT/'reports/event-witness-experiment'
POLICY=ROOT/'reports/conditional-action-experiment/policy-screen'
NESTED=ROOT/'reports/attention-reranker-experiment/nested'
PARAMS=dict(n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=50,colsample_bytree=.8,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=7329,deterministic=True,force_col_wise=True)
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    write(OUT/'plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
        'hypothesis':'A learned detector for annotated gameplay events, summarized across a pair, may supply nonlinear event combinations missed by marginal hand aggregates.',
        'event_features':'Same 285 raw/conditional/within-pair gameplay features as MIL; no identity or family flags.',
        'event_labels':'Annotated target hands versus hands of explicitly confirmed-none pairs. Unannotated target hands are excluded, not negative. Class mass equal; pair mass equal within each class. Scores are not calibrated probabilities of collusion.',
        'event_parameters':PARAMS,'pair_parameters':MODEL_PARAMS,
        'training':'Full histories for event fit, maximum128 normal hands per pair by fixed hash. Nested three-way pool OOF event scores for outer pair training; all normal-reference fits also exclude scored pools. Event summaries recomputed for full/.7/.4 training views and full/.65 held views.',
        'extra_features':'Probability and clipped-logit mean/std/max/q90/q95/q99/top3/top5; frequency score above .5/.9. 18 learned event aggregates.',
        'recipes':['witness','witness_equal_blend'],'recipe_definition':'897 incumbent features plus18 event summaries, unchanged pair-tree parameters. Direct or equal incumbent-risk blend; incumbent behavior/evidence retained.',
        'screen_gate':{'mean_pair_behavior_delta_min':.001,'worst_pair_behavior_delta_min':-.003,'mean_sparse_ap_delta_min':-.01},
        'confirmation_gate':{'known_mean_pair_behavior_delta_min':.001,'known_worst_delta_min':-.003,'known_mean_sparse_delta_min':-.01,'omitted_mean_sparse_delta_min':0.,'omitted_worst_sparse_delta_min':-.05,'stress_mean_pair_behavior_delta_min':0.,'stress_worst_delta_min':-.003},
        'limit':'Four event fits and one pair fit on fold0; only passing recipes progress to other folds/omitted families/coherent history/audit. All validation partitions reused; no untouched claims or public-score selection.',
        'protected':json.loads((ROOT/'reports/mil-pair-experiment/plan.json').read_text())['protected']})
    print(json.dumps({'plan_sha256':sha(OUT/'plan.json')}),flush=True)


def scored_features(frame,scores):
    assert np.isfinite(scores).all()
    p=np.clip(scores,1e-5,1-1e-5); z=np.log(p/(1-p))
    frame=frame.select('pair_id').with_columns(pl.Series('p',p),pl.Series('z',z))
    expr=[]
    for c in ['p','z']:
        expr += [pl.col(c).mean().alias('w_'+c+'_mean'),pl.col(c).std().fill_null(0).alias('w_'+c+'_std'),pl.col(c).max().alias('w_'+c+'_max')]
        expr += [pl.col(c).quantile(q).alias('w_'+c+'_q'+str(int(q*100))) for q in [.9,.95,.99]]
        expr += [pl.col(c).sort(descending=True).head(k).mean().alias('w_'+c+'_top'+str(k)) for k in [3,5]]
    expr += [(pl.col('p')>q).mean().alias('w_above_'+str(int(q*100))) for q in [.5,.9]]
    result=frame.group_by('pair_id').agg(expr).sort('pair_id'); assert result.width==19
    return result


def join_hand(raw,ca,labels,evidence):
    frame=raw.join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1').join(labels.select('pair_id','label','behavior_family'),on='pair_id',validate='m:1')
    frame=frame.join(evidence,on=['pair_id','hand_id'],how='left').with_columns(pl.col('relevance').fill_null(0))
    features=[c for c in raw.columns if c not in ['pair_id','table_id','hand_id']]+[c for c in ca.columns if c.startswith('ca_')]
    contexts=[c for c in features if c.startswith(('v2_','ca_'))]
    features+=['context_'+c for c in contexts]
    assert len(features)==285
    return frame,features,contexts


def fit_event(frame,features,contexts,trainpools,destination):
    frame=context(frame.filter(pl.col('table_id').is_in(sorted(trainpools))),contexts)
    normal=frame.filter(pl.col('label')==0).with_columns(pl.col('hand_id').hash(seed=55193).alias('_sample')).sort('pair_id','_sample','hand_id').group_by('pair_id',maintain_order=True).head(128).drop('_sample')
    positive=frame.filter((pl.col('label')==1)&(pl.col('relevance')==1))
    pieces=[]
    for target,part in [(0,normal),(1,positive)]:
        npairs=part['pair_id'].n_unique()
        pieces.append(part.with_columns(pl.lit(target).alias('event_y'),(1./pl.len().over('pair_id')/npairs).alias('weight')))
    training=pl.concat(pieces).sort('pair_id','hand_id')
    # Preserve mean weight one so regularization retains its declared strength.
    weights=training['weight'].to_numpy(); weights=weights/weights.mean()
    model=lgb.LGBMClassifier(**PARAMS).fit(matrix(training,features),training['event_y'].to_numpy(),sample_weight=weights)
    joblib.dump({'model':model,'features':features},destination)
    write(destination.with_suffix('.json'),{'model_sha256':sha(destination),'training_pools':sorted(trainpools),'normal_rows':normal.height,'annotated_rows':positive.height,'normal_pairs':normal['pair_id'].n_unique(),'target_pairs':positive['pair_id'].n_unique(),'unannotated_targets_used':False})
    return model


def run():
    plan=json.loads((OUT/'plan.json').read_text()); assert sha(__file__)==plan['source_sha256']
    dest=OUT/'fold0'; dest.mkdir(exist_ok=False)
    raw,full,baseviews=data_views(); labels=pl.read_csv(ROOT/'data/development_labels.csv')
    evidence=pl.read_csv(ROOT/'data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('relevance'))
    ca,policy=verified_policy(POLICY); caf=[c for c in ca.columns if c.startswith('ca_')]
    expanded,views=expanded_views(raw,full,baseviews,ca,caf)
    heldpools=set(policy['completed'][-1]['scored_pools']); train=expanded.filter(~pl.col('table_id').is_in(sorted(heldpools))); held=expanded.filter(pl.col('table_id').is_in(sorted(heldpools)))
    assert_disjoint(train,held); trainpools=set(train['table_id']); oof={k:[] for k in ['full','train70','train40']}; audit=[]
    for i in range(3):
        nested=NESTED/f'fold{i}'; receipt=json.loads((nested/'receipt.json').read_text())
        assert all(sha(nested/n)==h for n,h in receipt['files'].items())
        fitpools=set(receipt['ranker_train_pools']); scorepools=set(receipt['ranker_score_pools'])
        assert not fitpools&scorepools and fitpools|scorepools==trainpools and not (fitpools|scorepools)&heldpools
        innerca=pl.read_parquet(nested/'crossfit-features.parquet')
        for n in range(3):
            ref=json.loads((nested/f'reference{n}.json').read_text())['identity']
            assert not set(ref['training_pools'])&(set(ref['scored_pools'])|scorepools|heldpools)
        frame,features,contexts=join_hand(raw.filter(pl.col('table_id').is_in(sorted(trainpools))),innerca,labels,evidence)
        model=fit_event(frame,features,contexts,fitpools,dest/f'inner{i}.joblib')
        for name,key in [('full',None),('train70',TRAIN_VIEWS[0]),('train40',TRAIN_VIEWS[1])]:
            scored=frame.filter(pl.col('table_id').is_in(sorted(scorepools)))
            if key: scored=thin(scored,*key)
            scored=context(scored,contexts).sort('pair_id','hand_id')
            output=scored_features(scored,model.predict_proba(matrix(scored,features),num_threads=2)[:,1]); oof[name].append(output)
        audit.append({'inner':i,'training_pools':len(fitpools),'scored_pools':len(scorepools),'source_receipt_sha256':sha(nested/'receipt.json'),'nested_no_scored_pool_training':True})
        print(json.dumps({'event_fit_complete':i}),flush=True)
    training=[]
    for name,key in [('full',None),('train70',TRAIN_VIEWS[0]),('train40',TRAIN_VIEWS[1])]:
        scoreframe=pl.concat(oof[name]).sort('pair_id'); assert scoreframe['pair_id'].to_list()==train.sort('pair_id')['pair_id'].to_list()
        scoreframe.write_parquet(dest/(name+'-oof.parquet'))
        pf=expanded if key is None else views[key]
        training.append(pf.join(scoreframe,on='pair_id',validate='1:1'))
    bundle=joblib.load(ROOT/'reports/conditional-action-experiment/screen-results/conditional-pair.joblib')
    extra=[c for c in scoreframe.columns if c!='pair_id']; pairfeatures=bundle['features']+extra; augmented=pl.concat(training).sort('pair_id')
    pairmodel=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(augmented,pairfeatures),np.array([LABELS[f] for f in augmented['behavior_family']]),sample_weight=np.full(augmented.height,1/3))
    joblib.dump({'model':pairmodel,'features':pairfeatures},dest/'pair.joblib')
    frame,features,contexts=join_hand(raw,ca,labels,evidence)
    event=fit_event(frame,features,contexts,trainpools,dest/'outer.joblib')
    rows=[]; pred=[]
    for history,key in [('full',None),('shortened',HELDOUT_VIEW)]:
        hf=frame.filter(pl.col('table_id').is_in(sorted(heldpools)))
        if key: hf=thin(hf,*key)
        hf=context(hf,contexts).sort('pair_id','hand_id')
        scores=event.predict_proba(matrix(hf,features),num_threads=2)[:,1]
        hf.select('pair_id','hand_id').with_columns(pl.Series('score',scores)).write_parquet(dest/(history+'-hand-scores.parquet'))
        pf=(expanded if key is None else views[key]).join(scored_features(hf,scores),on='pair_id',validate='1:1').sort('pair_id')
        probabilities=bundle['model'].predict_proba(matrix(pf,bundle['features']),num_threads=2); control=1-probabilities[:,0]
        roles=bundle['model'].classes_[1:][np.argmax(probabilities[:,1:],axis=1)]; y=np.array([LABELS[f] for f in pf['behavior_family']])
        old=pl.read_csv(ROOT/'reports/diverse-next-day/fold0/pair-predictions.csv').filter((pl.col('history')==history)&(pl.col('candidate')=='baseline')).sort('pair_id')
        assert pf['pair_id'].to_list()==old['pair_id'].to_list(); np.testing.assert_array_equal(control,old['risk'].to_numpy())
        risk=1-pairmodel.predict_proba(matrix(pf,pairfeatures),num_threads=2)[:,0]; baseline=scorecard(y,control,roles)
        for candidate,values in [('baseline',control),('witness',risk),('witness_equal_blend',(risk+control)/2)]:
            result=scorecard(y,values,roles); rows.append({'candidate':candidate,'history':history,**result,'delta':{k:result[k]-baseline[k] for k in result}})
            pred.extend({'pair_id':p,'history':history,'candidate':candidate,'risk':float(v),'role':int(r)} for p,v,r in zip(pf['pair_id'],values,roles))
    decisions={}; gate=plan['screen_gate']
    for candidate in plan['recipes']:
        values=[r for r in rows if r['candidate']==candidate]; mean=float(np.mean([r['delta']['pair_behavior'] for r in values])); worst=min(r['delta']['pair_behavior'] for r in values); sparse=float(np.mean([r['delta']['sparse_ap'] for r in values]))
        decisions[candidate]={'passed_screen':mean>=gate['mean_pair_behavior_delta_min'] and worst>=gate['worst_pair_behavior_delta_min'] and sparse>=gate['mean_sparse_ap_delta_min'],'mean_pair_behavior_delta':mean,'worst_pair_behavior_delta':worst,'mean_sparse_ap_delta':sparse}
    pl.DataFrame(pred).write_csv(dest/'pair-predictions.csv')
    write(dest/'audit.json',{'plan_sha256':sha(OUT/'plan.json'),'nested':audit,'models':{p.name:sha(p) for p in dest.glob('*.joblib')},'outer_policy_summary_sha256':sha(POLICY/'summary.json'),'exact_incumbent_reproduction':True})
    result={'plan_sha256':sha(OUT/'plan.json'),'decisions':decisions,'metrics':rows}; write(dest/'screen.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); args=p.parse_args(); freeze() if args.freeze else run()
