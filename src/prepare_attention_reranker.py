"""Nested pool-separated shortlists for a fixed contextual evidence reranker."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
from datetime import datetime,timezone
import gc
import json
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,thin,TRAIN_VIEWS,HELDOUT_VIEW,MODEL_PARAMS,LABELS
from prepare_reserved_tabicl import data_views
from evaluate_conditional_action_screen import extra_aggregate
from evidence_capacity_experiment import load_sequence,fit_variant
from rank_evidence import context,family_columns,PARAMS
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from run_conditional_policy_screen import hand_features

OUT=ROOT/'reports/attention-reranker-experiment'
SOURCE=ROOT/'reports/conditional-action-experiment'


def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n')


def nested_reference(actions,trainpools,scorepools,inputs,parameters,destination):
    metadata=destination.with_suffix('.json'); modelpath=destination.with_suffix('.joblib')
    identity={'plan_sha256':sha(OUT/'plan.json'),'training_pools':sorted(trainpools),'scored_pools':sorted(scorepools)}
    assert not trainpools&scorepools
    if metadata.exists():
        saved=json.loads(metadata.read_text()); assert saved['identity']==identity
        assert sha(destination)==saved['features_sha256'] and sha(modelpath)==saved['model_sha256']
        return pl.read_parquet(destination)
    train=actions.filter((pl.col('label')==0)&pl.col('table_id').is_in(sorted(trainpools)))
    assert set(train['label_status'])=={'confirmed_non_target'}
    train=train.with_columns(pl.Series('sample_order',np.random.default_rng(55193).random(train.height))).sort('pair_id','sample_order').group_by('pair_id',maintain_order=True).head(128)
    if train.height>100000: train=train.sample(n=100000,seed=55193)
    scored=actions.filter(pl.col('table_id').is_in(sorted(scorepools)))
    X=matrix(train,inputs); Z=matrix(scored,inputs)
    classifier=lgb.LGBMClassifier(**parameters).fit(X,train['action_class'].to_numpy())
    regressor=lgb.LGBMRegressor(**parameters).fit(X,train['partner_strength'].to_numpy())
    assert list(classifier.classes_)==[0,1,2,3]
    features=hand_features(scored,classifier.predict_proba(Z,num_threads=2),regressor.predict(Z,num_threads=2))
    features.write_parquet(destination); joblib.dump({'classifier':classifier,'regressor':regressor,'inputs':inputs},modelpath)
    write(metadata,{'identity':identity,'features_sha256':sha(destination),'model_sha256':sha(modelpath),'training_actions':train.height,'scored_actions':scored.height})
    print(json.dumps({'reference':str(destination.relative_to(OUT)),'training_pools':len(trainpools),'score_pools':len(scorepools),'rows':features.height}),flush=True)
    return features


def expanded_views(raw,full,baseviews,ca,caf):
    joined=raw.select('pair_id','hand_id','table_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    expanded=full.join(extra_aggregate(joined,caf),on=['pair_id','table_id']).sort('pair_id')
    views={key:frame.join(extra_aggregate(thin(joined,*key),caf),on=['pair_id','table_id']).select(expanded.columns).sort('pair_id') for key,frame in baseviews.items()}
    return expanded,views


def roles(frame,bundle):
    probability=bundle['model'].predict_proba(matrix(frame,bundle['features']),num_threads=2)
    assert list(bundle['model'].classes_)==[0,1,2,3]
    inverse={v:k for k,v in LABELS.items()}
    return frame.select('pair_id').with_columns(pl.Series('predicted_behavior',[inverse[int(i)] for i in np.argmax(probability[:,1:],axis=1)+1]),
        *[pl.Series('pair_probability_'+str(i),probability[:,i].astype(np.float32)) for i in range(4)])


def shortlists(data,assigned,bundle,features,tag):
    data=family_columns(data.join(assigned,on='pair_id',validate='m:1',maintain_order='left').sort('pair_id','hand_id'),'predicted_behavior')
    score=bundle['model'].predict(matrix(data,bundle['features']),num_threads=2)
    scored=data.with_columns(pl.Series('base_score',score),pl.Series('tie',np.random.default_rng(931207).random(data.height)))
    arrays={key:[] for key in ['x','y','base','mask']}; rows=[]; all_scores=[]
    for key,group in scored.group_by('pair_id',maintain_order=True):
        ordered=group.sort(['base_score','tie'],descending=[True,False]); scores=ordered['base_score'].to_numpy()
        z=(scores-scores.mean())/max(scores.std(ddof=1) if len(scores)>1 else 0,.001)
        ordered=ordered.with_columns(pl.Series('baseline_z',z.astype(np.float32)),pl.Series('baseline_percentile',(1-np.arange(len(scores))/max(len(scores)-1,1)).astype(np.float32)))
        selected=ordered.head(50); length=selected.height
        x=np.zeros((50,len(features)),np.float32); x[:length]=matrix(selected,features)
        y=np.zeros(50,np.float32); y[:length]=selected['relevance'].to_numpy()
        baseline=np.zeros(50,np.float32); baseline[:length]=selected['baseline_z'].to_numpy()
        mask=np.arange(50)<length
        for name,value in [('x',x),('y',y),('base',baseline),('mask',mask)]: arrays[name].append(value)
        rows.append({'pair_id':key[0],'table_id':group['table_id'][0],'family':group['behavior_family'][0],'history':tag,
            'selected_hands':selected['hand_id'].to_list(),'all_hands':ordered['hand_id'].to_list(),
            'truth':group.filter(pl.col('relevance')>0)['hand_id'].to_list(),'selected_annotations':int(y.sum()),'shared_hands':group.height})
        all_scores.append(group.select('pair_id','hand_id','table_id','behavior_family','base_score'))
    return {key:np.stack(values) for key,values in arrays.items()},rows,pl.concat(all_scores).sort('pair_id','hand_id')


def main():
    planpath=OUT/'plan.json'; plan=json.loads(planpath.read_text())
    assert all(sha(ROOT/'src'/name)==digest for name,digest in plan['preparation_sources'].items())
    assert all(sha(ROOT/path)==digest for path,digest in plan['protected'].items())
    work=OUT/'nested'; work.mkdir(exist_ok=True)
    oldplan=json.loads((SOURCE/'policy-plan.json').read_text())
    labels=pl.read_csv(ROOT/'data/development_labels.csv')
    actions=pl.read_parquet(sorted((SOURCE/'action-cache').glob('*.parquet'))).join(labels.select('pair_id','label','label_status'),on='pair_id',validate='m:1').sort('pair_id','hand_id','direction','action_no')
    held=set(plan['held_pools']); outertrain=set(actions['table_id'])-held
    assert len(held)==100 and len(outertrain)==297
    actions=actions.filter(pl.col('table_id').is_in(sorted(outertrain)))
    raw,full,baseviews=data_views()
    evidence,current,_,_,_=load_sequence()
    evidence=context(evidence.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
    oldca=pl.read_parquet(SOURCE/'policy-screen/hand-features.parquet'); caf=[c for c in oldca.columns if c.startswith('ca_')]
    events=[c for c in raw.columns if c.startswith('v2_')]
    pairbundle=joblib.load(SOURCE/'screen-results/conditional-pair.joblib')
    rankbundle=joblib.load(SOURCE/'screen-results/conditional-evidence.joblib')
    rf=rankbundle['features']; features=rf+['pair_probability_'+str(i) for i in range(4)]+['baseline_z','baseline_percentile']
    assert len(rf)==382 and len(features)==388
    oof_arrays=[]; oof_rows=[]
    for index in range(3):
        dest=work/f'fold{index}'; dest.mkdir(exist_ok=True)
        receipt=dest/'receipt.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['plan_sha256']==sha(planpath)
            assert all(sha(dest/name)==digest for name,digest in saved['files'].items())
            oof_arrays.append(dict(np.load(dest/'arrays.npz',allow_pickle=False))); oof_rows+=json.loads((dest/'rows.json').read_text()); continue
        reference=json.loads((SOURCE/'policy-screen'/f'inner{index}.json').read_text())
        innerheld=set(reference['scored_pools']); innertrain=set(reference['training_pools'])
        assert innerheld|innertrain==outertrain and not innerheld&innertrain and not (innerheld|innertrain)&held
        train_order=np.array(sorted(innertrain)); np.random.default_rng(84201+index).shuffle(train_order)
        subfolds=[set(part.tolist()) for part in np.array_split(train_order,3)]
        pieces=[]
        for inner,scorepools in enumerate(subfolds):
            pieces.append(nested_reference(actions,innertrain-scorepools,scorepools,oldplan['inputs'],{**oldplan['parameters'],'n_jobs':2},dest/f'reference{inner}.parquet'))
        original=SOURCE/'policy-screen'/f'inner{index}.parquet'
        assert sha(original)==reference['feature_sha256']
        pieces.append(pl.read_parquet(original))
        ca=pl.concat(pieces).sort('pair_id','hand_id')
        assert set(ca['table_id'])==outertrain and ca.select(pl.struct('pair_id','hand_id').n_unique()).item()==ca.height
        ca.write_parquet(dest/'crossfit-features.parquet')
        expanded,views=expanded_views(raw,full,baseviews,ca,caf)
        training=pl.concat([expanded.filter(pl.col('table_id').is_in(sorted(innertrain)))]+[views[key].filter(pl.col('table_id').is_in(sorted(innertrain))) for key in TRAIN_VIEWS])
        model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(training,pairbundle['features']),np.array([LABELS[f] for f in training['behavior_family']]),sample_weight=np.full(training.height,1/3))
        classifier={'model':model,'features':pairbundle['features']}; joblib.dump(classifier,dest/'pair.joblib')
        data=context(evidence.join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1'),caf)
        trainhands=data.filter(pl.col('table_id').is_in(sorted(innertrain)))
        ranker={'model':fit_variant(trainhands,rf,PARAMS),'features':rf}; joblib.dump(ranker,dest/'ranker.joblib')
        assert not set(trainhands['table_id'])&innerheld
        pieces_arrays=[]; pieces_rows=[]
        for tag,view in [('full',None),('train70',TRAIN_VIEWS[0]),('train40',TRAIN_VIEWS[1])]:
            handframe=data.filter(pl.col('table_id').is_in(sorted(innerheld)))
            pairframe=(expanded if view is None else views[view]).filter(pl.col('table_id').is_in(sorted(innerheld)))
            if view is not None:
                handframe=context(handframe.join(thin(raw,*view).select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi'),events+SEQ+EQ+caf)
            arrays,rows,_=shortlists(handframe,roles(pairframe,classifier),ranker,features,tag)
            pieces_arrays.append(arrays); pieces_rows+=rows
        arrays={key:np.concatenate([part[key] for part in pieces_arrays]) for key in pieces_arrays[0]}
        np.savez_compressed(dest/'arrays.npz',**arrays); write(dest/'rows.json',pieces_rows)
        files={path.name:sha(path) for path in dest.iterdir() if path.is_file()}
        write(receipt,{'plan_sha256':sha(planpath),'files':files,'ranker_train_pools':sorted(innertrain),'ranker_score_pools':sorted(innerheld),'outer_held_excluded':True,'training_reference_crossfit':True})
        oof_arrays.append(arrays); oof_rows+=pieces_rows
        print(json.dumps({'first_stage_fold_complete':index,'queries':len(pieces_rows)}),flush=True); gc.collect()
    train_arrays={key:np.concatenate([part[key] for part in oof_arrays]) for key in oof_arrays[0]}
    assert not {row['table_id'] for row in oof_rows}&held
    assert len({row['pair_id'] for row in oof_rows})==274
    assert len({(row['pair_id'],row['history']) for row in oof_rows})==len(oof_rows)
    # Production-like held inputs use the previously validated outer reference/models.
    expanded,views=expanded_views(raw,full,baseviews,oldca,caf)
    data=context(evidence.join(oldca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1'),caf)
    package=OUT/'gpu-package'; package.mkdir(exist_ok=True)
    exported={'train_'+key:value for key,value in train_arrays.items()}
    write(OUT/'train-rows.json',oof_rows)
    held_rows={}
    for tag,view in [('full',None),('shortened',HELDOUT_VIEW)]:
        handframe=data.filter(pl.col('table_id').is_in(sorted(held)))
        pairframe=(expanded if view is None else views[view]).filter(pl.col('table_id').is_in(sorted(held)))
        if view is not None:
            handframe=context(handframe.join(thin(raw,*view).select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi'),events+SEQ+EQ+caf)
        arrays,rows,scoreframe=shortlists(handframe,roles(pairframe,pairbundle),rankbundle,features,tag)
        if tag=='full':
            expected=pl.read_parquet(ROOT/'reports/conditional-behavior-experiment/fold0/conditional_behavior-scores.parquet')
            assert scoreframe.select('pair_id','hand_id').equals(expected.select('pair_id','hand_id'))
            np.testing.assert_array_equal(scoreframe['base_score'].to_numpy(),expected['score'].to_numpy())
        held_rows[tag]=rows; scoreframe.write_parquet(OUT/(tag+'-baseline-scores.parquet'))
        # Held annotations are kept locally and never sent to the training worker.
        for key in ['x','base','mask']: exported[tag+'_'+key]=arrays[key]
    write(OUT/'held-rows.json',held_rows); write(OUT/'features.json',features)
    np.savez_compressed(package/'arrays.npz',**exported)
    write(OUT/'preparation.json',{'plan_sha256':sha(planpath),'arrays_sha256':sha(package/'arrays.npz'),'training_queries':len(oof_rows),
        'training_pairs':274,'features':len(features),'shortlist':50,'training_queries_without_shortlist_positive':int(np.sum(train_arrays['y'].sum(axis=1)==0)),
        'held_queries':{tag:len(rows) for tag,rows in held_rows.items()},'full_first_stage_scores_exact':True,
        'held_labels_exported':False,'nested_references_exclude_first_stage_score_pools':True,
        'held_rows_sha256':sha(OUT/'held-rows.json'),'train_rows_sha256':sha(OUT/'train-rows.json')})
    print(json.dumps({'prepared':True,'arrays_sha256':sha(package/'arrays.npz'),'queries':len(oof_rows),'features':len(features)}),flush=True)


if __name__=='__main__': main()
