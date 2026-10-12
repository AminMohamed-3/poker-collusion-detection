"""Frozen additional pool and omitted-family checks of conditional residuals."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import json
from datetime import datetime, timezone
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score
from robust_pair_experiment import ROOT, LABELS, TRAIN_VIEWS, HELDOUT_VIEW, MODEL_PARAMS, thin, aggregate, matrix, assert_disjoint, sha, stable_ap
from tail_pair_experiment import tails
from evaluate_conditional_action_screen import extra_aggregate, metrics
from evidence_capacity_experiment import load_sequence, fit_variant, compare
from rank_evidence import context, family_columns, PARAMS
from equity_evidence_features import FEATURES as EQ
from relative_evidence_experiment import records, summarize

OUT=ROOT/'reports/conditional-action-confirmation'
SCREEN=ROOT/'reports/conditional-action-experiment'
REFERENCE=ROOT/'reports/tabicl-confirmation'


def behavior_metrics(y,risk,predicted):
    # Keep both implementations explicit: official stable ties vs sklearn ties.
    ordinary=[average_precision_score(y==i,risk*(predicted==i)) for i in [1,2,3] if (y==i).any()]
    stable=[stable_ap(y==i,risk*(predicted==i)) for i in [1,2,3] if (y==i).any()]
    return {'behavior_map_present_families':float(np.mean(ordinary)),
            'stable_behavior_map_present_families':float(np.mean(stable)),
            'stable_pair_ap':stable_ap(y>0,risk),
            'weighted_pair_behavior':.7*stable_ap(y>0,risk)+.1*float(np.mean(stable))}


def verified_policy(directory):
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['hand_features_sha256']==sha(directory/'hand-features.parquet')
    for name,receipt in zip(['inner0','inner1','inner2','held'],summary['completed']):
        assert receipt==json.loads((directory/(name+'.json')).read_text())
        assert receipt['feature_sha256']==sha(directory/(name+'.parquet'))
        assert receipt['model_sha256']==sha(directory/(name+'-models.joblib'))
        assert not set(receipt['training_pools']) & set(receipt['scored_pools'])
    return pl.read_parquet(directory/'hand-features.parquet'),summary


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    assert plan['parent_screen_sha256']==sha(SCREEN/'screen-results/metrics.json')
    run=OUT/'results'; run.mkdir(exist_ok=True)
    identity={'source_sha256':sha(__file__),'plan_sha256':sha(OUT/'plan.json'),
        'behavior_recipe':'Fixed augmented 653-feature v2 model, trained only on each outer training partition and excluding omitted targets. Original fold0 saved predictions retained for the screen audit.',
        'metrics':'Frozen gate uses sklearn AP as in the screen. Stable-tie PairAP and BehaviorMAP recorded as diagnostics; missing target families excluded from omitted-family behavior means.',
        'short_history_limit':'Pair-local hand thinning; public style baselines retain observed full phase histories.'}
    if (run/'identity.json').exists(): assert json.loads((run/'identity.json').read_text())==identity
    else: (run/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    labels=pl.read_csv(ROOT/'data/development_labels.csv')
    raw=pl.read_parquet(sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet')))
    names=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]
    events=[c for c in names if c.startswith('v2_')]
    meta=json.loads((ROOT/'reports/tail-pair-experiment/metrics.json').read_text())
    base=meta['base_features']+meta['extra_features']; rolefeatures=meta['base_features']
    full=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').join(tails(raw,events),on=['pair_id','table_id']).join(labels,on='pair_id').sort('pair_id')
    rawviews={k:thin(raw,*k) for k in TRAIN_VIEWS+[HELDOUT_VIEW]}
    baseviews={k:aggregate(v,names).join(tails(v,events),on=['pair_id','table_id']).join(labels,on='pair_id').select(full.columns).sort('pair_id') for k,v in rawviews.items()}
    evidence,current,truths,screenroles,screenpools=load_sequence()
    evidence=context(evidence.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
    evidencebase=current+EQ+['context_'+c for c in EQ]
    assert len(base)==765 and len(rolefeatures)==653 and len(evidencebase)==316
    screenrows=pl.read_csv(SCREEN/'screen-results/pair-predictions.csv')
    screenframe=full.filter(pl.col('table_id').is_in(screenpools)).join(screenroles.select('pair_id','predicted_behavior'),on='pair_id').sort('pair_id')
    screenaudit=[]
    for (history,candidate),g in screenrows.group_by('history','candidate'):
        g=g.sort('pair_id'); assert g['pair_id'].to_list()==screenframe['pair_id'].to_list()
        screenaudit.append({'history':history,'candidate':candidate,**behavior_metrics(
            np.array([LABELS[f] for f in screenframe['behavior_family']]),g['risk'].to_numpy(),np.array([LABELS[f] for f in screenframe['predicted_behavior']]))})
    (run/'screen-behavior-audit.json').write_text(json.dumps(screenaudit,indent=2)+'\n')
    for job in plan['jobs']:
        dest=run/job; dest.mkdir(exist_ok=True)
        if (dest/'metrics.json').exists():
            print(json.dumps({'reused_completed_job':job}),flush=True); continue
        omitted=job.removeprefix('omit_') if job.startswith('omit_') else None
        reference=REFERENCE/job; heldkeys=pl.read_csv(reference/'held-pairs.csv')
        # Omitted job reference lists only evaluated families; use all fold0 pools.
        heldpools=screenpools if omitted else heldkeys['table_id'].unique().to_list()
        directory=SCREEN/'policy-screen' if omitted else OUT/('conditional-'+job+'-v1')/'policy-screen'
        ca,policy=verified_policy(directory); caf=policy['features']
        assert set(policy['completed'][-1]['scored_pools'])==set(heldpools)
        assert ca.select(pl.struct('pair_id','hand_id').n_unique()).item()==ca.height==raw.height
        added=[c+s for s in ['_mean','_std','_max','_q95'] for c in caf]
        combined=raw.select('pair_id','hand_id','table_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(pl.col(caf).fill_null(0))
        expanded=full.join(extra_aggregate(combined,caf),on=['pair_id','table_id']).sort('pair_id')
        views={}
        for k,rv in rawviews.items():
            cv=thin(combined,*k); assert rv.select('pair_id','hand_id').equals(cv.select('pair_id','hand_id'))
            views[k]=baseviews[k].join(extra_aggregate(cv,caf),on=['pair_id','table_id']).select(expanded.columns).sort('pair_id')
        training=pl.concat([expanded.filter(~pl.col('table_id').is_in(heldpools))]+[views[k].filter(~pl.col('table_id').is_in(heldpools)) for k in TRAIN_VIEWS])
        if omitted: training=training.filter(pl.col('behavior_family')!=omitted)
        heldframes={'full':expanded.join(heldkeys.select('pair_id'),on='pair_id',how='semi').sort('pair_id'),
            'shortened':views[HELDOUT_VIEW].join(heldkeys.select('pair_id'),on='pair_id',how='semi').sort('pair_id')}
        assert_disjoint(training,heldframes['full'])
        ytrain=np.array([LABELS[f] for f in training['behavior_family']]); weights=np.full(training.height,1/3)
        role=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(training,rolefeatures),ytrain,sample_weight=weights)
        joblib.dump({'model':role,'features':rolefeatures},dest/'behavior.joblib')
        roles={}
        for history,frame in heldframes.items():
            prob=role.predict_proba(matrix(frame,rolefeatures)); roles[history]=role.classes_[1:][np.argmax(prob[:,1:],axis=1)]
        tab=np.load(reference/'predictions.npz',allow_pickle=False); controls=np.load(reference/'arrays.npz',allow_pickle=False)
        pairresults=[]; pairrows=[]
        for name,features in [('v5_control',base),('conditional',base+added)]:
            model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(training,features),ytrain,sample_weight=weights)
            assert model.classes_[0]==0
            joblib.dump({'model':model,'features':features},dest/(name+'-pair.joblib'))
            for history,frame in heldframes.items():
                assert frame['pair_id'].to_list()==heldkeys['pair_id'].to_list()
                risk=1-model.predict_proba(matrix(frame,features))[:,0]
                if name=='v5_control': np.testing.assert_array_equal(risk,controls['control_'+history])
                y=np.array([LABELS[f] for f in frame['behavior_family']])
                for candidate,values in [(name,risk),('v10_control' if name=='v5_control' else 'conditional_hybrid',.5*(risk+tab[history]))]:
                    pairresults.append({'history':history,'candidate':candidate,**metrics(y,values),**behavior_metrics(y,values,roles[history])})
                    pairrows.extend({'pair_id':p,'history':history,'candidate':candidate,'risk':float(v),'predicted_class':int(c)} for p,v,c in zip(frame['pair_id'],values,roles[history]))
        pl.DataFrame(pairrows).write_csv(dest/'pair-predictions.csv')
        data=context(evidence.join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0)),caf)
        trainhands=data.filter(~pl.col('table_id').is_in(heldpools))
        heldhands=data.filter(pl.col('table_id').is_in(heldpools))
        if omitted:
            trainhands=trainhands.filter(pl.col('behavior_family')!=omitted)
            heldhands=heldhands.filter(pl.col('behavior_family')==omitted)
        inverse={v:k for k,v in LABELS.items()}
        assigned=heldframes['full'].select('pair_id').with_columns(pl.Series('predicted_behavior',[inverse[int(i)] for i in roles['full']]))
        heldhands=heldhands.join(assigned,on='pair_id',validate='m:1').sort('pair_id','hand_id')
        if omitted: assert omitted not in set(assigned['predicted_behavior']) and omitted not in set(trainhands['behavior_family'])
        rankinput=family_columns(heldhands,'predicted_behavior'); erows=[]; eresults=[]
        for name,features in [('equity',evidencebase),('conditional',evidencebase+caf+['context_'+c for c in caf])]:
            model=fit_variant(trainhands,features,PARAMS)
            score=model.predict(matrix(rankinput,features),num_threads=2)
            rows=records(heldhands,score,truths,job,name); erows.extend(rows)
            eresults.append({'candidate':name,**summarize(rows)})
            joblib.dump({'model':model,'features':features},dest/(name+'-evidence.joblib'))
            heldhands.select('pair_id','hand_id').with_columns(pl.Series('score',score)).write_parquet(dest/(name+'-evidence-scores.parquet'))
        pl.DataFrame(erows).write_csv(dest/'evidence-pairs.csv')
        result={'job':job,'omitted':omitted,'pair_results':pairresults,'evidence_results':eresults,
            'evidence_comparison':compare(erows,'conditional',baseline='equity'),'v5_reproduced_exactly':True,
            'policy_features_sha256':policy['hand_features_sha256'],'held_pools':sorted(heldpools),'train_pools':sorted(training['table_id'].unique())}
        (dest/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'job':job,'pair_results':pairresults,'evidence_comparison':result['evidence_comparison']}),flush=True)
    finish(plan,run)


def finish(plan,run):
    results={'fold0':json.loads((SCREEN/'screen-results/metrics.json').read_text())}
    results.update({j:json.loads((run/j/'metrics.json').read_text()) for j in plan['jobs']})
    known=['fold0','fold1','fold2']; omitted=[j for j in plan['jobs'] if j.startswith('omit_')]
    lookup={(job,r['history'],r['candidate']):r for job,v in results.items() for r in v['pair_results']}
    gates={}
    for c in ['conditional','conditional_hybrid']:
        ordinary=[lookup[j,h,c]['pair_ap']-lookup[j,h,'v10_control']['pair_ap'] for j in known for h in ['full','shortened']]
        short=[lookup[j,'shortened',c]['sparse_ap']-lookup[j,'shortened','v10_control']['sparse_ap'] for j in known]
        unseen=[lookup[j,h,c]['sparse_ap']-lookup[j,h,'v10_control']['sparse_ap'] for j in omitted for h in ['full','shortened']]
        passed=min(ordinary)>=-.003 and np.mean(ordinary)>0 and np.mean(short)>=.003 and np.mean(unseen)>=0 and min(unseen)>=-.03
        gates[c]={'passed':bool(passed),'ordinary_deltas':ordinary,'short_sparse_deltas':short,'omitted_sparse_deltas':unseen,
            'known_mean_short_sparse_ap':float(np.mean([lookup[j,'shortened',c]['sparse_ap'] for j in known]))}
    passing=[c for c,v in gates.items() if v['passed']]
    selected=max(passing,key=lambda c:gates[c]['known_mean_short_sparse_ap']) if passing else None
    allrows=pl.concat([pl.read_csv(SCREEN/'screen-results/evidence-pairs.csv')]+[pl.read_csv(run/j/'evidence-pairs.csv') for j in ['fold1','fold2']])
    # Pools overlap: average repeated pair observations before pool bootstrap.
    pooled=allrows.group_by('pair_id','table_id','family','candidate').agg(pl.col('map5').mean())
    comparison=compare(pooled.to_dicts(),'conditional',baseline='equity')
    known_delta=[results[j]['evidence_comparison']['delta'] for j in known]
    unseen_delta=[results[j]['evidence_comparison']['delta'] for j in omitted]
    epass=comparison['delta']>=.005 and min(known_delta)>=-.005 and min(comparison['by_family_delta'].values())>=-.025 and np.mean(unseen_delta)>=-.005 and min(unseen_delta)>=-.03
    payload={'completed_at':datetime.now(timezone.utc).isoformat(),'pair_gates':gates,'selected_pair':selected,
        'evidence_pooled_comparison':comparison,'evidence_known_deltas':known_delta,'evidence_omitted_deltas':unseen_delta,'advance_evidence':bool(epass),
        'all_v5_controls_exact':all(results[j]['v5_reproduced_exactly'] for j in plan['jobs']),
        'limits':'Reused overlapping pool splits; omitted known families are proxies for unseen behavior. No evaluation labels used; no candidate is publicly scored yet.'}
    (run/'summary.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload),flush=True)


if __name__=='__main__': main()
