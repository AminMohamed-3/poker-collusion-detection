"""Fixed pointwise evidence classifier and reciprocal-rank fusion experiment."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
from datetime import datetime, timezone
import json
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,matrix,sha
from rank_evidence import context
from relative_evidence_experiment import records,summarize
from evidence_capacity_experiment import compare
from evaluate_conditional_confirmation import verified_policy

OUT=ROOT/'reports/conditional-evidence-classifier'
SCREEN=ROOT/'reports/conditional-action-experiment'
CONFIRM=ROOT/'reports/conditional-action-confirmation'


def fusion(frame,baseline,point):
    scores=frame.select('pair_id','hand_id').with_columns(pl.Series('baseline',baseline),pl.Series('point',point),pl.Series('tie',np.random.default_rng(931207).random(frame.height)))
    for name in ['baseline','point']:
        ranked=scores.sort(['pair_id',name,'tie'],descending=[False,True,False]).with_columns(pl.col('hand_id').cum_count().over('pair_id').alias(name+'_rank')).select('pair_id','hand_id',name+'_rank')
        scores=scores.join(ranked,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    return (1/(10+scores['baseline_rank'].to_numpy())+1/(10+scores['point_rank'].to_numpy()))


def main():
    planpath=OUT/'plan.json'; plan=json.loads(planpath.read_text())
    identity={'plan_sha256':sha(planpath),'source_sha256':sha(__file__),
        'conditional_confirmation_sha256':sha(CONFIRM/'results/summary.json')}
    idpath=OUT/'identity.json'
    if idpath.exists(): assert json.loads(idpath.read_text())==identity
    else: idpath.write_text(json.dumps(identity,indent=2)+'\n')
    raw=pl.read_parquet(sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet')))
    names=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]; events=[c for c in names if c.startswith('v2_')]
    labels=pl.read_csv(ROOT/'data/development_labels.csv'); annotated=pl.read_csv(ROOT/'data/development_evidence.csv').select('pair_id','hand_id')
    truths={k[0]:set(g['hand_id']) for k,g in annotated.group_by('pair_id')}
    allrows=[]; results=[]
    for name in plan['jobs']:
        dest=OUT/name; dest.mkdir(exist_ok=True)
        if (dest/'metrics.json').exists():
            results.append(json.loads((dest/'metrics.json').read_text())); allrows.extend(pl.read_csv(dest/'pairs.csv').to_dicts()); continue
        omitted=name.removeprefix('omit_') if name.startswith('omit_') else None
        policyroot=SCREEN/'policy-screen' if name=='fold0' or omitted else CONFIRM/('conditional-'+name+'-v1')/'policy-screen'
        ca,policy=verified_policy(policyroot); caf=policy['features']; heldpools=policy['completed'][-1]['scored_pools']
        data=context(raw.join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0)),events+caf)
        features=names+caf+['context_'+c for c in events+caf]; assert len(features)==285
        data=data.join(labels.select('pair_id','label','label_status','behavior_family'),on='pair_id',validate='m:1').join(annotated.with_columns(pl.lit(1).alias('listed')),on=['pair_id','hand_id'],how='left').with_columns(pl.col('listed').fill_null(0)).sort('pair_id','hand_id')
        training=data.filter(~pl.col('table_id').is_in(heldpools))
        if omitted: training=training.filter(pl.col('behavior_family')!=omitted)
        negative=training.filter(pl.col('label')==0); assert set(negative['label_status'])=={'confirmed_non_target'}
        negative=negative.with_columns(pl.Series('_sample',np.random.default_rng(648291).random(negative.height))).sort('pair_id','_sample').group_by('pair_id',maintain_order=True).head(32).drop('_sample').sort('pair_id','hand_id')
        positive=training.filter((pl.col('label')==1)&(pl.col('listed')==1))
        assert set(positive['label_status'])=={'confirmed_target'}
        training=pl.concat([negative,positive]).sort('pair_id','hand_id')
        held=data.filter(pl.col('table_id').is_in(heldpools)&(pl.col('label')==1))
        if omitted: held=held.filter(pl.col('behavior_family')==omitted)
        held=held.sort('pair_id','hand_id')
        assert not set(training['table_id'])&set(held['table_id'])
        model=lgb.LGBMClassifier(**plan['model']).fit(matrix(training,features),training['listed'].to_numpy())
        prediction=model.predict_proba(matrix(held,features))[:,1]
        baselinepath=(SCREEN/'screen-results' if name=='fold0' else CONFIRM/'results'/name)/'conditional-evidence-scores.parquet'
        saved=pl.read_parquet(baselinepath)
        assert saved.select('pair_id','hand_id').equals(held.select('pair_id','hand_id'))
        baseline=saved['score'].to_numpy(); joined=fusion(held,baseline,prediction)
        joblib.dump({'model':model,'features':features},dest/'model.joblib')
        rows=[]; summaries=[]
        for candidate,score in [('conditional',baseline),('pointwise',prediction),('equal_rrf_k10',joined)]:
            selected=records(held,score,truths,name,candidate); rows.extend(selected)
            summaries.append({'candidate':candidate,**summarize(selected)})
        allrows.extend(rows); pl.DataFrame(rows).write_csv(dest/'pairs.csv')
        held.select('pair_id','hand_id').with_columns(pl.Series('pointwise',prediction),pl.Series('equal_rrf_k10',joined)).write_parquet(dest/'scores.parquet')
        result={'job':name,'training_positive_hands':positive.height,'training_normal_hands':negative.height,'features':len(features),
            'results':summaries,'comparisons':{c:compare(rows,c,baseline='conditional') for c in ['pointwise','equal_rrf_k10']},
            'model_sha256':sha(dest/'model.joblib'),'baseline_score_sha256':sha(baselinepath)}
        # The baseline records must exactly match the earlier frozen ranker.
        prior=pl.read_csv((SCREEN/'screen-results' if name=='fold0' else CONFIRM/'results'/name)/'evidence-pairs.csv').filter(pl.col('candidate')=='conditional').sort('pair_id')
        reproduced=pl.DataFrame(rows).filter(pl.col('candidate')=='conditional').sort('pair_id')
        np.testing.assert_array_equal(prior['map5'].to_numpy(),reproduced['map5'].to_numpy())
        result['baseline_exact']=True
        (dest/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); results.append(result)
        print(json.dumps({'job':name,'comparisons':result['comparisons']}),flush=True)
    frame=pl.DataFrame(allrows); known=frame.filter(pl.col('experiment').is_in(['fold0','fold1','fold2']))
    pooled=known.group_by('pair_id','table_id','family','candidate').agg(pl.col('map5').mean())
    gates={}
    for candidate in ['pointwise','equal_rrf_k10']:
        combined=compare(pooled.to_dicts(),candidate,baseline='conditional')
        ordinary=[r['comparisons'][candidate]['delta'] for r in results if r['job'].startswith('fold')]
        omitted=[r['comparisons'][candidate]['delta'] for r in results if r['job'].startswith('omit_')]
        passed=combined['delta']>=.005 and min(ordinary)>=-.005 and min(combined['by_family_delta'].values())>=-.025 and np.mean(omitted)>=-.005 and min(omitted)>=-.03
        gates[candidate]={'passed':bool(passed),'pooled_comparison':combined,'known_deltas':ordinary,'omitted_deltas':omitted}
    passing=[n for n,g in gates.items() if g['passed']]; selected=max(passing,key=lambda c:gates[c]['pooled_comparison']['delta']) if passing else None
    payload={'completed_at':datetime.now(timezone.utc).isoformat(),'gates':gates,'selected':selected,'all_baselines_exact':True}
    (OUT/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload),flush=True)


if __name__=='__main__': main()
