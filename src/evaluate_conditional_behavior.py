"""Fixed behavior assignments from existing out-of-pool conditional classifiers."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import json
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,LABELS,stable_ap
from tail_pair_experiment import tails
from evaluate_conditional_action_screen import extra_aggregate
from evidence_capacity_experiment import load_sequence,compare
from relative_evidence_experiment import records,summarize
from rank_evidence import context,family_columns
from equity_evidence_features import FEATURES as EQ

OUT=ROOT/'reports/conditional-behavior-experiment'
SCREEN=ROOT/'reports/conditional-action-experiment'
CONFIRM=ROOT/'reports/conditional-action-confirmation'


def family_metrics(truth,risk,assignment):
    by_family={family:stable_ap(truth==value,risk*(assignment==value)) for family,value in LABELS.items() if value>0 and (truth==value).any()}
    return {'behavior_map':float(np.mean(list(by_family.values()))),'by_family_ap':by_family}


def main():
    plan=json.loads((OUT/'plan.json').read_text()); identity={'source_sha256':sha(__file__),'plan_sha256':sha(OUT/'plan.json')}
    idpath=OUT/'identity.json'
    if idpath.exists(): assert json.loads(idpath.read_text())==identity
    else: idpath.write_text(json.dumps(identity,indent=2)+'\n')
    raw=pl.read_parquet(sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet')))
    events=[c for c in raw.columns if c.startswith('v2_')]; labels=pl.read_csv(ROOT/'data/development_labels.csv')
    full=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').join(tails(raw,events),on=['pair_id','table_id']).join(labels,on='pair_id').sort('pair_id')
    evidence,_,truths,screenroles,_=load_sequence()
    evidence=context(evidence.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
    inverse={v:k for k,v in LABELS.items()}; allrows=[]; results=[]
    for job in plan['ordinary_jobs']+plan['omitted_jobs']:
        dest=OUT/job; dest.mkdir(exist_ok=True)
        if (dest/'metrics.json').exists():
            results.append(json.loads((dest/'metrics.json').read_text())); allrows.extend(pl.read_csv(dest/'evidence-pairs.csv').to_dicts()); continue
        omitted=job.startswith('omit_')
        policy=SCREEN/'policy-screen' if job=='fold0' or omitted else CONFIRM/('conditional-'+job+'-v1/policy-screen')
        modeldir=SCREEN/'screen-results' if job=='fold0' else CONFIRM/'results'/job
        reference=ROOT/'reports/tabicl-screen' if job=='fold0' else ROOT/'reports/tabicl-confirmation'/job
        heldkeys=pl.read_csv(reference/'held-pairs.csv')
        ca=pl.read_parquet(policy/'hand-features.parquet'); caf=[c for c in ca.columns if c.startswith('ca_')]
        joined=raw.select('pair_id','hand_id','table_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(pl.col(caf).fill_null(0))
        frame=full.join(extra_aggregate(joined,caf),on=['pair_id','table_id'],validate='1:1').join(heldkeys.select('pair_id'),on='pair_id',how='semi').sort('pair_id')
        bundle=joblib.load(modeldir/'conditional-pair.joblib'); probability=bundle['model'].predict_proba(matrix(frame,bundle['features']))
        risk=1-probability[:,0]; new=bundle['model'].classes_[1:][np.argmax(probability[:,1:],axis=1)]
        oldpairs=pl.read_csv(modeldir/'pair-predictions.csv').filter((pl.col('candidate')=='conditional')&(pl.col('history')=='full')).sort('pair_id')
        assert frame['pair_id'].to_list()==oldpairs['pair_id'].to_list(); np.testing.assert_array_equal(risk,oldpairs['risk'].to_numpy())
        old=np.array([LABELS[f] for f in frame.select('pair_id').join(screenroles.select('pair_id','predicted_behavior'),on='pair_id',maintain_order='left')['predicted_behavior']]) if job=='fold0' else oldpairs['predicted_class'].to_numpy()
        y=np.array([LABELS[f] for f in frame['behavior_family']]); pairap=stable_ap(y>0,risk)
        data=context(evidence.join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0)),caf).join(heldkeys.select('pair_id'),on='pair_id',how='semi').sort('pair_id','hand_id')
        ranker=joblib.load(modeldir/'conditional-evidence.joblib'); rows=[]; metrics=[]
        for candidate,roles in [('original_behavior',old),('conditional_behavior',new)]:
            assigned=frame.select('pair_id').with_columns(pl.Series('predicted_behavior',[inverse[int(i)] for i in roles]))
            held=data.join(assigned,on='pair_id',validate='m:1',maintain_order='left')
            score=ranker['model'].predict(matrix(family_columns(held,'predicted_behavior'),ranker['features']),num_threads=2)
            if candidate=='original_behavior':
                expected=pl.read_parquet(modeldir/'conditional-evidence-scores.parquet')
                assert expected.select('pair_id','hand_id').equals(held.select('pair_id','hand_id')); np.testing.assert_array_equal(score,expected['score'].to_numpy())
            erows=records(held,score,truths,job,candidate); rows.extend(erows)
            e=summarize(erows); b=family_metrics(y,risk,roles)
            metrics.append({'candidate':candidate,'pair_ap':pairap,**b,'evidence':e,'composite':.7*pairap+.1*b['behavior_map']+.2*e['map5']})
            held.select('pair_id','hand_id').with_columns(pl.Series('score',score)).write_parquet(dest/(candidate+'-scores.parquet'))
        frame.select('pair_id','behavior_family').with_columns(pl.Series('risk',risk),pl.Series('original_class',old),pl.Series('conditional_class',new)).write_csv(dest/'pair-predictions.csv')
        pl.DataFrame(rows).write_csv(dest/'evidence-pairs.csv'); allrows.extend(rows)
        by={r['candidate']:r for r in metrics}
        result={'job':job,'results':metrics,'behavior_delta':by['conditional_behavior']['behavior_map']-by['original_behavior']['behavior_map'],
            'composite_delta':by['conditional_behavior']['composite']-by['original_behavior']['composite'],
            'family_ap_deltas':{f:by['conditional_behavior']['by_family_ap'][f]-v for f,v in by['original_behavior']['by_family_ap'].items()},
            'evidence_comparison':compare(rows,'conditional_behavior',baseline='original_behavior'),
            'changed_assignments':int(np.sum(old!=new)),'pair_risk_and_original_evidence_exact':True}
        (dest/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
    known=[r for r in results if r['job'] in plan['ordinary_jobs']]; omitted=[r for r in results if r['job'] in plan['omitted_jobs']]
    behavior=[r['behavior_delta'] for r in known]; composite=[r['composite_delta'] for r in known]; family=[d for r in known for d in r['family_ap_deltas'].values()]
    evidence=[r['evidence_comparison']['delta'] for r in known]; unseen=[r['evidence_comparison']['delta'] for r in omitted]
    passed=np.mean(behavior)>=.005 and min(family)>=-.025 and np.mean(composite)>=.001 and min(composite)>=-.003 and np.mean(evidence)>=-.005 and min(evidence)>=-.02 and np.mean(unseen)>=-.005 and min(unseen)>=-.03
    payload={'behavior_deltas':behavior,'composite_deltas':composite,'family_deltas':family,'evidence_deltas':evidence,'omitted_evidence_deltas':unseen,
        'ordinary_and_omitted_gate_passed':bool(passed),'coherent_confirmation_required_before_evaluation':bool(passed),'all_controls_exact':True}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload),flush=True)


if __name__=='__main__': main()
