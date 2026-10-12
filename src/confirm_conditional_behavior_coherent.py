"""Confirm family assignment changes on recomputed reduced observation histories."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import json
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,aggregate,LABELS,stable_ap
from run_conditional_policy_screen import hand_features
from tail_pair_experiment import tails
from evaluate_conditional_action_screen import extra_aggregate
from evaluate_conditional_behavior import family_metrics
from rank_evidence import context,family_columns
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from relative_evidence_experiment import records,summarize
from evidence_capacity_experiment import compare

OUT=ROOT/'reports/conditional-behavior-experiment'
COHERENT=ROOT/'reports/conditional-coherent-history'


def main():
    assert json.loads((OUT/'summary.json').read_text())['ordinary_and_omitted_gate_passed']
    dest=OUT/'coherent'; dest.mkdir(exist_ok=False)
    identity={'source_sha256':sha(__file__),'plan_sha256':sha(OUT/'plan.json'),'ordinary_sha256':sha(OUT/'summary.json'),'coherent_source_sha256':sha(COHERENT/'summary.json')}
    (dest/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    labels=pl.read_csv(ROOT/'data/development_labels.csv'); annotation=pl.read_csv(ROOT/'data/development_evidence.csv').select('pair_id','hand_id')
    metadata=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').select('pair_id','table_id').join(labels,on='pair_id')
    names=[c for c in pl.scan_parquet(ROOT/'outputs_v2/hands_development/batch_000.parquet').collect_schema().names() if c not in ['pair_id','hand_id','table_id']]
    events=[c for c in names if c.startswith('v2_')]
    seq=pl.read_parquet(ROOT/'reports/sequence-evidence-experiment/development.parquet')
    eq=pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet')
    raws={f:pl.read_parquet(sorted((COHERENT/str(f)).glob('*-raw.parquet'))) for f in [.65,.4]}
    actions={f:pl.read_parquet(sorted((COHERENT/str(f)).glob('*-actions.parquet'))) for f in [.65,.4]}
    inverse={v:k for k,v in LABELS.items()}; results=[]; allrows=[]
    for fold in [0,1,2]:
        name=f'fold{fold}'
        policy=ROOT/('reports/conditional-action-experiment/policy-screen' if fold==0 else 'reports/conditional-action-confirmation/conditional-'+name+'-v1/policy-screen')
        modeldir=ROOT/('reports/conditional-action-experiment/screen-results' if fold==0 else 'reports/conditional-action-confirmation/results/'+name)
        reference=joblib.load(policy/'held-models.joblib'); pools=json.loads((policy/'held.json').read_text())['scored_pools']
        pair=joblib.load(modeldir/'conditional-pair.joblib'); ranker=joblib.load(modeldir/'conditional-evidence.joblib')
        for fraction in [.65,.4]:
            history='coherent_'+str(round(fraction*100)); raw=raws[fraction].filter(pl.col('table_id').is_in(pools)); action=actions[fraction].filter(pl.col('table_id').is_in(pools))
            x=action.select(reference['inputs']).to_numpy().astype(np.float32)
            ca=hand_features(action,reference['classifier'].predict_proba(x,num_threads=2),reference['regressor'].predict(x,num_threads=2)); caf=[c for c in ca.columns if c.startswith('ca_')]
            ca=raw.select('pair_id','hand_id','table_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0))
            frame=metadata.filter(pl.col('table_id').is_in(pools)).join(aggregate(raw,names),on=['pair_id','table_id'],how='left',validate='1:1').join(tails(raw,events),on=['pair_id','table_id'],how='left',validate='1:1').join(extra_aggregate(ca,caf),on=['pair_id','table_id'],how='left',validate='1:1').sort('pair_id')
            probability=pair['model'].predict_proba(matrix(frame,pair['features'])); risk=1-probability[:,0]; new=pair['model'].classes_[1:][np.argmax(probability[:,1:],axis=1)]
            original=pl.read_parquet(COHERENT/name/(history+'-pairs.parquet'))
            assert frame['pair_id'].to_list()==original['pair_id'].to_list(); np.testing.assert_array_equal(risk,original['conditional'].to_numpy())
            old=original['predicted_class'].to_numpy(); y=np.array([LABELS[f] for f in frame['behavior_family']]); pairap=stable_ap(y>0,risk)
            data=raw.join(labels.filter(pl.col('label')==1).select('pair_id','behavior_family'),on='pair_id').join(seq,on=['pair_id','hand_id'],validate='1:1').join(eq,on=['pair_id','hand_id'],validate='1:1').join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1')
            observed=annotation.join(data.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi'); truths={k[0]:set(g['hand_id']) for k,g in observed.group_by('pair_id')}
            data=context(data,events+SEQ+EQ+caf).filter(pl.col('pair_id').is_in(list(truths))).sort('pair_id','hand_id')
            rows=[]; metrics=[]
            for candidate,roles in [('original_behavior',old),('conditional_behavior',new)]:
                assigned=frame.select('pair_id').with_columns(pl.Series('predicted_behavior',[inverse[int(i)] for i in roles]))
                held=family_columns(data.join(assigned,on='pair_id',validate='m:1',maintain_order='left'),'predicted_behavior')
                score=ranker['model'].predict(matrix(held,ranker['features']),num_threads=2)
                if candidate=='original_behavior':
                    expected=pl.read_parquet(COHERENT/name/(history+'-conditional-evidence.parquet'))
                    assert expected.select('pair_id','hand_id').equals(held.select('pair_id','hand_id')); np.testing.assert_array_equal(score,expected['score'].to_numpy())
                erows=records(held,score,truths,name+'_'+history,candidate); rows.extend(erows)
                e=summarize(erows); b=family_metrics(y,risk,roles)
                metrics.append({'candidate':candidate,**b,'evidence':e,'composite':.7*pairap+.1*b['behavior_map']+.2*e['map5']})
                held.select('pair_id','hand_id').with_columns(pl.Series('score',score)).write_parquet(dest/(name+'_'+history+'_'+candidate+'.parquet'))
            allrows.extend(rows); by={r['candidate']:r for r in metrics}
            result={'fold':fold,'history':history,'results':metrics,'behavior_delta':by['conditional_behavior']['behavior_map']-by['original_behavior']['behavior_map'],
                'composite_delta':by['conditional_behavior']['composite']-by['original_behavior']['composite'],
                'evidence_comparison':compare(rows,'conditional_behavior',baseline='original_behavior'),'controls_exact':True}
            results.append(result); print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
    pl.DataFrame(allrows).write_csv(dest/'evidence-pairs.csv')
    delta=[r['composite_delta'] for r in results]; passed=np.mean(delta)>0 and min(delta)>=-.003
    summary={'results':results,'composite_deltas':delta,'coherent_gate_passed':bool(passed),'advance_behavior':bool(passed),'all_controls_exact':True}
    (OUT/'coherent-summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__': main()
