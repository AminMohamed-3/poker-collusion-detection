"""Evaluate frozen portfolio models on fully rebuilt shorter observation histories."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import json
import joblib
import numpy as np
import polars as pl
from diverse_next_day import OUT,write,locations,SEEDS
from robust_pair_experiment import ROOT,sha,matrix,aggregate,LABELS,stable_ap
from run_conditional_policy_screen import hand_features
from tail_pair_experiment import tails
from evaluate_conditional_action_screen import extra_aggregate
from evaluate_conditional_behavior import family_metrics
from rank_evidence import context,family_columns
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from relative_evidence_experiment import records
from evidence_capacity_experiment import compare

COHERENT=ROOT/'reports/conditional-coherent-history'
def main():
    dest=OUT/'coherent'; dest.mkdir(exist_ok=False)
    write(dest/'identity.json',{'source_sha256':sha(__file__),'parent_plan_sha256':sha(OUT/'plan.json'),'binary_plan_sha256':sha(OUT/'binary-plan.json')})
    labels=pl.read_csv(ROOT/'data/development_labels.csv'); annotation=pl.read_csv(ROOT/'data/development_evidence.csv').select('pair_id','hand_id')
    metadata=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').select('pair_id','table_id').join(labels,on='pair_id')
    names=[c for c in pl.scan_parquet(ROOT/'outputs_v2/hands_development/batch_000.parquet').collect_schema().names() if c not in ['pair_id','hand_id','table_id']]; events=[c for c in names if c.startswith('v2_')]
    seq=pl.read_parquet(ROOT/'reports/sequence-evidence-experiment/development.parquet'); eq=pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet')
    raw={f:pl.read_parquet(sorted((COHERENT/str(f)).glob('*-raw.parquet'))) for f in [.65,.4]}
    actions={f:pl.read_parquet(sorted((COHERENT/str(f)).glob('*-actions.parquet'))) for f in [.65,.4]}
    pairrows=[]; erows=[]; comparisons={}
    for job in ['fold0','fold1','fold2']:
        heldpath,policy,modeldir,_=locations(job); keys=pl.read_csv(heldpath); pools=sorted(keys['table_id'].unique())
        reference=joblib.load(policy/'held-models.joblib'); base=joblib.load(modeldir/'conditional-pair.joblib'); ranker=joblib.load(modeldir/'conditional-evidence.joblib')
        pbag=[base]+[joblib.load(OUT/job/f'pair-seed-{s}.joblib') for s in SEEDS[1:]]; rbag=[ranker]+[joblib.load(OUT/job/f'evidence-seed-{s}.joblib') for s in SEEDS[1:]]
        hist=joblib.load(OUT/job/'pair-history.joblib'); binary=joblib.load(OUT/job/'pair-binary.joblib'); rhist=joblib.load(OUT/job/'evidence-history.joblib')
        tab=np.load(COHERENT/'gpu-returned/results'/(job+'.npz'))
        for fraction in [.65,.4]:
            history='coherent_'+str(round(fraction*100)); tag=job+'_'+history; r=raw[fraction].filter(pl.col('table_id').is_in(pools)); a=actions[fraction].filter(pl.col('table_id').is_in(pools))
            x=matrix(a,reference['inputs']); ca=hand_features(a,reference['classifier'].predict_proba(x,num_threads=2),reference['regressor'].predict(x,num_threads=2)); caf=[c for c in ca.columns if c.startswith('ca_')]
            ca=r.select('pair_id','hand_id','table_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0))
            frame=metadata.filter(pl.col('table_id').is_in(pools)).join(aggregate(r,names),on=['pair_id','table_id'],how='left',validate='1:1').join(tails(r,events),on=['pair_id','table_id'],how='left',validate='1:1').join(extra_aggregate(ca,caf),on=['pair_id','table_id'],how='left',validate='1:1').sort('pair_id')
            x=matrix(frame,base['features']); prob=base['model'].predict_proba(x); incumbent=1-prob[:,0]; roles=base['model'].classes_[1:][np.argmax(prob[:,1:],axis=1)]
            assert frame['pair_id'].to_list()==keys['pair_id'].to_list(); np.testing.assert_array_equal(incumbent,pl.read_parquet(COHERENT/job/(history+'-pairs.parquet'))['conditional'].to_numpy())
            risk={'baseline':incumbent,'hybrid':.5*(incumbent+tab[history]),'seed_risk':np.mean([1-b['model'].predict_proba(x)[:,0] for b in pbag],axis=0),'history_risk':1-hist['model'].predict_proba(x)[:,0],'binary':binary['model'].predict_proba(x)[:,1]}
            risk['binary_mixed']=.5*(incumbent+risk['binary']); risk['mixed']=np.mean([risk[k] for k in ['hybrid','seed_risk','history_risk']],axis=0)
            y=np.array([LABELS[f] for f in frame['behavior_family']]); weights=np.where(y>0,.01/np.mean(y>0),.99/np.mean(y==0))
            for candidate,values in risk.items():
                ap=stable_ap(y>0,values); bm=family_metrics(y,values,roles)['behavior_map']
                pairrows.append({'job':job,'history':history,'candidate':candidate,'pair_ap':ap,'sparse_ap':stable_ap(y>0,values,weights),'behavior_map':bm,'pair_behavior':.7*ap+.1*bm})
            frame.select('pair_id').with_columns(*[pl.Series(k,v) for k,v in risk.items()]).write_parquet(dest/(tag+'-pair-predictions.parquet'))
            data=r.join(labels.filter(pl.col('label')==1).select('pair_id','behavior_family'),on='pair_id').join(seq,on=['pair_id','hand_id'],validate='1:1').join(eq,on=['pair_id','hand_id'],validate='1:1').join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1')
            observed=annotation.join(data.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi'); truths={k[0]:set(g['hand_id']) for k,g in observed.group_by('pair_id')}
            assigned=frame.select('pair_id').with_columns(pl.Series('predicted_behavior',[{v:k for k,v in LABELS.items()}[int(i)] for i in roles]))
            data=family_columns(context(data,events+SEQ+EQ+caf).filter(pl.col('pair_id').is_in(list(truths))).join(assigned,on='pair_id',validate='m:1'),'predicted_behavior').sort('pair_id','hand_id'); x=matrix(data,ranker['features'])
            scores={'baseline':ranker['model'].predict(x,num_threads=2),'seed_evidence':np.mean([b['model'].predict(x,num_threads=2) for b in rbag],axis=0),'history_evidence':rhist['model'].predict(x,num_threads=2)}
            prior=pl.read_parquet(ROOT/'reports/conditional-behavior-experiment/coherent'/(tag+'_conditional_behavior.parquet')); assert prior.select('pair_id','hand_id').equals(data.select('pair_id','hand_id')); np.testing.assert_array_equal(prior['score'].to_numpy(),scores['baseline'])
            rows=[]
            for candidate,values in scores.items():
                rows.extend(records(data,values,truths,tag,candidate)); data.select('pair_id','hand_id').with_columns(pl.Series('score',values)).write_parquet(dest/(tag+'-'+candidate+'-evidence.parquet'))
            erows.extend(rows); comparisons[tag]={c:compare(rows,c,baseline='baseline') for c in ['seed_evidence','history_evidence']}
            print(json.dumps({'completed':tag,'evidence_delta':{c:v['delta'] for c,v in comparisons[tag].items()}}),flush=True)
    pl.DataFrame(erows).write_csv(dest/'evidence-pairs.csv'); write(dest/'metrics.json',{'pair_results':pairrows,'evidence_comparisons':comparisons,'all_incumbent_controls_exact':True})

if __name__=='__main__': main()
