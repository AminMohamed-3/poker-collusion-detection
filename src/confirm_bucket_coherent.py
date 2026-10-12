"""Six coherent-history controls and final joint admission for one fixed risk blend."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import json
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,aggregate,LABELS
from diverse_next_day import locations
from run_conditional_policy_screen import hand_features
from conditional_bucket_features import bucket_features
from tail_pair_experiment import tails
from evaluate_conditional_action_screen import extra_aggregate
from evaluate_mil_pairs import scorecard

OUT=ROOT/'reports/conditional-bucket-experiment'
COHERENT=ROOT/'reports/conditional-coherent-history'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    plan=json.loads((OUT/'confirmation-plan.json').read_text()); assert all(sha(ROOT/'src'/p)==h for p,h in plan['sources'].items())
    for job in ['fold1','fold2','audit']:
        r=json.loads((OUT/'confirmation'/job/'metrics.json').read_text()); assert sha(OUT/'confirmation'/job/'pair.joblib')==r['model_sha256'] and r['all_controls_exact']
    dest=OUT/'coherent'; dest.mkdir(exist_ok=False); inputs={}
    for fraction in [.65,.4]:
        for path in sorted((COHERENT/str(fraction)).glob('batch_*.json')):
            receipt=json.loads(path.read_text())
            for name,h in receipt['files'].items(): assert sha(path.parent/name)==h; inputs[str((path.parent/name).relative_to(ROOT))]=h
    labels=pl.read_csv(ROOT/'data/development_labels.csv'); metadata=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').select('pair_id','table_id').join(labels,on='pair_id')
    names=[c for c in pl.scan_parquet(ROOT/'outputs_v2/hands_development/batch_000.parquet').collect_schema().names() if c not in ['pair_id','hand_id','table_id']]; events=[c for c in names if c.startswith('v2_')]
    raw={f:pl.read_parquet(sorted((COHERENT/str(f)).glob('*-raw.parquet'))) for f in [.65,.4]}; actions={f:pl.read_parquet(sorted((COHERENT/str(f)).glob('*-actions.parquet'))) for f in [.65,.4]}
    rows=[]
    for job in ['fold0','fold1','fold2']:
        heldpath,policy,modeldir,_=locations(job); keys=pl.read_csv(heldpath); pools=keys['table_id'].unique().to_list()
        ref=joblib.load(policy/'held-models.joblib'); base=joblib.load(modeldir/'conditional-pair.joblib')
        if job=='fold0': bucketpath=OUT/'policy-screen/held.joblib'; modelpath=OUT/'screen/pair.joblib'
        else: bucketpath=OUT/'confirmation'/job/'policy/held.joblib'; modelpath=OUT/'confirmation'/job/'pair.joblib'
        reference=joblib.load(bucketpath); bundle=joblib.load(modelpath); inputs[str(bucketpath.relative_to(ROOT))]=sha(bucketpath); inputs[str(modelpath.relative_to(ROOT))]=sha(modelpath)
        for fraction in [.65,.4]:
            history='coherent_'+str(round(fraction*100)); r=raw[fraction].filter(pl.col('table_id').is_in(pools)); a=actions[fraction].filter(pl.col('table_id').is_in(pools)); x=matrix(a,ref['inputs'])
            ap=ref['classifier'].predict_proba(x,num_threads=2); ca=hand_features(a,ap,ref['regressor'].predict(x,num_threads=2)); cb=bucket_features(a,ap,reference['model'].predict_proba(x,num_threads=2)); caf=[c for c in ca.columns if c.startswith('ca_')]; cbf=[c for c in cb.columns if c.startswith('cb_')]
            ca=r.select('pair_id','hand_id','table_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0))
            cb=r.select('pair_id','hand_id','table_id').join(cb.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(cbf).fill_null(0))
            pf=metadata.filter(pl.col('table_id').is_in(pools)).join(aggregate(r,names),on=['pair_id','table_id'],how='left',validate='1:1').join(tails(r,events),on=['pair_id','table_id'],how='left',validate='1:1').join(extra_aggregate(ca,caf),on=['pair_id','table_id'],how='left',validate='1:1').join(extra_aggregate(cb,cbf),on=['pair_id','table_id'],how='left',validate='1:1').sort('pair_id')
            p=base['model'].predict_proba(matrix(pf,base['features']),num_threads=2); control=1-p[:,0]; roles=base['model'].classes_[1:][np.argmax(p[:,1:],axis=1)]; y=np.array([LABELS[f] for f in pf['behavior_family']]); baseline=scorecard(y,control,roles)
            assert pf['pair_id'].to_list()==keys['pair_id'].to_list(); np.testing.assert_array_equal(control,pl.read_parquet(COHERENT/job/(history+'-pairs.parquet'))['conditional'].to_numpy())
            risk=.5*(control+1-bundle['model'].predict_proba(matrix(pf,bundle['features']),num_threads=2)[:,0]); result=scorecard(y,risk,roles); rows.append({'job':job,'history':history,'candidate':'bucket_equal_blend',**result,'delta':{k:result[k]-baseline[k] for k in result}})
            pf.select('pair_id').with_columns(pl.Series('baseline',control),pl.Series('bucket_equal_blend',risk)).write_parquet(dest/(job+'-'+history+'.parquet')); print(json.dumps({'completed':job+'-'+history,'delta':rows[-1]['delta']}),flush=True)
    write(dest/'metrics.json',{'plan_sha256':sha(OUT/'confirmation-plan.json'),'metrics':rows,'all_controls_exact':True,'inputs':inputs})
    screen=json.loads((OUT/'screen/screen.json').read_text()); known=[r for r in screen['pair_metrics'] if r['candidate']=='bucket_equal_blend']
    for job in ['fold1','fold2']: known+=json.loads((OUT/'confirmation'/job/'metrics.json').read_text())['metrics']
    stress=rows+json.loads((OUT/'confirmation/audit/metrics.json').read_text())['metrics']; omitted=json.loads((OUT/'omitted-results.json').read_text())['decisions']['bucket_equal_blend']; gate=plan['gates']
    values={'known_mean_delta':float(np.mean([r['delta']['pair_behavior'] for r in known])),'known_worst_delta':min(r['delta']['pair_behavior'] for r in known),'known_mean_sparse_delta':float(np.mean([r['delta']['sparse_ap'] for r in known])),'stress_mean_delta':float(np.mean([r['delta']['pair_behavior'] for r in stress])),'stress_worst_delta':min(r['delta']['pair_behavior'] for r in stress),'omitted':omitted}
    passed=omitted['passed_omitted_gate'] and all(values[k]>=gate[k+'_min'] for k in ['known_mean_delta','known_worst_delta','known_mean_sparse_delta','stress_mean_delta','stress_worst_delta'])
    original=json.loads((OUT/'plan.json').read_text()); assert all(sha(ROOT/p)==h for p,h in original['protected'].items())
    result={'plan_sha256':sha(OUT/'plan.json'),'confirmation_plan_sha256':sha(OUT/'confirmation-plan.json'),'candidate':'bucket_equal_blend','admitted':passed,'metrics':values,'known_comparisons':len(known),'stress_comparisons':len(stress),'all_controls_exact':True,'no_untouched_holdout':True,'evidence_endpoint_rejected':True}; write(OUT/'admission.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
