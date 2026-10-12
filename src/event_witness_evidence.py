"""Explicit new evidence endpoint, frozen before examining held hand ranks."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse
from datetime import datetime,timezone
import json
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha
from evidence_capacity_experiment import compare
from relative_evidence_experiment import records

OUT=ROOT/'reports/event-witness-experiment'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def freeze():
    path=OUT/'evidence-plan.json'; assert not path.exists()
    write(path,{'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
        'adaptation':'After the distinct pair-risk endpoint failed, test whether annotations-versus-confirmed-normal hand training helps rank evidence. No held hand-rank results inspected before this plan. This endpoint is adaptively proposed and shares all reused partitions.',
        'recipes':['event_direct','event_rrf'],'recipe':'Use existing outer witness hand score directly, or equal incumbent/witness reciprocal rank fusion, average ties, offset10. Risk and behavior unchanged.',
        'screen_gate':{'full_pooled_map_delta_min':.005,'shortened_map_delta_min':-.005,'worst_family_delta_min':-.02},
        'confirmation':'Require known pooled full/short gains >=.005; every family delta>=-.02; omitted mean>=0,worst>=-.025; coherent/audit mean>=0,worst>=-.02. No subgroup rescue.',
        'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'fold0/outer.joblib',OUT/'fold0/full-hand-scores.parquet',OUT/'fold0/shortened-hand-scores.parquet',ROOT/'reports/diverse-next-day/fold0/full-baseline-evidence.parquet',ROOT/'reports/diverse-next-day/fold0/shortened-baseline-evidence.parquet']}})
    print(json.dumps({'evidence_plan_sha256':sha(path)}))


def run():
    plan=json.loads((OUT/'evidence-plan.json').read_text())
    # Execution repair: table_id belongs to gameplay pair metadata, not labels.
    if plan['source_sha256']!=sha(__file__):
        repair=OUT/'evidence-schema-repair.json'
        value={'original_source_sha256':plan['source_sha256'],'repaired_source_sha256':sha(__file__),'reason':'Before any evidence scores were evaluated, join table_id from existing pairs_development.parquet; development_labels has no table_id. No model, ranking recipe or gate changed.'}
        if repair.exists(): assert json.loads(repair.read_text())==value
        else: write(repair,value)
    assert all(sha(ROOT/p)==h for p,h in plan['input_sha256'].items())
    assert not (OUT/'evidence-screen.json').exists()
    labels=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','behavior_family').join(pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').select('pair_id','table_id'),on='pair_id',validate='1:1')
    evidence=pl.read_csv(ROOT/'data/development_evidence.csv').select('pair_id','hand_id')
    allrows=[]; comparisons={}
    for history in ['full','shortened']:
        baseline=pl.read_parquet(ROOT/'reports/diverse-next-day/fold0'/(history+'-baseline-evidence.parquet')).rename({'score':'baseline'})
        frame=baseline.join(pl.read_parquet(OUT/'fold0'/(history+'-hand-scores.parquet')).rename({'score':'event'}),on=['pair_id','hand_id'],validate='1:1').join(labels,on='pair_id',validate='m:1').sort('pair_id','hand_id')
        truth=evidence.join(frame.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi')
        truths={k[0]:set(g['hand_id']) for k,g in truth.group_by('pair_id')}
        frame=frame.with_columns((1/(10+pl.col('baseline').rank('average',descending=True).over('pair_id'))+1/(10+pl.col('event').rank('average',descending=True).over('pair_id'))).alias('rrf'))
        rows=[]
        for name,column in [('baseline','baseline'),('event_direct','event'),('event_rrf','rrf')]: rows+=records(frame,frame[column].to_numpy(),truths,history,name)
        allrows+=rows; comparisons[history]={name:compare(rows,name,'baseline') for name in plan['recipes']}
    decisions={}; gate=plan['screen_gate']
    for name in plan['recipes']:
        full=comparisons['full'][name]; short=comparisons['shortened'][name]
        worst=min([*full['by_family_delta'].values(),*short['by_family_delta'].values()])
        decisions[name]={'passed_screen':full['delta']>=gate['full_pooled_map_delta_min'] and short['delta']>=gate['shortened_map_delta_min'] and worst>=gate['worst_family_delta_min'],'full_delta':full['delta'],'shortened_delta':short['delta'],'worst_family_delta':worst}
    pl.DataFrame(allrows).write_csv(OUT/'evidence-pairs.csv')
    result={'plan_sha256':sha(OUT/'evidence-plan.json'),'decisions':decisions,'comparisons':comparisons}; write(OUT/'evidence-screen.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); args=p.parse_args(); freeze() if args.freeze else run()
