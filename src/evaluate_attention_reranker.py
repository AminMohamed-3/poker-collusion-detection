"""Score only the three predeclared ensembles after all fixed fits finish."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from datetime import datetime,timezone
import json
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha
from evidence_context_experiment import ap5
from evidence_capacity_experiment import compare
from relative_evidence_experiment import summarize

OUT=ROOT/'reports/attention-reranker-experiment'


def main():
    plan=json.loads((OUT/'plan.json').read_text()); worker=OUT/'gpu-package'; result=worker/'results'
    finished=json.loads((result/'summary.json').read_text())
    assert finished['fits']==6 and finished['held_labels_used'] is False
    assert finished['plan_sha256']==sha(worker/'worker-plan.json')
    preparation=json.loads((OUT/'preparation.json').read_text())
    assert preparation['plan_sha256']==sha(OUT/'plan.json') and preparation['held_labels_exported'] is False
    assert preparation['arrays_sha256']==sha(worker/'arrays.npz')
    assert preparation['held_rows_sha256']==sha(OUT/'held-rows.json')
    assert all(sha(ROOT/path)==digest for path,digest in plan['protected'].items())
    held=json.loads((OUT/'held-rows.json').read_text())
    by={}
    for receipt in finished['receipts']:
        assert receipt['plan_sha256']==finished['plan_sha256'] and receipt['reload_exact']
        assert all(sha(result/name)==digest for name,digest in receipt['files'].items())
        tag=f"{receipt['objective']}-{receipt['seed']}"
        by[receipt['objective'],receipt['seed']]=np.load(result/(tag+'.npz'),allow_pickle=False)
    assert set(by)=={(objective,seed) for objective in plan['optimization']['objectives'] for seed in plan['optimization']['seeds']}
    rows=[]; summary=[]; comparisons=[]
    expected=pl.read_csv(ROOT/'reports/conditional-behavior-experiment/fold0/evidence-pairs.csv').filter(pl.col('candidate')=='conditional_behavior')
    expected=dict(zip(expected['pair_id'],expected['map5']))
    outputs={}
    for history,metadata in held.items():
        scores={objective:np.mean([by[objective,seed][history] for seed in plan['optimization']['seeds']],axis=0) for objective in plan['optimization']['objectives']}
        scores['ensemble']=.5*(scores['pairwise']+scores['listwise'])
        for candidate,value in scores.items():
            assert value.shape==(len(metadata),50) and np.isfinite(value).all(); outputs[history+'_'+candidate]=value
        historyrows=[]
        for index,record in enumerate(metadata):
            truth=set(record['truth'])
            if not truth: continue
            original=record['all_hands']; count=len(record['selected_hands'])
            assert record['selected_hands']==original[:50]
            for candidate in ['baseline','pairwise','listwise','ensemble']:
                if candidate=='baseline': ranking=original
                else:
                    order=np.argsort(-scores[candidate][index,:count],kind='stable')
                    ranking=[record['selected_hands'][i] for i in order]+original[count:]
                assert set(ranking)==set(original) and len(ranking)==len(original)
                value=ap5(ranking[:5],truth)
                if candidate=='baseline' and history=='full': assert abs(value-expected[record['pair_id']])<1e-12
                historyrows.append({'pair_id':record['pair_id'],'table_id':record['table_id'],'family':record['family'],
                    'experiment':history,'candidate':candidate,'map5':value})
        rows+=historyrows
        for candidate in ['baseline','pairwise','listwise','ensemble']:
            summary.append({'history':history,'candidate':candidate,**summarize([row for row in historyrows if row['candidate']==candidate])})
        for candidate in ['pairwise','listwise','ensemble']:
            comparisons.append({'history':history,'candidate':candidate,**compare(historyrows,candidate,baseline='baseline')})
    pl.DataFrame(rows).write_csv(OUT/'screen-pairs.csv'); np.savez_compressed(OUT/'ensemble-predictions.npz',**outputs)
    gates=[]
    for candidate in ['pairwise','listwise','ensemble']:
        records=[row for row in comparisons if row['candidate']==candidate]
        gains=[row['delta'] for row in records]
        family=[value for row in records for value in row['by_family_delta'].values()]
        gates.append({'candidate':candidate,'full_and_short_gains':gains,'mean_gain':float(np.mean(gains)),
            'minimum_family_gain':min(family),'passed':bool(min(gains)>=.005 and min(family)>=-.025)})
    passing=[row for row in gates if row['passed']]
    selected=max(passing,key=lambda row:row['mean_gain'])['candidate'] if passing else None
    report={'completed_at':datetime.now(timezone.utc).isoformat(),'evaluation_source_sha256':sha(__file__),
        'plan_sha256':sha(OUT/'plan.json'),'worker_summary_sha256':sha(result/'summary.json'),
        'device':finished['device'],'summary':summary,'comparisons':comparisons,'gates':gates,'selected_for_confirmation':selected,
        'all_full_baseline_pair_metrics_exact':True,'all_protected_submissions_unchanged':True,
        'short_history_scope':'Pair thinning with rebuilt ranking contexts; full-phase baselines and normal references retained. Coherent-history confirmation required if candidate advances.',
        'next_action':'Freeze selected recipe and confirm other pools/omitted families/coherent histories before evaluation inference.' if selected else 'Reject fixed construction; no subgroup rescue, test-population inference, or new submission.'}
    path=OUT/'screen.json'; assert not path.exists(); path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
