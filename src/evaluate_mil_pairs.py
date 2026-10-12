"""Local-only held-label scoring for predeclared MIL risk recipes."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse
import json
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,LABELS,stable_ap
from evaluate_conditional_behavior import family_metrics

OUT=ROOT/'reports/mil-pair-experiment'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def scorecard(y,risk,roles):
    weights=np.where(y>0,.01/np.mean(y>0),.99/np.mean(y==0))
    ap=stable_ap(y>0,risk); bm=family_metrics(y,risk,roles)['behavior_map']
    return {'pair_ap':ap,'sparse_ap':stable_ap(y>0,risk,weights),'behavior_map':bm,'pair_behavior':.7*ap+.1*bm}


def evaluate(job):
    root=OUT/job; report=root/'screen.json'; assert not report.exists()
    plan=json.loads((OUT/'plan.json').read_text()); audit=json.loads((root/'audit.json').read_text()); package=root/'package'
    assert audit['parent_plan_sha256']==sha(OUT/'plan.json')
    assert audit['metadata_sha256']==sha(root/'metadata.json')
    assert audit['worker_plan_sha256']==sha(package/'worker-plan.json')
    summary=json.loads((root/'returned/summary.json').read_text()); worker_sha=audit['worker_plan_sha256']
    if (root/'returned/serialization-equivalence.json').exists():
        equivalent=json.loads((root/'returned/serialization-equivalence.json').read_text()); expected=json.loads((root/'expected-npy-members.json').read_text())
        assert equivalent['original_worker_plan_sha256']==worker_sha and equivalent['members']==expected['members'] and equivalent['all_uncompressed_npy_members_identical']
        original=json.loads((package/'worker-plan.json').read_text()); returned=json.loads((root/'returned/worker-plan.json').read_text())
        assert returned['files']['arrays.npz']==equivalent['repacked_array_archive_sha256']
        returned['files']['arrays.npz']=equivalent['original_array_archive_sha256']; assert returned==original
        worker_sha=equivalent['worker_plan_sha256']; assert sha(root/'returned/worker-plan.json')==worker_sha
    assert summary['plan_sha256']==worker_sha and summary['fits']==6
    assert all(sha(ROOT/p)==h for p,h in plan['protected'].items())
    predictions={}
    for r in summary['receipts']:
        assert r['plan_sha256']==summary['plan_sha256'] and r['reload_exact'] and not r['held_labels_used']
        assert r['fixed_epochs']==plan['optimization']['epochs']
        assert all(sha(root/'returned'/p)==h for p,h in r['files'].items())
        tag=r['name']+'-'+str(r['seed']); predictions[tag]=dict(np.load(root/'returned'/(tag+'.npz'),allow_pickle=False))
    meta=json.loads((root/'metadata.json').read_text()); rows=[]; allpred=[]; seedrows=[]
    for history in ['full','shortened']:
        m=meta[history]; y=np.array([LABELS[p['behavior_family']] for p in m['pairs']]); control=np.array(m['control_risk']); roles=np.array(m['control_roles'])
        recipes={'baseline':control}
        for name in plan['optimization']['objectives']:
            values=np.stack([predictions[name+'-'+str(s)][history] for s in plan['optimization']['seeds']])
            np.testing.assert_allclose(values.sum(2),1,atol=1e-6)
            risk=1-values.mean(0)[:,0]; recipes[name+'_diagnostic']=risk; recipes[name+'_equal_blend']=(risk+control)/2
            for seed,prob in zip(plan['optimization']['seeds'],values): seedrows.append({'history':history,'name':name,'seed':seed,**scorecard(y,1-prob[:,0],roles)})
        baseline=scorecard(y,control,roles)
        for name,risk in recipes.items():
            metrics=scorecard(y,risk,roles)
            rows.append({'job':job,'history':history,'candidate':name,**metrics,'delta':{k:metrics[k]-baseline[k] for k in metrics}})
            allpred.extend({'pair_id':p['pair_id'],'table_id':p['table_id'],'history':history,'candidate':name,'risk':float(r),'role':int(c)} for p,r,c in zip(m['pairs'],risk,roles))
    decisions={}
    gates=plan['screen_gate']; amendment=json.loads((OUT/'ceiling-amendment.json').read_text())
    assert amendment['original_plan_sha256']==sha(OUT/'plan.json')
    base_mean=np.mean([r['pair_behavior'] for r in rows if r['candidate']=='baseline'])
    minimum=min(amendment['maximum_minimum_gain'],amendment['fraction_remaining']*max(0,amendment['ceiling']-base_mean))
    for candidate in plan['recipes']:
        items=[r for r in rows if r['candidate']==candidate]
        mean=float(np.mean([r['delta']['pair_behavior'] for r in items])); worst=min(r['delta']['pair_behavior'] for r in items); sparse=float(np.mean([r['delta']['sparse_ap'] for r in items]))
        passed=mean>=minimum and worst>=gates['worst_pair_behavior_delta_min'] and sparse>=gates['mean_sparse_ap_delta_min']
        decisions[candidate]={'passed_screen':passed,'mean_pair_behavior_delta':mean,'required_mean_gain':minimum,'worst_pair_behavior_delta':worst,'mean_sparse_ap_delta':sparse}
    pl.DataFrame(allpred).write_csv(root/'pair-predictions.csv')
    value={'job':job,'plan_sha256':sha(OUT/'plan.json'),'ceiling_amendment_sha256':sha(OUT/'ceiling-amendment.json'),'evaluation_source_sha256':sha(__file__),'results_sha256':sha(root/'returned/summary.json'),'decisions':decisions,'metrics':rows,'seed_diagnostics':seedrows,'not_an_untouched_holdout':True}
    write(report,value); print(json.dumps(value),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--job',required=True); args=p.parse_args(); evaluate(args.job)
