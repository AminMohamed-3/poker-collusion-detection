"""Score isolated pretrained predictions locally under the frozen gates."""
import argparse
import json
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,LABELS
from evaluate_mil_pairs import scorecard

OUT=ROOT/'reports/tabpfn-v2-experiment'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def run(job):
    dest=OUT/job; path=dest/'screen.json'; assert not path.exists()
    plan=json.loads((OUT/'plan.json').read_text()); audit=json.loads((dest/'audit.json').read_text())
    assert sha(dest/'metadata.json')==audit['metadata_sha256']; assert audit['plan_sha256']==sha(OUT/'plan.json')
    summary=json.loads((dest/'returned/summary.json').read_text()); assert summary['plan_sha256']==audit['worker_plan_sha256']
    predictions={}
    for receipt in summary['receipts']:
        file=dest/'returned'/(receipt['variant']+'.npz')
        assert sha(file)==receipt['predictions_sha256'] and receipt['checkpoint_sha256']==plan['checkpoint_sha256'] and not receipt['held_labels_used']
        predictions[receipt['variant']]=dict(np.load(file,allow_pickle=False))
    meta=json.loads((dest/'metadata.json').read_text()); rows=[]; predrows=[]
    for history,m in meta.items():
        y=np.array([LABELS[p['behavior_family']] for p in m['pairs']]); control=np.array(m['control_risk']); roles=np.array(m['control_roles']); baseline=scorecard(y,control,roles)
        candidates={'baseline':control}
        for tag,result in predictions.items():
            p=result[history]; np.testing.assert_allclose(p.sum(1),1,atol=1e-6); risk=1-p[:,np.flatnonzero(result['classes']==0)[0]]
            candidates[tag+'_diagnostic']=risk; candidates[tag+'_equal_blend']=(risk+control)/2
        for name,values in candidates.items():
            metrics=scorecard(y,values,roles); rows.append({'job':job,'history':history,'candidate':name,**metrics,'delta':{k:metrics[k]-baseline[k] for k in metrics}})
            predrows.extend({'pair_id':p['pair_id'],'table_id':p['table_id'],'history':history,'candidate':name,'risk':float(v),'role':int(r)} for p,v,r in zip(m['pairs'],values,roles))
    gate=plan['known_screen_gate']; decisions={}
    for name in plan['recipes']:
        selected=[r for r in rows if r['candidate']==name]; mean=float(np.mean([r['delta']['pair_behavior'] for r in selected])); worst=min(r['delta']['pair_behavior'] for r in selected); sparse=float(np.mean([r['delta']['sparse_ap'] for r in selected]))
        decisions[name]={'known_screen_pass':mean>=gate['mean_pair_behavior_delta_min'] and worst>=gate['worst_pair_behavior_delta_min'] and sparse>=gate['mean_sparse_ap_delta_min'],'mean_pair_behavior_delta':mean,'worst_pair_behavior_delta':worst,'mean_sparse_ap_delta':sparse}
    pl.DataFrame(predrows).write_csv(dest/'pair-predictions.csv')
    result={'job':job,'plan_sha256':sha(OUT/'plan.json'),'evaluation_source_sha256':sha(__file__),'decisions':decisions,'metrics':rows}; write(path,result); print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--job',required=True); args=p.parse_args(); run(args.job)
