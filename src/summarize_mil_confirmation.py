"""Apply the original omitted-family gate; never use known gates for omission."""
import json
from datetime import datetime,timezone
import numpy as np
from robust_pair_experiment import ROOT,sha

OUT=ROOT/'reports/mil-pair-experiment'
def main():
    plan=json.loads((OUT/'plan.json').read_text()); paths=[OUT/('omit_'+f)/'screen.json' for f in ['directed_transfer','soft_play','coordinated_isolation']]
    available=[p for p in paths if p.exists()]; assert len(available)>=2
    results=[json.loads(p.read_text()) for p in available]; assert all(r['plan_sha256']==sha(OUT/'plan.json') for r in results)
    gate=plan['confirmation_gate']; decisions={}
    for candidate in plan['recipes']:
        values=[r['delta']['sparse_ap'] for result in results for r in result['metrics'] if r['candidate']==candidate]
        worst=min(values); complete=len(available)==len(paths)
        passed=complete and float(np.mean(values))>=gate['omitted_mean_sparse_delta_min'] and worst>=gate['omitted_worst_sparse_delta_min']
        decisions[candidate]={'admitted':False,'omitted_gate_passed':passed,'complete_omitted_results':complete,'observed_mean_sparse_delta':float(np.mean(values)),'observed_worst_sparse_delta':worst,'fails_worst_gate_already':worst<gate['omitted_worst_sparse_delta_min'],'evaluated_cases':len(values)}
    assert all(d['fails_worst_gate_already'] for d in decisions.values()),'A candidate requires further confirmation'
    value={'created_at':datetime.now(timezone.utc).isoformat(),'parent_plan_sha256':sha(OUT/'plan.json'),'ceiling_amendment_sha256':sha(OUT/'ceiling-amendment.json'),
           'results_sha256':{p.parent.name:sha(p) for p in available},'decisions':decisions,'production_predictions_created':False,'further_fits_stopped':True,'reason':'Each equal blend already violates the frozen worst omitted-case floor of -.05. An unretrieved remaining case cannot repair a worst-case failure.' if len(available)<3 else 'Both fixed blends fail omitted-family protection; retain incumbent.'}
    (OUT/'admission.json').write_text(json.dumps(value,indent=2)+'\n')
    lines=['# Multiple-instance pair detection','', 'Both fixed neural risk blends are rejected. Their first screen passed, but withheld-family performance violates the original worst-case protection. No production file uses either model.','', '| Recipe | Mean observed sparse AP delta | Worst observed delta | Evaluated cases |','|---|---:|---:|---:|']
    for n,d in decisions.items(): lines.append(f"| {n} | {d['observed_mean_sparse_delta']:+.6f} | {d['observed_worst_sparse_delta']:+.6f} | {d['evaluated_cases']} |")
    lines+=['',value['reason'],'','Six fold0 fits and three six-fit omitted-family jobs completed remotely before the instance restarted. Returned artifacts must have verified archive/member hashes and reload receipts before local scoring. See the individual transfer records for which results are locally backed up.','', 'The initial known-screen threshold was amended before any held predictions were imported/scored because its requested gain exceeded the mathematical remaining ceiling. That change is fully recorded; the omitted gate was never amended.','', 'Models pool all observed hands using mean, maximum, attention and event evidence, with three fixed seeds and final-ten-epoch weight averaging. The auxiliary version ranks supplied annotations only within target pairs. All normal references are pool-cross-fitted; held labels stayed local. No public scores choose weights, seeds, epochs or family-specific exceptions.','', 'Omitted packages reconstructed remotely have different ZIP container encodings, but every complete uncompressed NPY member is byte-identical to the frozen local package. Serialization-equivalence receipts bind both archive hashes and worker plans. This changes no array or optimization setting.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines)); print(json.dumps(value),flush=True)

if __name__=='__main__': main()
