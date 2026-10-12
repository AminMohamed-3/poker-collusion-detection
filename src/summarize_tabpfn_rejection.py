import json
from pathlib import Path
import numpy as np
from robust_pair_experiment import ROOT,sha
out=ROOT/'reports/tabpfn-v2-experiment'; plan=json.loads((out/'plan.json').read_text()); assert not (out/'admission.json').exists()
allrows=[r for job in ['omit_soft_play','omit_coordinated_isolation','omit_directed_transfer'] for r in json.loads((out/job/'screen.json').read_text())['metrics']]
decisions={}
for candidate in plan['recipes']:
    rows=[r for r in allrows if r['candidate']==candidate]; mean=float(np.mean([r['delta']['sparse_ap'] for r in rows])); worst=min(r['delta']['sparse_ap'] for r in rows)
    passed=mean>=plan['confirmation_gate']['omitted_mean_sparse_delta_min'] and worst>=plan['confirmation_gate']['omitted_worst_sparse_delta_min']
    decisions[candidate]={'omitted_gate_passed':passed,'mean_sparse_delta':mean,'worst_sparse_delta':worst,'admitted':False}
assert not any(v['omitted_gate_passed'] for v in decisions.values())
value={'parent_plan_sha256':sha(out/'plan.json'),'decisions':decisions,'union_results_sha256':sha(out/'union-screen.json'),'fits_completed':8,'further_fits_stopped':True,'production_predictions_created':False,'external_data_uploaded_to_inference_api':False}
(out/'admission.json').write_text(json.dumps(value,indent=2)+'\n')
(out/'REPORT.md').write_text('''# TabPFN v2: rejected for this submission pipeline

Eight fixed GPU context fits completed: widths128/256 on the existing known screen and all three omitted-family tests. Checkpoint revision f851f2a3c941544733b712d8c0f96dfae9b28862, SHA-256 cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49. The public v2 attribution license was retained. Packages use a private overlay; telemetry is disabled, and competition data never go to an external inference API.

Both equal blends improved known full/short histories. But their worst omitted-family sparse-prevalence AP changes were -0.235980 and -0.234378, exceeding the declared -0.03 limit. Mean omitted gains were positive (+0.018016 and +0.029982), demonstrating why the worst-family protection matters.

A separately frozen, adaptively proposed test of global maximum/noisy-OR combinations also failed all four recipes. They applied the same formula to every pair; no subgroup gates or fitted weights were used. Those alternatives did not fix the soft-play failure. All remain excluded; no extra known/audit/coherent fits or production prediction are warranted under the frozen progression.

All returned archive and prediction hashes verified. Sources, plans, metadata, complete metrics, predictions and return receipts are retained. All partitions were reused development checks. No public-score fitting, upload, leaderboard improvement or prize resulted from this experiment.
''')
print(json.dumps(value,indent=2))
