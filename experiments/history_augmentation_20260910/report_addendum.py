"""Explain the material ensemble disagreement without changing any screen."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;O=H/'trial'
r=json.loads((O/'evaluation.json').read_text());rows=r['metrics'];cases=[]
for history in ['full','shortened']:
    b=next(v for v in rows if v['history']==history and v['arm']=='resampled_control' and v['variant']=='single_pass')
    t=next(v for v in rows if v['history']==history and v['arm']=='calibrated' and v['variant']=='single_pass')
    cases.append({'history':history,'control_logloss':b['log_loss'],'augmented_logloss':t['log_loss'],'control_pair_ap':b['pair_ap'],'augmented_pair_ap':t['pair_ap']})
s=json.loads((H/'summary.json').read_text());s['singlepass_ensemble_caveat']=cases
s['singlepass_pass_scope']='The predeclared average-per-seed screen passes, but mean-probability ensembling worsens loss in both histories and shortened-history AP. Descriptive loss interval crosses zero. Replication lead only; no production or architecture admission.'
s['synthetic_stress_shared_policy_source']=True
(H/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
p=H/'RESULTS.md';text=p.read_text(encoding='utf-8')
addition='''
**The SinglePass pass is limited.** Its mean per-seed log-loss improvement is .003254, but the descriptive 95% pool-bootstrap interval is [-.007091, .014807], which includes zero. The three-seed probability ensemble worsens full-history log loss from .003175 to .006245 and shortened-history log loss from .022727 to .023477. Its shortened-history Pair AP falls from .999320 to .998129. These were already reported metrics, not a newly chosen rejection gate. The fixed per-seed screen still records a pass; the ensemble disagreement and uncertainty make this a replication lead, not a production candidate or evidence that mHC architecture is superior.

'''
text=text.replace('## What was tested\n',addition+'## What was tested\n',1)
text=text.replace('These are fixed-model predictions on generated data excluded from training.', 'These are fixed-model predictions on generated data excluded from training. The shifted and calibrated batches share ordinary-policy models and intervention rules, so the stress gains do not establish robustness to independently generated behavior.')
p.write_text(text,encoding='utf-8')
(H/'report-addendum-receipt.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'purpose':'Report material prespecified ensemble disagreement and interval, preserving every frozen gate, metric and model.','evaluation_sha256':hashlib.sha256((O/'evaluation.json').read_bytes()).hexdigest(),'no_metric_or_gate_changed':True,'cases':cases},indent=2))
print(json.dumps(cases,indent=2))
