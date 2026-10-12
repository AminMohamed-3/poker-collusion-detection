"""Report all fixed models on deliberate synthetic shifts; no selection."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'trial'
sys.path.insert(0,str(R/'src'))
from robust_pair_experiment import stable_ap,LABELS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ce(y,p):return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())
plan=json.loads((O/'plan.json').read_text());assert 'varied' not in plan['arms']
frozen=json.loads((H/'stress-plan.json').read_text());assert frozen['trial_plan_sha256']==sha(O/'plan.json')
metadata=json.loads((H/'varied/features/metadata.json').read_text())['views'][0]
y=np.array([LABELS[p['behavior_family']] for p in metadata]);g=np.array([int(p['table_id'].replace('SSPOOL','')) for p in metadata])
labels=['long_deep','moderate_deep','moderate_shallow','short_shallow']
nr=json.loads((O/'stress-inference-receipt.json').read_text());tr=json.loads((O/'tree-stress-receipt.json').read_text())
assert nr['no_fitting'] and nr['no_checkpoint_or_weight_selection'] and tr['no_fitting']
assert nr['source_sha256']==frozen['stress_inference_sha256']
assert nr['stress_features_sha256']==tr['features_sha256']==frozen['stress_dataset_sha256']
rows=[]
for arm in plan['arms']:
    for variant in plan['primary_data_endpoints']:
        predictions=[]
        seeds=plan['seeds'] if variant in plan['variants'] else [0]
        for seed in seeds:
            d=O/arm/(f'{variant}-{seed}' if seed else variant)
            receipts=nr['fits'] if seed else tr['fits']
            r=next(r for r in receipts if r['arm']==arm and r['variant']==variant and (not seed or r['seed']==seed))
            assert sha(d/'stress-predictions.npz')==r['predictions_sha256']
            predictions.append(np.load(d/'stress-predictions.npz')['full'])
        p=np.mean(predictions,axis=0);assert p.shape==(len(y),4)
        for name,mask in [('all',np.ones(len(y),bool))]+[(name,((g-64)//16)==i) for i,name in enumerate(labels)]:
            yy=y[mask];pp=p[mask]
            rows.append({'arm':arm,'variant':variant,'regime':name,'pairs':int(sum(mask)),'positive_pairs':int(sum(yy>0)),
                         'pair_ap':stable_ap(yy>0,1-pp[:,0]),'log_loss':ce(yy,pp)})
deltas=[]
for variant in plan['primary_data_endpoints']:
    for regime in ['all']+labels:
        b=next(r for r in rows if r['arm']=='resampled_control' and r['variant']==variant and r['regime']==regime)
        t=next(r for r in rows if r['arm']=='calibrated' and r['variant']==variant and r['regime']==regime)
        deltas.append({'variant':variant,'regime':regime,'pair_ap_delta':t['pair_ap']-b['pair_ap'],'logloss_improvement':b['log_loss']-t['log_loss']})
out=O/'stress-evaluation.json';assert not out.exists()
out.write_text(json.dumps({'source_sha256':sha(__file__),'plan_sha256':sha(O/'plan.json'),'metrics':rows,'deltas':deltas,'production_admitted':False,'interpretation':'Inference only on explicitly shifted synthetic histories excluded from training. Few positives per regime; no claim that these represent the private test. No tuning or gate changes from this diagnostic.'},indent=2))
print(json.dumps(deltas,indent=2))
