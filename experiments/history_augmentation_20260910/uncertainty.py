"""Descriptive pool-bootstrap loss intervals, separate from frozen admission gates."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'trial'
plan=json.loads((O/'plan.json').read_text());assert (O/'evaluation.json').exists()
meta=json.loads((R/'experiments/mhc_local/fold0/held-metadata.json').read_text())
mapping={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}
groups=np.array([r['table_id'] for r in meta['full']['pairs']]);pools=np.unique(groups)
assert [r['pair_id'] for r in meta['full']['pairs']]==[r['pair_id'] for r in meta['shortened']['pairs']]
y=np.array([mapping[r['behavior_family']] for r in meta['full']['pairs']])
ids=[np.flatnonzero(groups==p) for p in pools];counts=np.array([len(i) for i in ids])
rng=np.random.default_rng(11891);weights=rng.multinomial(len(pools),np.full(len(pools),1/len(pools)),size=2000)
rows=[]
for variant in plan['primary_data_endpoints']:
    diffs=[]
    for seed in plan['seeds'] if variant in plan['variants'] else [0]:
        name=f'{variant}-{seed}' if seed else variant
        b=np.load(O/'resampled_control'/name/'predictions.npz');t=np.load(O/'calibrated'/name/'predictions.npz')
        for history in ['full','shortened']:
            # Positive differences mean lower treatment loss.
            diffs.append(np.log(np.maximum(t[history][np.arange(len(y)),y],1e-12))-np.log(np.maximum(b[history][np.arange(len(y)),y],1e-12)))
    diff=np.mean(diffs,axis=0);sums=np.array([diff[i].sum() for i in ids]);samples=(weights@sums)/(weights@counts)
    rows.append({'variant':variant,'mean_paired_logloss_improvement':float(diff.mean()),'pool_bootstrap_95_interval':[float(x) for x in np.quantile(samples,[.025,.975])]})
out=O/'uncertainty.json';assert not out.exists()
out.write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,'pools':len(pools),'replicates':2000,'seed':11891,'interpretation':'Descriptive reused-holdout intervals; pools are resampled jointly across both histories and all neural seeds. Conditional on fixed fitted models; not seed uncertainty, correction for repeated research, or a new admission criterion.'},indent=2))
print(json.dumps(rows,indent=2))
