"""Match reference sample count for a descriptive coverage comparison."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from scipy.spatial.distance import cdist
H=Path(__file__).resolve().parent;E=H.parent
def main():
    d=json.loads((H/'result.json').read_text());n=d['results']['pooled']['examples'];plan={'at':datetime.now(timezone.utc).isoformat(),'purpose':'Sample-size adjustment to the preceding descriptive neighbor audit. Does not change any gate or filter data.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'samples':1000,'seed':87201,'rows_per_sample':n,'reference':'Sample without replacement from the92 original targets, preserving their leave-pool-out nearest-neighbor mappings. Intervals describe this finite reference resampling, not generator uncertainty.'};(H/'matched-reference-plan.json').write_text(json.dumps(plan,indent=2))
    meta=json.loads((E/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs'];lookup={r['pair_id']:r['table_id'] for r in meta};groups=np.array([lookup[k] for k in d['nearest_target_order']]);z=np.load(H/'embedding.npz')['real'];dd=cdist(z,z);dd[groups[:,None]==groups[None,:]]=np.inf;nearest=dd.argmin(1);rng=np.random.default_rng(87201);rows=[]
    for _ in range(1000):
        ids=rng.choice(len(z),n,replace=False);counts=np.bincount(nearest[ids],minlength=len(z));p=counts[counts>0]/n;rows.append([int((counts>0).sum()),float(np.exp(-np.sum(p*np.log(p)))),float(np.sort(counts)[-5:].sum()/n)])
    a=np.array(rows);keys=['distinct_nearest_training_targets','effective_nearest_targets','top5_neighbor_share'];out={'training_only':True,'post_primary_audit_sample_size_correction':True,'reference_mean':dict(zip(keys,a.mean(0).tolist())),'reference_95pct_sampling_range':dict(zip(keys,np.quantile(a,[.025,.975],axis=0).T.tolist())),'synthetic_pooled':{k:d['results']['pooled'][k] for k in keys},'not_a_gate_or_generator_confidence_interval':True};(H/'matched-reference.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
