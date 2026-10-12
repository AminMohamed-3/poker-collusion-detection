"""Fixed pair-tree recipes in both matched trial arms; no held labels loaded."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import hashlib,json
import joblib
import numpy as np
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((H/'plan.json').read_text());assert sha(__file__)==p['source_hashes'][Path(__file__).name]
for arm in p['arms']:
    a=np.load(H/arm/'tree-arrays.npz');assert sha(H/arm/'tree-arrays.npz')==p['tree_inputs'][arm]
    for n in [897,1217]:
        d=H/arm/f'tree{n}';d.mkdir(exist_ok=False)
        model=LGBMClassifier(**p['tree_parameters']).fit(a['train_pair'][:,:n],a['train_y'],sample_weight=np.full(len(a['train_y']),1/3))
        joblib.dump(model,d/'model.joblib')
        np.savez_compressed(d/'predictions.npz',**{tag:model.predict_proba(a[tag+'_pair'][:,:n],num_threads=2) for tag in ['full','shortened']})
        (d/'receipt.json').write_text(json.dumps({'plan_sha256':sha(H/'plan.json'),'model_sha256':sha(d/'model.joblib'),'predictions_sha256':sha(d/'predictions.npz'),'held_labels_used':False,'rows':len(a['train_y'])},indent=2));print(arm,n,'complete',flush=True)
