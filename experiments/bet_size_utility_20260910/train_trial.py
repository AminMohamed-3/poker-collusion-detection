"""Thirty-three predetermined small tree fits, no validation labels."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,time
import numpy as np
import joblib
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent;D=H/'trial'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    f=json.loads((D/'freeze.json').read_text());plan=json.loads((H/'study-plan.json').read_text())
    assert sha(__file__)==f['source_hashes']['train_trial.py'] and sha(H/'study-plan.json')==f['study_plan_sha256']
    for arm in plan['detector_arms']:
        assert sha(D/(arm+'.npz'))==f['inputs'][arm]['sha256'];a=np.load(D/(arm+'.npz'))
        for seed in plan['model_seeds']:
            p=D/arm/str(seed);p.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
            params={**plan['tree_parameters'],'random_state':seed}
            m=LGBMClassifier(**params).fit(a['train_x'],a['train_y'],sample_weight=np.full(len(a['train_y']),1/3))
            joblib.dump(m,p/'model.joblib');prob={tag:m.predict_proba(a[tag+'_x'],num_threads=2) for tag in ['full','shortened']}
            mm=joblib.load(p/'model.joblib')
            for tag in prob:
                np.testing.assert_array_equal(prob[tag],mm.predict_proba(a[tag+'_x'],num_threads=2))
                assert np.isfinite(prob[tag]).all();np.testing.assert_allclose(prob[tag].sum(1),1,atol=1e-12)
            np.savez_compressed(p/'predictions.npz',**prob)
            r={'arm':arm,'seed':seed,'seconds':time.perf_counter()-start,'parameters':params,'training_rows':len(a['train_y']),'held_labels_used':False,'reload_exact':True,'freeze_sha256':sha(D/'freeze.json'),'model_sha256':sha(p/'model.joblib'),'predictions_sha256':sha(p/'predictions.npz')}
            (p/'receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()
