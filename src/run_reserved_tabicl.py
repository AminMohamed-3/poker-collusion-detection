"""One frozen GPU control fit for the 169-pool audit, with no model selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='4'
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from tabicl import TabICLClassifier
from huggingface_hub import hf_hub_download

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    root=Path(__file__).resolve().parent; plan=json.loads((root/'plan.json').read_text())
    assert all(sha(root/n)==h for n,h in plan['files'].items())
    out=root/'results'; out.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.set_num_interop_threads(4); assert torch.cuda.is_available()
    checkpoint=Path(hf_hub_download(plan['checkpoint_repo'],plan['checkpoint_file'],revision=plan['checkpoint_revision'],local_files_only=True)); assert sha(checkpoint)==plan['checkpoint_sha256']
    arrays=np.load(root/'arrays.npz',allow_pickle=False); started=time.perf_counter()
    model=TabICLClassifier(n_estimators=2,model_path=checkpoint,allow_auto_download=False,**plan['common_parameters'])
    model.fit(arrays['x_train'],arrays['y_train']); assert list(model.classes_)==[0,1,2,3]
    predictions={history:1-model.predict_proba(arrays['x_'+history])[:,0] for history in ['full','shortened']}
    assert all(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) for p in predictions.values())
    path=out/'predictions.npz'; np.savez_compressed(path,**predictions)
    receipt={'plan_sha256':sha(root/'plan.json'),'predictions_sha256':sha(path),'checkpoint_sha256':sha(checkpoint),'seconds':time.perf_counter()-started}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)

if __name__=='__main__': main()
