"""Fixed pretrained TabPFN v2 inference; all data remain on the private instance."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from pathlib import Path
import sys
sys.path.insert(0,'/workspace/poker-research/tabpfn-v2/libs')
import hashlib
import json
import time
import numpy as np
import torch
from tabpfn import TabPFNClassifier

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    root=Path(__file__).resolve().parent; planpath=root/'worker-plan.json'; plan=json.loads(planpath.read_text())
    assert all(sha(root/n)==h for n,h in plan['files'].items())
    checkpoint=Path('/workspace/poker-research/tabpfn-v2')/plan['checkpoint_file']; assert sha(checkpoint)==plan['checkpoint_sha256']
    data=np.load(root/'arrays.npz',allow_pickle=False); assert set(data.files)=={'x_train','y_train','x_full','x_shortened'}
    torch.set_num_threads(2); torch.set_num_interop_threads(2); out=root/'results'; out.mkdir(exist_ok=True); receipts=[]
    for width in plan['variants']:
        tag='TabPFN'+str(width); receiptpath=out/(tag+'.json')
        if receiptpath.exists():
            receipt=json.loads(receiptpath.read_text()); assert receipt['plan_sha256']==sha(planpath) and sha(out/(tag+'.npz'))==receipt['predictions_sha256']; receipts.append(receipt); continue
        started=time.perf_counter()
        model=TabPFNClassifier(model_path=checkpoint,n_estimators=plan['n_estimators'],random_state=plan['random_state'],**plan['parameters'])
        model.fit(data['x_train'][:,:width],data['y_train'])
        predictions={}; classes=model.classes_; assert 0 in classes
        for history in ['full','shortened']:
            probability=model.predict_proba(data['x_'+history][:,:width])
            assert probability.shape==(len(data['x_'+history]),len(classes)) and np.isfinite(probability).all()
            predictions[history]=probability; print(json.dumps({'variant':tag,'history':history,'rows':len(probability),'seconds':time.perf_counter()-started}),flush=True)
        predictions['classes']=classes; path=out/(tag+'.npz'); np.savez_compressed(path,**predictions)
        receipt={'plan_sha256':sha(planpath),'variant':tag,'width':width,'predictions_sha256':sha(path),'checkpoint_sha256':sha(checkpoint),'seconds':time.perf_counter()-started,'classes':classes.tolist(),'held_labels_used':False,'telemetry_disabled':True,'torch':torch.__version__}
        write(receiptpath,receipt); receipts.append(receipt); del model; torch.cuda.empty_cache()
    write(out/'summary.json',{'plan_sha256':sha(planpath),'receipts':receipts,'fits':len(receipts)}); print(json.dumps({'complete':True,'fits':len(receipts)}),flush=True)


if __name__=='__main__': main()
