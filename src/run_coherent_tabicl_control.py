"""Frozen GPU control: original training contexts, new coherent held histories."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='4'
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from huggingface_hub import hf_hub_download
from tabicl import TabICLClassifier


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(y,risk):
    pos=y>0; weights=np.where(pos,.01/pos.mean(),.99/(1-pos.mean()))
    return {'pair_ap':float(average_precision_score(pos,risk)),'sparse_ap':float(average_precision_score(pos,risk,sample_weight=weights))}


def main():
    root=Path(__file__).resolve().parent; planpath=root/'plan.json'; plan=json.loads(planpath.read_text())
    assert all(sha(root/n)==h for n,h in plan['files'].items())
    out=root/'results'; out.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(4); assert torch.cuda.is_available()
    checkpoint=Path(hf_hub_download(plan['checkpoint_repo'],plan['checkpoint_file'],revision=plan['checkpoint_revision'],local_files_only=True))
    assert sha(checkpoint)==plan['checkpoint_sha256']
    results=[]
    for fold in plan['folds']:
        receipt=out/f'fold{fold}.json'; predictionpath=out/f'fold{fold}.npz'
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['plan_sha256']==sha(planpath) and saved['predictions_sha256']==sha(predictionpath)
            results.extend(saved['results']); continue
        start=time.perf_counter(); data=np.load(root/f'fold{fold}.npz',allow_pickle=False)
        model=TabICLClassifier(n_estimators=2,model_path=checkpoint,allow_auto_download=False,**plan['common_parameters'])
        model.fit(data['x_train'],data['y_train'])
        full=1-model.predict_proba(data['x_full'])[:,0]
        maxdiff=float(np.max(np.abs(full-data['reference_full_tabicl'])))
        assert maxdiff<=5e-5,'GPU full-history control differs unexpectedly'
        assert abs(metric(data['y_held'],full)['pair_ap']-metric(data['y_held'],data['reference_full_tabicl'])['pair_ap'])<=1e-5
        rows=[]; predictions={'full':full}
        for history in plan['histories']:
            risk=1-model.predict_proba(data['x_'+history])[:,0]; predictions[history]=risk
            for name,values in [('v10',.5*(risk+data['v5_'+history])),('conditional',data['conditional_'+history])]:
                rows.append({'fold':fold,'history':history,'candidate':name,**metric(data['y_held'],values)})
        np.savez_compressed(predictionpath,**predictions)
        saved={'plan_sha256':sha(planpath),'predictions_sha256':sha(predictionpath),'full_control_max_difference':maxdiff,'seconds':time.perf_counter()-start,'results':rows}
        receipt.write_text(json.dumps(saved,indent=2)+'\n'); results.extend(rows)
        print(json.dumps(saved),flush=True); del model; torch.cuda.empty_cache()
    lookup={(r['fold'],r['history'],r['candidate']):r for r in results}
    ordinary=[lookup[f,h,'conditional']['pair_ap']-lookup[f,h,'v10']['pair_ap'] for f in plan['folds'] for h in plan['histories']]
    sparse=[lookup[f,h,'conditional']['sparse_ap']-lookup[f,h,'v10']['sparse_ap'] for f in plan['folds'] for h in plan['histories']]
    passed=min(ordinary)>=-.003 and np.mean(ordinary)>0 and np.mean(sparse)>=.003 and min(sparse)>=-.03
    summary={'results':results,'ordinary_deltas':ordinary,'sparse_deltas':sparse,'pair_gate_passed':bool(passed)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
