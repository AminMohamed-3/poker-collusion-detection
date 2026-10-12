"""Bounded cross-pool action-model capacity test, no generator or held scoring."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (H/'capacity-probe.json').exists();fit=json.loads((H/'controller-fit.json').read_text());f=pl.read_parquet(H/'training-actions-causal.parquet').filter(pl.col('behavior_family')=='soft_play');old=np.load(H/'own_style-oof.npz');y=f['action_class'].to_numpy();g=f['table_id'].to_numpy();np.testing.assert_array_equal(y,old['y']);np.testing.assert_array_equal(g.astype(str),old['groups']);x=f.select(fit['inputs']).to_numpy();counts=f.group_by('pair_id').len();w=f.select('pair_id').join(counts,on='pair_id',how='left')['len'].to_numpy()**-1.*len(y)/counts.height
    variants={'depth4_300':{**fit['parameters'],'n_estimators':300,'learning_rate':.03,'num_leaves':15,'max_depth':4},'depth5_300':{**fit['parameters'],'n_estimators':300,'learning_rate':.03,'num_leaves':31,'max_depth':5}}
    plan={'at':datetime.now(timezone.utc).isoformat(),'parameters':variants,'inputs':fit['inputs'],'gate':'Weighted OOF action loss gain >=.002 versus selected100-tree causal-style model, positive in all3 fixed pool folds. Prefer depth4 if it passes; select depth5 only if it passes and adds at least .001 mean gain beyond depth4. No outer-held data or new generator until this probe completes.','training_only':True,'source_sha256':sha(Path(__file__)),'baseline_oof_sha256':sha(H/'own_style-oof.npz')};(H/'capacity-probe-plan.json').write_text(json.dumps(plan,indent=2));results={}
    for name,params in variants.items():
        pred=np.zeros_like(old['p']);folds=[]
        for _,(tr,te) in enumerate(GroupKFold(3).split(x,groups=g)):
            train=f[tr];n=train.group_by('pair_id').len();tw=train.select('pair_id').join(n,on='pair_id',how='left')['len'].to_numpy()**-1.*len(tr)/n.height;m=LGBMClassifier(**params).fit(x[tr],y[tr],sample_weight=tw);pred[te]=m.predict_proba(x[te]);gain=np.log(np.maximum(pred[te,y[te]],1e-12))-np.log(np.maximum(old['p'][te,y[te]],1e-12));folds.append(float(np.average(gain,weights=w[te])))
        gain=np.log(np.maximum(pred[np.arange(len(y)),y],1e-12))-np.log(np.maximum(old['p'][np.arange(len(y)),y],1e-12));mean=float(np.average(gain,weights=w));results[name]={'mean_oof_loss_gain':mean,'fold_gains':folds,'passes':bool(mean>=.002 and min(folds)>0)};np.savez_compressed(H/(name+'-oof.npz'),p=pred,y=y,groups=g.astype(str));print(name,results[name],flush=True)
    selected='depth4_300' if results['depth4_300']['passes'] else None
    if results['depth5_300']['passes'] and (selected is None or results['depth5_300']['mean_oof_loss_gain']-results['depth4_300']['mean_oof_loss_gain']>=.001):selected='depth5_300'
    out={'results':results,'selected':selected,'no_outer_validation_read':True,'no_generator_or_detector_modified':True};(H/'capacity-probe.json').write_text(json.dumps(out,indent=2));print('Selected:',selected,flush=True)
if __name__=='__main__':main()
