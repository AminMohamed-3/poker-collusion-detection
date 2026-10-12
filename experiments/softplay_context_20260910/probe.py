"""Prespecified training-pool cross-validation of richer causal action context."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent;R=H.parents[1];OLD=H.parent/'softplay_conditional_20260910';sys.path.insert(0,str(R/'src'))
from conditional_action_features import INPUTS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def weight(f):
    counts=f.group_by('pair_id').len();return f.select('pair_id').join(counts,on='pair_id',how='left')['len'].to_numpy()**-1.*f.height/counts.height
def main():
    assert not (H/'probe.json').exists();fit=json.loads((OLD/'controller-fit.json').read_text());f=pl.read_parquet(H.parent/'softplay_fast_20260910/training-actions.parquet').filter(pl.col('behavior_family')=='soft_play');assert set(f['table_id'])<=set(fit['training_pools'])
    manifest=json.loads((OLD/'archive-manifest.json').read_text());assert sha(OLD/'controller-oof.npz')==manifest['files']['controller-oof.npz']['sha256'];prior=np.load(OLD/'controller-oof.npz',allow_pickle=True);y=f['action_class'].to_numpy();g=f['table_id'].to_numpy();np.testing.assert_array_equal(y,prior['y']);np.testing.assert_array_equal(g,prior['groups'])
    public=[n for n in INPUTS if not n.startswith(('own_style_','partner_style_'))];assert len(public)==29 and len(fit['inputs'])==21
    variants={'public29':public,'public29_partner_strength':public+['partner_strength']}
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'base_oof_sha256':sha(OLD/'controller-oof.npz'),'variants':variants,'parameters':fit['parameters'],'gate':'Mean equal-pair weighted OOF action logloss improvement >=.001 over existing21-input model, with positive loss gain in all3 fixed pool folds. Choose public29 if it passes; consider partner strength only if public29 fails and private variant passes. Private feature gain is not proof of actual card sharing. No outer validation, classifier tuning or generation until this probe completes.','training_only':True,'style_exclusion':'Whole-history style columns excluded because they are not causally available to a sequential policy.'};(H/'probe-plan.json').write_text(json.dumps(plan,indent=2));results={}
    for name,inputs in variants.items():
        pred=np.zeros_like(prior['p']);folds=[]
        for k,(tr,te) in enumerate(GroupKFold(3).split(np.zeros(len(y)),groups=g)):
            assert not set(g[tr])&set(g[te]);model=LGBMClassifier(**fit['parameters']).fit(f[tr].select(inputs).to_numpy(),y[tr],sample_weight=weight(f[tr]));pred[te]=model.predict_proba(f[te].select(inputs).to_numpy());delta=np.log(np.maximum(pred[te,y[te]],1e-12))-np.log(np.maximum(prior['p'][te,y[te]],1e-12));folds.append(float(np.average(delta,weights=prior['weights'][te])))
        delta=np.log(np.maximum(pred[np.arange(len(y)),y],1e-12))-np.log(np.maximum(prior['p'][np.arange(len(y)),y],1e-12));mean=float(np.average(delta,weights=prior['weights']));contexts={}
        for context,take in [('all',np.ones(len(y),dtype=bool)),('facing_partner',f['facing_partner'].to_numpy()==1),('postflop',f['street_index'].to_numpy()>0),('folds',y==0)]:contexts[context]=float(np.average(delta[take],weights=prior['weights'][take]))
        results[name]={'inputs':inputs,'mean_weighted_oof_loss_gain':mean,'fold_loss_gains':folds,'contexts':contexts,'passes_action_probe':bool(mean>=.001 and min(folds)>0)}
        np.savez_compressed(H/(name+'-oof.npz'),p=pred,y=y,groups=g.astype(str));print(name,results[name],flush=True)
    chosen=next((name for name in variants if results[name]['passes_action_probe']),None)
    out={'results':results,'selected_for_generator_trial':chosen,'no_outer_validation_read':True,'no_new_generator_or_utility_score':True,'pool_count':len(set(g)),'pairs':f['pair_id'].n_unique(),'plan_sha256':sha(H/'probe-plan.json')};(H/'probe.json').write_text(json.dumps(out,indent=2));print('Selected:',chosen,flush=True)
if __name__=='__main__':main()
