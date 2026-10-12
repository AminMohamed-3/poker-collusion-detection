"""Fixed own-style and pair-style action prediction checks on original training pools."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent;OLD=H.parent/'softplay_conditional_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def weight(f):
    counts=f.group_by('pair_id').len();return f.select('pair_id').join(counts,on='pair_id',how='left')['len'].to_numpy()**-1.*f.height/counts.height
def main():
    assert not (H/'probe.json').exists();fit=json.loads((OLD/'controller-fit.json').read_text());f=pl.read_parquet(H/'training-actions-causal.parquet').filter(pl.col('behavior_family')=='soft_play');assert set(f['table_id'])<=set(fit['training_pools'])
    assert sha(OLD/'controller-oof.npz')==json.loads((OLD/'archive-manifest.json').read_text())['files']['controller-oof.npz']['sha256'];prior=np.load(OLD/'controller-oof.npz',allow_pickle=True);y=f['action_class'].to_numpy();g=f['table_id'].to_numpy();np.testing.assert_array_equal(y,prior['y']);np.testing.assert_array_equal(g,prior['groups'])
    own=['causal_own_style_'+n for n in ['fold','check','call','raise']];partner=[n.replace('_own_','_partner_') for n in own];variants={'own_style':fit['inputs']+own,'pair_style':fit['inputs']+own+partner}
    plan={'at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'data_sha256':sha(H/'training-actions-causal.parquet'),'base_oof_sha256':sha(OLD/'controller-oof.npz'),'variants':variants,'parameters':fit['parameters'],'gate':'Equal-pair weighted mean OOF action loss gain >=.001 and gain >0 in all3 fixed pool folds. Prefer own_style if it passes; choose pair_style instead only if it passes and improves mean gain by at least .001 beyond own_style. If neither passes, no new generator model.','training_only':True,'causality':'Action tendencies use completed earlier hands only, uniform pseudocount1, no current hand or future.'};(H/'probe-plan.json').write_text(json.dumps(plan,indent=2));results={}
    for name,inputs in variants.items():
        pred=np.zeros_like(prior['p']);folds=[]
        for k,(tr,te) in enumerate(GroupKFold(3).split(np.zeros(len(y)),groups=g)):
            assert not set(g[tr])&set(g[te]);m=LGBMClassifier(**fit['parameters']).fit(f[tr].select(inputs).to_numpy(),y[tr],sample_weight=weight(f[tr]));pred[te]=m.predict_proba(f[te].select(inputs).to_numpy());delta=np.log(np.maximum(pred[te,y[te]],1e-12))-np.log(np.maximum(prior['p'][te,y[te]],1e-12));folds.append(float(np.average(delta,weights=prior['weights'][te])))
        delta=np.log(np.maximum(pred[np.arange(len(y)),y],1e-12))-np.log(np.maximum(prior['p'][np.arange(len(y)),y],1e-12));mean=float(np.average(delta,weights=prior['weights']));results[name]={'inputs':inputs,'mean_weighted_oof_loss_gain':mean,'fold_loss_gains':folds,'passes':bool(mean>=.001 and min(folds)>0)}
        np.savez_compressed(H/(name+'-oof.npz'),p=pred,y=y,groups=g.astype(str));print(name,mean,folds,flush=True)
    chosen='own_style' if results['own_style']['passes'] else None
    if results['pair_style']['passes'] and (chosen is None or results['pair_style']['mean_weighted_oof_loss_gain']-results['own_style']['mean_weighted_oof_loss_gain']>=.001):chosen='pair_style'
    out={'results':results,'selected':chosen,'no_outer_validation_read':True,'no_generator_or_detector_modified':True,'plan_sha256':sha(H/'probe-plan.json')};(H/'probe.json').write_text(json.dumps(out,indent=2));print('Selected:',chosen,flush=True)
if __name__=='__main__':main()
