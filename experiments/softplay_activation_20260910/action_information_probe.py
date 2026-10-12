"""Separate training-pool OOF diagnostic; does not alter the frozen controllers."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent;P=H.parent/'softplay_conditional_20260910';OLD=H.parent/'softplay_fast_20260910'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (H/'action-information-probe.json').exists()
    fit=json.loads((P/'controller-fit.json').read_text());f=pl.read_parquet(OLD/'training-actions.parquet').filter(pl.col('behavior_family')=='soft_play');assert sha(P/'controller-oof.npz')==json.loads((P/'archive-manifest.json').read_text())['files']['controller-oof.npz']['sha256'];prior=np.load(P/'controller-oof.npz',allow_pickle=True);y=f['action_class'].to_numpy();groups=f['table_id'].to_numpy();np.testing.assert_array_equal(y,prior['y']);np.testing.assert_array_equal(groups,prior['groups']);assert set(groups)<=set(fit['training_pools'])
    inputs=fit['inputs']+['partner_strength'];freeze={'at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'prior_oof_sha256':sha(P/'controller-oof.npz'),'inputs':inputs,'parameters':fit['parameters'],'purpose':'Check whether present partner-card strength adds conditional action information beyond own/public fields. OOF training-pool diagnostic only; no generator change or held-utility selection.'};(H/'action-information-probe-plan.json').write_text(json.dumps(freeze,indent=2))
    pred=np.zeros_like(prior['p']);folds=[]
    for i,(tr,te) in enumerate(GroupKFold(3).split(np.zeros(len(y)),groups=groups)):
        assert not set(groups[tr])&set(groups[te]);train=f[tr];counts=train.group_by('pair_id').len();weights=train.select('pair_id').join(counts,on='pair_id',how='left')['len'].to_numpy()**-1.*len(tr)/counts.height
        m=LGBMClassifier(**fit['parameters']).fit(train.select(inputs).to_numpy(),y[tr],sample_weight=weights);pred[te]=m.predict_proba(f[te].select(inputs).to_numpy())
        delta=np.log(np.maximum(pred[te,y[te]],1e-12))-np.log(np.maximum(prior['p'][te,y[te]],1e-12));folds.append(float(np.average(delta,weights=prior['weights'][te])))
    delta=np.log(np.maximum(pred[np.arange(len(y)),y],1e-12))-np.log(np.maximum(prior['p'][np.arange(len(y)),y],1e-12));w=prior['weights'];pools=sorted(set(groups));sums=np.array([np.sum(delta[groups==g]*w[groups==g]) for g in pools]);weights=np.array([sum(w[groups==g]) for g in pools]);b=np.random.default_rng(97591).integers(0,len(pools),(1000,len(pools)));est=sums[b].sum(1)/weights[b].sum(1)
    out={'training_only':True,'original_training_pools_used':len(pools),'original_soft_pairs':f['pair_id'].n_unique(),'mean_oof_loss_improvement_with_partner_strength':float(np.average(delta,weights=w)),'fold_improvements':folds,'pool_bootstrap_95pct_interval':np.quantile(est,[.025,.975]).tolist(),'no_controller_or_detector_modified':True,'not_causal_evidence_of_card_sharing':True,'interpretation':'An incremental prediction gain would identify omitted statistical information, not prove the organizer policy uses that information or that a new generator would help. No private-card policy was deployed or generated here.'}
    np.savez_compressed(H/'action-information-probe-oof.npz',p=pred,y=y,groups=groups);(H/'action-information-probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
