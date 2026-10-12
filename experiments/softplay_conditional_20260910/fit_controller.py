"""Training-pool-only action policies, with fixed three-fold pool checks."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import polars as pl
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
H=Path(__file__).resolve().parent;R=H.parents[1];PREV=H.parent/'softplay_fast_20260910'
INPUTS=['street_index','own_strength','own_pair','own_suited','own_high_rank','own_low_rank','position_from_button','price','log_pot_bb','log_stack_bb','players_active','prior_actions','own_prior_actions','own_prior_aggression','own_street_aggression','partner_prior_actions','partner_prior_aggression','partner_street_aggression','partner_last_action','outsider_prior_aggression','facing_partner']
PARAMS=dict(n_estimators=100,learning_rate=.05,num_leaves=7,max_depth=3,min_child_samples=40,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=31379,deterministic=True,force_col_wise=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weights(f):
    n=f.group_by('pair_id').len();return f.select('pair_id').join(n,on='pair_id',how='left')['len'].to_numpy().astype(float)**-1*len(f)/n.height
def model(f):return LGBMClassifier(**PARAMS).fit(f.select(INPUTS).to_numpy(),f['action_class'].to_numpy(),sample_weight=weights(f))
def main():
    assert not (H/'controller.joblib').exists()
    data=pl.read_parquet(PREV/'training-actions.parquet');pools=set(json.loads((H.parent/'history_realism_20260910/persistent_sessions/training-calibration.json').read_text())['training_pools']);assert set(data['table_id'])<=pools
    soft=data.filter(pl.col('behavior_family')=='soft_play');none=data.filter(pl.col('behavior_family')=='none')
    groups=soft['table_id'].to_numpy();pred=np.zeros((soft.height,4));base=np.zeros_like(pred);folds=[]
    # Four ordinary policies have previously used these development pools. This
    # check concerns only these new action models, not independent generators.
    for i,(tr,te) in enumerate(GroupKFold(3).split(np.zeros(soft.height),groups=groups)):
        held=set(groups[te]);train=soft[tr];normal=none.filter(~pl.col('table_id').is_in(sorted(held)));test=soft[te]
        assert not set(train['table_id'])&held and not set(normal['table_id'])&held
        m=model(train);n=model(normal);xx=test.select(INPUTS).to_numpy();pred[te]=m.predict_proba(xx);base[te]=n.predict_proba(xx)
        folds.append({'fold':i,'held_pools':sorted(held),'soft_train_pools':sorted(set(train['table_id'])),'normal_train_pools':sorted(set(normal['table_id'])),'soft_loss':float(log_loss(test['action_class'],pred[te],labels=[0,1,2,3],sample_weight=weights(test))),'normal_loss':float(log_loss(test['action_class'],base[te],labels=[0,1,2,3],sample_weight=weights(test)))})
    sw=weights(soft);y=soft['action_class'].to_numpy();row_gain=np.log(np.maximum(pred[np.arange(len(y)),y],1e-12))-np.log(np.maximum(base[np.arange(len(y)),y],1e-12))
    # Publicly observed strengths; these are feature scores, not true equities.
    thresholds={str(st):float(soft.filter((pl.col('street_index')==st)&(pl.col('action_class')==2))['own_strength'].quantile(.25)) for st in range(4)}
    m=model(soft);n=model(none);joblib.dump({'soft':m.booster_,'normal':n.booster_,'inputs':INPUTS,'minimum_call_strength':thresholds},H/'controller.joblib')
    receipt={'created_at':datetime.now(timezone.utc).isoformat(),'inputs':INPUTS,'parameters':PARAMS,'training_pools':sorted(pools),'soft_pairs':soft['pair_id'].n_unique(),'soft_actions':soft.height,'normal_actions':none.height,'folds':folds,'oof_logloss_improvement':float(np.average(row_gain,weights=sw)),'minimum_call_strength':thresholds,'source_sha256':sha(__file__),'model_sha256':sha(H/'controller.joblib'),'no_outer_validation_or_evaluation_data':True,'fit_to_all_soft_pair_actions_not_evidence_selected':True}
    (H/'controller-fit.json').write_text(json.dumps(receipt,indent=2));np.savez_compressed(H/'controller-oof.npz',y=y,p=pred,normal=base,weights=sw,groups=groups)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['folds','training_pools','inputs','parameters']},indent=2))
if __name__=='__main__':main()
