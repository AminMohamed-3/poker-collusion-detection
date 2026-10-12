"""Audit legal-class projection and fixed OOF policy temperatures, training only."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold
H=Path(__file__).resolve().parent;P=H.parent/'softplay_causal_style_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def project(raw,free):
    p=np.array(raw,dtype=float,copy=True)
    p[free,0]=0;p[~free,1]=0
    p[free,1]+=p[free,2];p[free,2]=0
    return p/p.sum(1,keepdims=True)
def main():
    assert not (H/'calibration.json').exists();f=pl.read_parquet(P/'training-actions-causal.parquet').filter(pl.col('behavior_family')=='soft_play');prior=np.load(P/'own_style-oof.npz');y=f['action_class'].to_numpy();g=f['table_id'].to_numpy();np.testing.assert_array_equal(y,prior['y']);np.testing.assert_array_equal(g.astype(str),prior['groups']);free=f['price'].to_numpy()==0
    assert not np.any((y==0)&free) and not np.any((y==2)&free) and not np.any((y==1)&~free)
    counts=f.group_by('pair_id').len();w=f.select('pair_id').join(counts,on='pair_id',how='left')['len'].to_numpy()**-1.*f.height/counts.height
    plan={'at':datetime.now(timezone.utc).isoformat(),'training_only':True,'oof_sha256':sha(P/'own_style-oof.npz'),'temperatures':[.7,.85,1.0,1.15],'selection':'Temperature change is eligible only if mean OOF action loss improves at least .001 versus correctly projected T1 and each3 fixed pool fold improves. Choose largest eligible gain, deterministic listed-order tie-break. A temperature probe is not generator-realism evidence.','projection':'No-debt actions can only check or raise. Combine call-class mass into check-class mass before coupling, in both ordinary and soft distributions. This is a correctness repair regardless of any chosen temperature.','source_sha256':sha(Path(__file__))};(H/'calibration-plan.json').write_text(json.dumps(plan,indent=2))
    baseline=project(prior['p'],free);base_loss=-np.log(np.maximum(baseline[np.arange(len(y)),y],1e-12));results={};splits=list(GroupKFold(3).split(np.zeros(len(y)),groups=g))
    for temp in plan['temperatures']:
        pred=project(np.maximum(prior['p'],1e-12)**(1/temp),free);loss=-np.log(np.maximum(pred[np.arange(len(y)),y],1e-12));gain=base_loss-loss;folds=[float(np.average(gain[te],weights=w[te])) for _,te in splits];mean=float(np.average(gain,weights=w));results[str(temp)]={'mean_loss_gain':mean,'fold_loss_gains':folds,'eligible':bool(temp!=1 and mean>=.001 and min(folds)>0),'weighted_logloss':float(np.average(loss,weights=w))}
    eligible=[t for t in plan['temperatures'] if results[str(t)]['eligible']];chosen=max(eligible,key=lambda t:results[str(t)]['mean_loss_gain']) if eligible else 1.0
    mass=prior['p'][free,2];result={'results':results,'selected_temperature':chosen,'free_action_rows':int(free.sum()),'predicted_call_class_mass_when_free':{'mean':float(mass.mean()),'q95':float(np.quantile(mass,.95)),'max':float(mass.max())},'projection_bug_confirmed_by_code_review':True,'interpretation':'Current coupling infers the ordinary source class as check when a free call resolves to check, but its probability vector still carries a separate call class. This breaks the claimed exact transport marginal. Prior legal/accounting checks remain valid; realism failures stay failed. Correct the projection and test resolved-action sampling, not only abstract classes.','no_outer_holdout_read':True,'no_policy_or_detector_modified':True};(H/'calibration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
