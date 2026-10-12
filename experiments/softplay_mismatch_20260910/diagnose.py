"""Explain remaining generator mismatch using original training pools only."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent;R=H.parents[1];P=H.parent/'softplay_replication_20260910'
sys.path.insert(0,str(H.parent/'softplay_fast_20260910'))
from diagnose_actions import summary
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (H/'diagnosis.json').exists()
    ref=H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz'
    original=np.load(ref)['train_pair'][:1403]
    om=json.loads((H.parent/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs']
    pools=set(json.loads((P/'b0_calibrated/training-calibration.json').read_text())['training_pools']);assert len(pools)==297 and {r['table_id'] for r in om}==pools
    names=json.loads((P/'b0_calibrated/features/metadata.json').read_text())['features'];datasets={}
    for arm in ['b0_calibrated','b1_calibrated']:
        m=json.loads((P/arm/'features/metadata.json').read_text())['views'][0];x=np.load(P/arm/'features/synthetic-pairs.npz')['x'][:len(m)];datasets[arm]=(x,m)
    results={}
    for behavior in ['soft_play','none']:
        keep=np.array([r['behavior_family']==behavior for r in om]);ox=original[keep];og=np.array([r['table_id'] for r in om])[keep];xx=[];gg=[]
        for arm,(x,m) in datasets.items():
            take=np.array([r['behavior_family']==behavior for r in m]);xx.append(x[take]);gg.extend(r['table_id'] for r in m if r['behavior_family']==behavior)
        sx=np.concatenate(xx);x=np.r_[ox,sx];g=np.r_[og,gg];y=np.r_[np.zeros(len(ox)),np.ones(len(sx))];gains=[];oofs=[];aucs=[]
        for seed in [19083,27031,35027]:
            pred=np.full(len(y),np.nan)
            for fold,(tr,te) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=seed).split(x,y,g)):
                assert not set(g[tr])&set(g[te])
                model=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=30,class_weight='balanced',learning_rate=.05,n_jobs=2,verbosity=-1,random_state=seed+fold).fit(x[tr],y[tr]);pred[te]=model.predict_proba(x[te])[:,1];gains.append(model.booster_.feature_importance('gain'))
            aucs.append(float(roc_auc_score(y,pred)));oofs.append(pred)
        importance=np.mean(gains,axis=0);order=np.argsort(-importance);features=[]
        for j in order[:40]:
            scale=max(float(ox[:,j].std()),1e-9)
            features.append({'name':names[j],'gain':float(importance[j]),'real_mean':float(ox[:,j].mean()),'real_median':float(np.median(ox[:,j])),'real_std':float(ox[:,j].std()),'synthetic_mean':float(sx[:,j].mean()),'synthetic_median':float(np.median(sx[:,j])),'standardized_mean_gap':float((sx[:,j].mean()-ox[:,j].mean())/scale),'synthetic_batch_means':[float(v[:,j].mean()) for v in xx]})
        np.savez_compressed(H/(behavior+'-source-oof.npz'),pred=np.stack(oofs),source=y,groups=g.astype(str),feature_gains=np.stack(gains))
        results[behavior]={'original_pairs':len(ox),'synthetic_pairs':len(sx),'source_auc_three_group_splits':aucs,'mean_auc':float(np.mean(aucs)),'important_features':features}
        print(behavior,'source AUCs',aucs,flush=True)
        print(json.dumps(features[:15],indent=2),flush=True)
    real=pl.read_parquet(H.parent/'softplay_fast_20260910/training-actions.parquet');assert set(real['table_id'])<=pools
    contexts={'real_soft':summary(real.filter(pl.col('behavior_family')=='soft_play')),'real_none':summary(real.filter(pl.col('behavior_family')=='none'))}
    for arm in datasets:contexts[arm]=summary(pl.read_parquet(P/arm/'soft-actions.parquet'))
    # Equal-pair joint context/action moments; public and own-card quantities only.
    joint={}
    frames={'real_soft':real.filter(pl.col('behavior_family')=='soft_play'),**{a:pl.read_parquet(P/a/'soft-actions.parquet') for a in datasets}}
    for name,frame in frames.items():
        entries=[]
        for street in range(4):
            for action in range(4):
                f=frame.filter((pl.col('street_index')==street)&(pl.col('action_class')==action))
                if not f.height:continue
                p=f.group_by('pair_id').agg(pl.col('own_strength').mean(),pl.col('price').mean(),pl.col('players_active').mean(),pl.col('facing_partner').mean(),pl.len().alias('actions'))
                entries.append({'street':street,'action':action,'pairs':p.height,**{k:float(p[k].mean()) for k in ['own_strength','price','players_active','facing_partner','actions']}})
        joint[name]=entries
    output={'created_at':datetime.now(timezone.utc).isoformat(),'training_only':True,'original_training_pools':297,'no_outer_held_labels_or_scores':True,'diagnostic_not_new_utility_trial':True,'results':results,'action_contexts':contexts,'joint_action_moments':joint,'source_sha256':sha(Path(__file__)),'limitations':'Exploratory feature attribution with correlated inputs. Importance does not establish causality. Related simulation policies share original training information.'}
    (H/'diagnosis.json').write_text(json.dumps(output,indent=2))
if __name__=='__main__':main()
