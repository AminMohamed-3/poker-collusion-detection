"""Test whether different frozen feature-reference models explain source artifacts."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys
import numpy as np
import polars as pl
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent;R=H.parents[1];P=H.parent/'softplay_coupling_20260910';sys.path.insert(0,str(R/'src'))
from run_conditional_policy_screen import hand_features
from conditional_bucket_features import bucket_features
from conditional_inference_worker import pair_aggregate
from conditional_action_features import INPUTS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (H/'results.json').exists()
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'training_only':True,'question':'Do original cross-fitted reference features versus synthetic full-training-reference features create spurious source differences?','variants':['canonical_mixed_references','both_full_training_reference','synthetic_deterministic_inner_reference'],'group_cv_seeds':[19083,27031,35027],'synthetic_fold_rule':'sha256(reference-alignment-v1:table_id) mod3','no_utility_models_or_outer_labels':True,'source_sha256':sha(Path(__file__))};(H/'plan.json').write_text(json.dumps(plan,indent=2))
    names=json.loads((P/'b0_coupled/features/metadata.json').read_text())['features'];idx=np.array([i for i,n in enumerate(names) if n.startswith(('ca_','cb_'))]);learned=[names[i] for i in idx]
    meta=json.loads((H.parent/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs'];take=np.array([r['behavior_family']=='soft_play' for r in meta]);realx=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz')['train_pair'][:1403][take];realm=[r for r in meta if r['behavior_family']=='soft_play'];real=pl.read_parquet(H.parent/'softplay_fast_20260910/training-actions.parquet').filter(pl.col('behavior_family')=='soft_play')
    pools=set(json.loads((P/'b0_coupled/training-calibration.json').read_text())['training_pools']);assert set(real['table_id'])<=pools
    models={};references={}
    for name in ['inner0','inner1','inner2','held']:
        a=R/'reports/conditional-action-experiment/policy-screen';b=R/'reports/conditional-bucket-experiment/policy-screen';ar=json.loads((a/(name+'.json')).read_text());br=json.loads((b/(name+'.json')).read_text());ap=a/(name+'-models.joblib');bp=b/(name+'.joblib');assert sha(ap)==ar['model_sha256'] and sha(bp)==br['model_sha256'];assert set(ar['training_pools'])<=pools and set(br['training_pools'])==set(ar['training_pools']);models[name]=(joblib.load(ap),joblib.load(bp));references[name]=ar
    def build(f,model):
        a,b=model;x=f.select(INPUTS).to_numpy().astype(np.float32);p=a['classifier'].predict_proba(x,num_threads=2);ca=hand_features(f,p,a['regressor'].predict(x,num_threads=2));cb=bucket_features(f,p,b['model'].predict_proba(x,num_threads=2));aa=pair_aggregate(ca,[c for c in ca.columns if c.startswith('ca_')]);bb=pair_aggregate(cb,[c for c in cb.columns if c.startswith('cb_')]);return aa.join(bb,on=['pair_id','table_id']).sort('pair_id')
    pieces=[]
    for name in ['inner0','inner1','inner2']:pieces.append(build(real.filter(pl.col('table_id').is_in(references[name]['scored_pools'])),models[name]))
    control=pl.concat(pieces).sort('pair_id');assert control['pair_id'].to_list()==[r['pair_id'] for r in realm];np.testing.assert_allclose(control.select(learned).to_numpy().astype(np.float32),realx[:,idx],rtol=1e-5,atol=1e-6)
    realfull=realx.copy();full=build(real,models['held']);assert full['pair_id'].to_list()==control['pair_id'].to_list();realfull[:,idx]=full.select(learned).to_numpy().astype(np.float32)
    syn=[];aligned=[];groups=[];coverage=[]
    for arm in ['b0_coupled','b1_coupled']:
        m=json.loads((P/arm/'features/metadata.json').read_text())['views'][0];keep=np.array([r['behavior_family']=='soft_play' for r in m]);sm=[r for r in m if r['behavior_family']=='soft_play'];xx=np.load(P/arm/'features/synthetic-pairs.npz')['x'][:len(m)][keep];f=pl.read_parquet(P/arm/'soft-actions.parquet');check=build(f,models['held']);assert check['pair_id'].to_list()==[r['pair_id'] for r in sm]
        # Quantify, rather than hide, any zero-action hand filling difference.
        delta=np.abs(check.select(learned).to_numpy().astype(np.float32)-xx[:,idx]);coverage.append({'arm':arm,'same_reference_max_feature_gap':float(delta.max())})
        np.testing.assert_allclose(check.select(learned).to_numpy().astype(np.float32),xx[:,idx],rtol=1e-5,atol=1e-6)
        parts=[]
        for k in range(3):
            selected=sorted({r['table_id'] for r in sm if int(hashlib.sha256(('reference-alignment-v1:'+r['table_id']).encode()).hexdigest(),16)%3==k})
            if selected:parts.append(build(f.filter(pl.col('table_id').is_in(selected)),models[f'inner{k}']))
        rebuilt=pl.concat(parts).sort('pair_id');assert rebuilt['pair_id'].to_list()==check['pair_id'].to_list();new=xx.copy();new[:,idx]=rebuilt.select(learned).to_numpy().astype(np.float32);syn.append(xx);aligned.append(new);groups.extend(r['table_id'] for r in sm)
    sx=np.concatenate(syn);ax=np.concatenate(aligned);g=np.array([r['table_id'] for r in realm]+groups);y=np.r_[np.zeros(len(realx)),np.ones(len(sx))]
    results={}
    for name,(ox,ss) in {'canonical_mixed_references':(realx,sx),'both_full_training_reference':(realfull,sx),'synthetic_deterministic_inner_reference':(realx,ax)}.items():
        x=np.r_[ox,ss];aucs=[]
        for seed in plan['group_cv_seeds']:
            pred=np.full(len(y),np.nan)
            for k,(tr,te) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=seed).split(x,y,g)):
                assert not set(g[tr])&set(g[te]);model=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=30,class_weight='balanced',learning_rate=.05,n_jobs=2,verbosity=-1,random_state=seed+k).fit(x[tr],y[tr]);pred[te]=model.predict_proba(x[te])[:,1]
            aucs.append(float(roc_auc_score(y,pred)))
        top={}
        for feature in ['ca_residual_0_mean_std','ca_residual_0_absmax_mean','ca_residual_0_absmax_max','cb_cross_0_3_mean_std']:
            j=names.index(feature);top[feature]={'original_mean':float(ox[:,j].mean()),'synthetic_mean':float(ss[:,j].mean())}
        results[name]={'source_auc':aucs,'mean_auc':float(np.mean(aucs)),'key_feature_means':top};print(name,aucs,flush=True)
    np.savez_compressed(H/'reference-comparison-arrays.npz',real_canonical=realx,real_full_reference=realfull,synthetic_full_reference=sx,synthetic_inner_reference=ax,source=y,groups=g.astype(str))
    out={'plan_sha256':sha(H/'plan.json'),'original_reconstruction_matches_frozen_training':True,'synthetic_same_reference_reconstruction':coverage,'reference_training_pool_counts':{k:len(v['training_pools']) for k,v in references.items()},'results':results,'no_outer_holdout_read':True,'no_generator_or_detector_modified':True,'limitation':'Changing feature references is a diagnostic, not evidence of utility. Full-reference original data include reference-training pools, so that arm is not cross-fitted and cannot replace valid training features.'};(H/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
