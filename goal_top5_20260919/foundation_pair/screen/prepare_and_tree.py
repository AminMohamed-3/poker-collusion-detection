import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
import sys, site, json, hashlib, time
from pathlib import Path
site.addsitedir(r'C:\Data\Projects\Money Making\work\poker\.venv\Lib\site-packages')
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

ROOT=Path(r'C:\Data\Projects\Money Making')
OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/screen'
LAB=ROOT/'work/poker/data/development_labels.csv'
FULL=ROOT/'work/poker/Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet'
P60=ROOT/'work/poker/Claude Version/experiments/pu_pairs_20260911'
SEED=35935; SALT='foundation-pf35-screen-v1:'
FAMS={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}

def fold_of(t): return int(hashlib.sha256((SALT+t).encode()).hexdigest()[:16],16)%2
def model(seed):
    return LGBMClassifier(objective='multiclass',num_class=4,n_estimators=300,learning_rate=.03,
      num_leaves=15,min_child_samples=25,reg_lambda=10,colsample_bytree=.8,random_state=seed,
      n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
def ap(y,s): return float(average_precision_score(y,s))
def metrics(y,p):
    risk=1-p[:,0]; pred=1+np.argmax(p[:,1:],axis=1)
    bm=float(np.mean([ap(y==k,np.where(pred==k,risk,0.0)) for k in (1,2,3)]))
    by={str(k):ap(y==k,p[:,k]) for k in (1,2,3)}
    w=sum((y==k).sum()*by[str(k)] for k in (1,2,3))/max((y>0).sum(),1)
    pa=ap(y>0,risk)
    return {'pair_ap':pa,'behavior_map_predicted':bm,'pair_behavior':.7*pa+.1*bm,'rare_positive_weighted_ap':float(w),'family_ovr_ap':by}
def load_prefix60():
    frames=[]
    for f in (1,2):
      files=sorted((P60/f'fold{f}-prefix60').glob('batch_*-pairs.parquet'))
      frames.append(pl.concat([pl.read_parquet(x) for x in files],how='vertical_relaxed'))
    return pl.concat(frames,how='vertical_relaxed').unique('pair_id',keep='first')
def main():
    t=time.time(); OUT.mkdir(parents=True,exist_ok=True)
    x=pl.read_parquet(FULL); lab=pl.read_csv(LAB)
    # table_id is taken from raw aggregates, then labels are joined 1:1.
    d=x.join(lab.select('pair_id','label','behavior_family'),on='pair_id',validate='1:1')
    assert d.height==1860 and d['label'].to_numpy().sum()==372
    y=np.array([FAMS[v] if z==1 else 0 for z,v in zip(d['label'],d['behavior_family'])],dtype=np.int8)
    tids=d['table_id'].to_list(); folds=np.array([fold_of(v) for v in tids],dtype=np.int8)
    feat=[c for c,dt in d.schema.items() if c not in {'pair_id','table_id','label','behavior_family'} and dt.is_numeric()]
    assert len(feat)==653
    X=d.select(feat).to_numpy().astype('float32'); X[~np.isfinite(X)]=np.nan
    p128=np.zeros((len(y),4)); p653=np.zeros((len(y),4)); manifest=[]; stress={}; stress_arrays={}
    p60=load_prefix60().select(['pair_id']+feat).join(d.select('pair_id'),on='pair_id',how='inner',validate='1:1')
    # Fold1/2 prefix caches cover 813 of the confirmed rows. Evaluate exactly that
    # confirmed intersection; absence is never filled or interpreted as negative.
    ids=d['pair_id'].to_list()
    p60_ids=set(p60['pair_id'].to_list())
    p60_map={v:i for i,v in enumerate(p60['pair_id'].to_list())}
    p60_X=p60.select(feat).to_numpy().astype('float32'); p60_X[~np.isfinite(p60_X)]=np.nan
    for f in (0,1):
      tr=folds!=f; va=~tr
      selmod=model(SEED+f); selmod.fit(X[tr],y[tr])
      imp=selmod.booster_.feature_importance(importance_type='gain')
      order=sorted(range(len(feat)),key=lambda j:(-float(imp[j]),feat[j])); ix=np.array(order[:128])
      m128=model(SEED+10+f); m128.fit(X[tr][:,ix],y[tr]); p128[va]=m128.predict_proba(X[va][:,ix])
      m653=model(SEED+20+f); m653.fit(X[tr],y[tr]); p653[va]=m653.predict_proba(X[va])
      keys=np.array(d['pair_id'].to_list(),dtype='U')[va]
      np.savez_compressed(OUT/f'context_fold{f}.npz',X_train=X[tr][:,ix],y_train=y[tr],X_held=X[va][:,ix],pair_id=keys,feature_name=np.array([feat[j] for j in ix],dtype='U'))
      hidx=np.array([i for i,v in enumerate(ids) if va[i] and v in p60_ids])
      hp=np.array([p60_map[ids[int(i)]] for i in hidx])
      np.savez_compressed(OUT/f'context_fold{f}_prefix60.npz',X_train=X[tr][:,ix],y_train=y[tr],X_held=p60_X[hp][:,ix],pair_id=np.array(ids,dtype='U')[hidx],feature_name=np.array([feat[j] for j in ix],dtype='U'))
      np.savez_compressed(OUT/f'evaluator_fold{f}_prefix60.npz',pair_id=np.array(ids,dtype='U')[hidx],y=y[hidx],table_id=np.array(tids,dtype='U')[hidx])
      hp60=m128.predict_proba(p60_X[hp][:,ix]); stress_arrays[f'prefix60_fold{f}']=hp60
      stress[f'prefix60_fold{f}']=metrics(y[hidx],hp60)
      np.savez_compressed(OUT/f'evaluator_fold{f}.npz',pair_id=keys,y=y[va],table_id=np.array(tids,dtype='U')[va])
      for omit in (1,2,3):
        otr=tr & (y!=omit); ova=va & ((y==0)|(y==omit))
        los=model(SEED+200+10*f+omit); los.fit(X[otr],y[otr]); li=los.booster_.feature_importance(importance_type='gain')
        lorder=sorted(range(len(feat)),key=lambda j:(-float(li[j]),feat[j])); lix=np.array(lorder[:128])
        np.savez_compressed(OUT/f'context_fold{f}_omit{omit}.npz',X_train=X[otr][:,lix],y_train=y[otr],X_held=X[ova][:,lix],pair_id=np.array(d['pair_id'].to_list(),dtype='U')[ova],feature_name=np.array([feat[j] for j in lix],dtype='U'))
        np.savez_compressed(OUT/f'evaluator_fold{f}_omit{omit}.npz',pair_id=np.array(d['pair_id'].to_list(),dtype='U')[ova],y=y[ova],table_id=np.array(tids,dtype='U')[ova])
        mo=model(SEED+100+10*f+omit); mo.fit(X[otr][:,lix],y[otr]); po=mo.predict_proba(X[ova][:,lix])
        # The omitted class cannot be emitted. The honest stress statistic is
        # target-vs-confirmed-negative AP using non-none probability.
        risk=1-po[:,list(mo.classes_).index(0)]
        stress[f'omit{omit}_fold{f}']={'rows':int(ova.sum()),'positives':int((y[ova]==omit).sum()),'omitted_vs_confirmed_negative_ap':ap(y[ova]==omit,risk),'trained_classes':[int(v) for v in mo.classes_],'selected_features':[feat[j] for j in lix]}
        stress_arrays[f'omit{omit}_fold{f}']=po
      manifest.append({'fold':f,'train_rows':int(tr.sum()),'held_rows':int(va.sum()),'train_tables':len(set(np.array(tids)[tr])),'held_tables':len(set(np.array(tids)[va])),'selected_features':[feat[j] for j in ix]})
    np.savez_compressed(OUT/'tree_oof_predictions.npz',pair_id=np.array(d['pair_id'].to_list(),dtype='U'),fold=folds,lgb128=p128,lgb653=p653)
    np.savez_compressed(OUT/'tree_stress_predictions.npz',**stress_arrays)
    result={'scope':'curated known-label screen; not STEP272 comparison','metrics':{'LGB128':metrics(y,p128),'LGB653':metrics(y,p653)},'fold_metrics':{str(f):{'LGB128':metrics(y[folds==f],p128[folds==f]),'LGB653':metrics(y[folds==f],p653[folds==f])} for f in (0,1)},'stress':stress,'seconds':time.time()-t}
    (OUT/'tree_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    players=dict(zip(lab['pair_id'].to_list(),zip(lab['player_1'].to_list(),lab['player_2'].to_list())))
    pfold=[set(v for i,pid in enumerate(ids) if folds[i]==f for v in players[pid]) for f in (0,1)]
    receipt={'seed':SEED,'salt':SALT,'feature_count':len(feat),'selected_count':128,'classes':FAMS,'raw_only_proof':{'source':str(FULL),'excluded':['pair_id','table_id','label','behavior_family'],'no_score_columns':not any(c in {'risk','p0','p1','p2','p3','predicted_class'} for c in feat)},'history60_source':'fold1-prefix60 + fold2-prefix60 raw batch pair parquets; score parquets ignored','history60_confirmed_intersection':len(p60_ids),'player_overlap_crossfold':len(pfold[0]&pfold[1]),'table_overlap_crossfold':len(set(np.array(tids)[folds==0])&set(np.array(tids)[folds==1])),'lofo_feature_selection':'inside otr; omitted family excluded from selector and classifier','folds':manifest,'seconds':time.time()-t}
    (OUT/'prep_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
if __name__=='__main__': main()
