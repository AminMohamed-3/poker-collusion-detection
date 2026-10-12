"""Additional whole-player history audit, frozen before inspecting repaired outputs."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import hashlib,json,sys
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent

def feature_frame(h,s):
    clock=h.sort('table_id','started_at','hand_id').with_columns(pl.int_range(pl.len()).over('table_id').alias('time_index')).select('hand_id','table_id','big_blind','time_index')
    f=s.join(clock,on='hand_id').with_columns((pl.col('starting_stack')/pl.col('big_blind')).alias('bb'),(pl.col('net_chips')/pl.col('big_blind')).alias('net_bb')).sort('player_id','time_index')
    f=f.with_columns(pl.col('bb').shift(1).over('player_id').alias('last_bb'),pl.col('bb').shift(5).over('player_id').alias('last5_bb'),pl.col('net_bb').shift(1).over('player_id').alias('last_net'),pl.col('time_index').diff().over('player_id').alias('gap'))
    f=f.with_columns((pl.col('bb')-pl.col('last_bb')).alias('delta'),(pl.col('bb')-pl.col('last_bb')-pl.col('last_net')).abs().gt(1e-6).alias('reset'))
    out=f.group_by('table_id','player_id').agg(
        pl.len().alias('appearances'),
        *[expr.alias(name) for name,expr in {
            'stack_mean':pl.col('bb').mean(),'stack_sd':pl.col('bb').std(),'stack_min':pl.col('bb').min(),'stack_max':pl.col('bb').max(),
            'stack_lag1':pl.corr('bb','last_bb'),'stack_lag5':pl.corr('bb','last5_bb'),
            'abs_delta_mean':pl.col('delta').abs().mean(),'abs_delta_q90':pl.col('delta').abs().quantile(.9),'unchanged_fraction':(pl.col('delta')==0).mean(),
            'reset_fraction':pl.col('reset').mean(),'gap_mean':pl.col('gap').mean(),'gap_sd':pl.col('gap').std(),'gap_q90':pl.col('gap').quantile(.9),'gap_max':pl.col('gap').max(),
            'net_mean':pl.col('net_bb').mean(),'net_sd':pl.col('net_bb').std(),'net_abs_q90':pl.col('net_bb').abs().quantile(.9),
            'fold_fraction':pl.col('folded').mean(),'showdown_fraction':pl.col('went_to_showdown').mean()
        }.items()]).sort('table_id','player_id')
    return out.with_columns(pl.col(pl.NUMERIC_DTYPES).fill_nan(0).fill_null(0))

def main():
    name=sys.argv[1] if len(sys.argv)>1 else 'persistent_balance';D=H/name
    target=D/'temporal-audit.json';assert not target.exists()
    cached=H/'original-temporal-features.parquet'
    if cached.exists():original=pl.read_parquet(cached)
    else:
        original=feature_frame(pl.read_parquet(H/'training-hands.parquet'),pl.read_parquet(H/'training-seats.parquet'));original.write_parquet(cached)
    dirs=sorted((D/'pilot').glob('pool-*'))
    syn=feature_frame(pl.concat([pl.read_parquet(p/'hands.parquet') for p in dirs]),pl.concat([pl.read_parquet(p/'seats.parquet') for p in dirs]));syn.write_parquet(D/'temporal-features.parquet')
    cols=[c for c in original.columns if c not in ['table_id','player_id']]
    x=np.r_[original.select(cols).to_numpy(),syn.select(cols).to_numpy()];y=np.r_[np.zeros(original.height),np.ones(syn.height)];g=np.r_[original['table_id'].to_numpy(),syn['table_id'].to_numpy()]
    splits=list(StratifiedGroupKFold(4,shuffle=True,random_state=244017).split(x,y,g))
    result={}
    for group,selected in [('all',cols),('balance_process',[c for c in cols if c.startswith(('stack_','abs_delta','unchanged','reset_'))]),('attendance',[c for c in cols if c.startswith(('appearances','gap_'))]),('outcomes',[c for c in cols if c.startswith(('net_','fold_','showdown_'))])]:
        indices=[cols.index(c) for c in selected];pred=np.zeros(len(y));imp=np.zeros(len(indices));folds=[]
        for i,(tr,te) in enumerate(splits):
            assert not set(g[tr])&set(g[te])
            model=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=30,class_weight='balanced',learning_rate=.05,n_jobs=2,verbosity=-1,random_state=244017+i)
            model.fit(x[tr][:,indices],y[tr]);pred[te]=model.predict_proba(x[te][:,indices])[:,1];imp+=model.booster_.feature_importance(importance_type='gain');folds.append(float(roc_auc_score(y[te],pred[te])))
        result[group]={'auc':float(roc_auc_score(y,pred)),'fold_auc':folds,'top_features':[{'feature':selected[i],'gain':float(imp[i]),'original_median':float(np.median(x[y==0,indices[i]])),'synthetic_median':float(np.median(x[y==1,indices[i]]))} for i in np.argsort(-imp)[:8]]}
    r={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'original_players':original.height,'synthetic_players':syn.height,'gates':result,'all_feature_auc_ceiling':.8,'passed':result['all']['auc']<=.8,'no_original_outer_held_or_evaluation_rows':True,'note':'Additional exploratory training-only diagnostic. Calibration data reused; passing does not establish independent generalization. All-player histories include controlled actors in both sources.'}
    target.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

if __name__=='__main__':main()
