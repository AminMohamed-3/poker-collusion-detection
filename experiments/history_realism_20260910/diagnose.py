"""Training-only diagnosis of source artifacts; no held or evaluation scoring."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='2'
from pathlib import Path
import hashlib,json
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier

H=Path(__file__).resolve().parent; R=H.parents[1]; D=H.parent/'data_expansion'

def quant(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(zip(['min','p01','p10','median','p90','p99','max'],np.quantile(x,[0,.01,.1,.5,.9,.99,1]).tolist()))

def corr(x,y):
    x=np.asarray(x);y=np.asarray(y);valid=np.isfinite(x)&np.isfinite(y)
    return float(np.corrcoef(x[valid],y[valid])[0,1])

def history_stats(h,s):
    f=s.join(h.select('hand_id','table_id','big_blind','started_at'),on='hand_id').with_columns((pl.col('starting_stack')/pl.col('big_blind')).alias('bb')).sort('player_id','started_at','hand_id')
    f=f.with_columns(pl.col('bb').mean().over('player_id').alias('player_mean'),pl.col('bb').mean().over('table_id').alias('pool_mean'),pl.col('bb').mean().over('hand_id').alias('hand_mean'))
    f=f.with_columns((pl.col('bb')-pl.col('player_mean')).alias('residual'))
    f=f.with_columns(pl.col('residual').shift(1).over('player_id').alias('prior_residual'),pl.col('net_chips').shift(1).over('player_id').alias('prior_net'),pl.col('bb').shift(1).over('player_id').alias('prior_bb'))
    ps=f.group_by('table_id','player_id').agg(pl.col('bb').min().alias('low'),pl.col('bb').max().alias('high'),pl.col('bb').mean().alias('mean'),pl.col('bb').std().alias('std'),pl.len().alias('hands'))
    pool=ps.group_by('table_id').agg(pl.col('mean').std().alias('between_player_mean_std'),pl.col('low').std().alias('between_player_low_std'),pl.col('high').std().alias('between_player_high_std'))
    # Every pair at a table shares a hand-level stack environment. Correlate
    # demeaned player stacks within each hand, without using actions or labels.
    a=f.select('hand_id','player_id','residual')
    pairs=a.join(a,on='hand_id',suffix='_b').filter(pl.col('player_id')<pl.col('player_id_b'))
    result={'hands':h.height,'players':ps.height,'stack_bb':quant(f['bb']),'player_stack_min':quant(ps['low']),'player_stack_max':quant(ps['high']),'player_stack_sd':quant(ps['std']),'player_hands':quant(ps['hands']),'pool_player_mean_sd':quant(pool['between_player_mean_std']),'pool_player_min_sd':quant(pool['between_player_low_std']),'pool_player_max_sd':quant(pool['between_player_high_std']),'own_lag_stack_residual_corr':corr(f['residual'],f['prior_residual']),'own_stack_residual_prior_net_corr':corr(f['residual'],f['prior_net']),'same_hand_demeaned_stack_corr':corr(pairs['residual'],pairs['residual_b']),'integer_bb_fraction':float(((f['bb']%1)==0).mean()),'half_bb_fraction':float(((f['bb']%.5)==0).mean())}
    return result,ps,f

def source_ablation():
    original=np.load(D/'augmentation/synthetic/tree-arrays.npz');syn=np.load(D/'profile_repair/features/synthetic-pairs.npz')
    meta=json.loads((D/'profile_repair/features/metadata.json').read_text());old=json.loads((R/'reports/mil-pair-experiment/fold0/metadata.json').read_text())
    n=len(old['train_full']['pairs']);sn=len(meta['views'][0]);names=meta['features']
    x=np.r_[original['train_pair'][:n],syn['x'][:sn]];y=np.r_[np.zeros(n),np.ones(sn)]
    g=np.array([p['table_id'] for p in old['train_full']['pairs']]+[p['table_id'] for p in meta['views'][0]])
    mask=np.r_[original['train_y'][:n],syn['y'][:sn]]==0;x=x[mask];y=y[mask];g=g[mask]
    groups={'all':list(range(len(names))),'stack_only':[i for i,n in enumerate(names) if n.startswith('stack_bb')],'without_stacks':[i for i,n in enumerate(names) if not n.startswith('stack_bb')],'without_stacks_and_count':[i for i,n in enumerate(names) if not n.startswith('stack_bb') and n!='shared_hands'],'conditional_only':[i for i,n in enumerate(names) if n.startswith(('ca_','cb_'))]}
    results={}
    splits=list(StratifiedGroupKFold(4,shuffle=True,random_state=19083).split(x,y,g))
    for name,cols in groups.items():
        pred=np.zeros(len(y));imp=np.zeros(len(cols))
        for i,(tr,te) in enumerate(splits):
            m=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=30,class_weight='balanced',learning_rate=.05,n_jobs=2,verbosity=-1,random_state=19083+i)
            m.fit(x[tr][:,cols],y[tr]);pred[te]=m.predict_proba(x[te][:,cols])[:,1];imp+=m.booster_.feature_importance(importance_type='gain')
        top=np.argsort(-imp)[:10]
        results[name]={'auc':float(roc_auc_score(y,pred)),'features':len(cols),'top_features':[{'name':names[cols[i]],'gain':float(imp[i]),'original':quant(x[y==0,cols[i]]),'synthetic':quant(x[y==1,cols[i]])} for i in top]}
        print(name,results[name]['auc'],flush=True)
    return results

def main():
    assert not (H/'diagnosis.json').exists()
    pools=json.loads((D/'training-calibration.json').read_text())['training_pools']
    h=pl.read_parquet(R/'data/hands.parquet').filter((pl.col('phase')=='development')&pl.col('table_id').is_in(pools))
    s=pl.scan_parquet(R/'data/seats.parquet').join(h.lazy().select('hand_id'),on='hand_id',how='semi').collect()
    # Cache only eligible original development observations for later mechanistic checks.
    h.write_parquet(H/'training-hands.parquet');s.write_parquet(H/'training-seats.parquet')
    original,ps,joined=history_stats(h,s);ps.write_parquet(H/'training-player-stack-stats.parquet')
    dirs=sorted((D/'profile_repair/pilot').glob('pool-*'))
    sh=pl.concat([pl.read_parquet(p/'hands.parquet') for p in dirs]);ss=pl.concat([pl.read_parquet(p/'seats.parquet') for p in dirs])
    synthetic,sps,sjoined=history_stats(sh,ss);sps.write_parquet(H/'synthetic-player-stack-stats.parquet')
    # Does pool context explain stack co-occurrence? Compare pair overlap in
    # real pools against independent player prototypes used in the simulator.
    allpairs=ps.select('table_id','player_id','low','high','mean').join(ps.select('table_id','player_id','low','high','mean'),on='table_id',suffix='_b').filter(pl.col('player_id')<pl.col('player_id_b'))
    pairs=sps.select('table_id','player_id','low','high','mean').join(sps.select('table_id','player_id','low','high','mean'),on='table_id',suffix='_b').filter(pl.col('player_id')<pl.col('player_id_b'))
    similarity={}
    for name,f in [('original',allpairs),('synthetic',pairs)]:
        similarity[name]={'mean_gap':quant((f['mean']-f['mean_b']).abs()),'upper_bound_gap':quant((f['high']-f['high_b']).abs()),'lower_bound_gap':quant((f['low']-f['low_b']).abs())}
    result={'original':original,'synthetic':synthetic,'within_pool_player_similarity':similarity,'ablations':source_ablation(),'training_pools':pools,'no_outer_held_or_evaluation_data':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (H/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['ablations','training_pools']},indent=2),flush=True)

if __name__=='__main__':main()
