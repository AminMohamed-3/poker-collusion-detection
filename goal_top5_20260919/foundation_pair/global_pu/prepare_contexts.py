import hashlib, json, os, site
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','POLARS_MAX_THREADS'): os.environ[key]='2'
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import numpy as np, polars as pl
from lightgbm import LGBMClassifier

R=Path(r'C:/Data/Projects/Money Making/work/poker'); CV=R/'Claude Version'; E=CV/'experiments/pu_pairs_20260911'; O=R/'goal_top5_20260919/foundation_pair/global_pu'; C=O/'contexts'; C.mkdir(exist_ok=True)
LAB=R/'data/development_labels.csv'; FULL=CV/'experiments/crossphase_20260911/lab_on_dev.parquet'; SEED=35935; SHARD=4096
FAMS={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selector(seed): return LGBMClassifier(objective='multiclass',num_class=4,n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=25,reg_lambda=10,colsample_bytree=.8,random_state=seed,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)

lab=pl.read_csv(LAB); full=pl.read_parquet(FULL); d=full.join(lab.select('pair_id','player_1','player_2','label','behavior_family'),on='pair_id',validate='1:1'); assert d.height==1860
feat=[c for c,dt in full.schema.items() if c not in {'pair_id','table_id'} and dt.is_numeric()]; assert len(feat)==653
y=np.array([FAMS[v] if z==1 else 0 for z,v in zip(d['label'],d['behavior_family'])],dtype=np.int8); X=d.select(feat).to_numpy().astype(np.float32);X[~np.isfinite(X)]=np.nan
receipt={'seed':SEED,'features_total':653,'selected':128,'shard_rows':SHARD,'source_hashes':{str(LAB):sha(LAB),str(FULL):sha(FULL)},'folds':[]}
for f in (1,2):
 scores=pl.read_parquet(E/f'fold{f}-prefix60/scores.parquet').filter(pl.col('shared_hands')>=38).sort('pair_id')
 held_tables=set(scores['table_id']); held_players=set(scores['player_1'])|set(scores['player_2']); tr=np.array([t not in held_tables for t in d['table_id']],bool)
 train_players=set(d.filter(pl.Series(tr))['player_1'])|set(d.filter(pl.Series(tr))['player_2']); assert not (held_players&train_players) and not (held_tables&set(d.filter(pl.Series(tr))['table_id']))
 sm=selector(SEED+f).fit(X[tr],y[tr]);gain=sm.booster_.feature_importance(importance_type='gain');order=sorted(range(len(feat)),key=lambda j:(-float(gain[j]),feat[j]));ix=np.array(order[:128]);chosen=[feat[j] for j in ix]
 raw=pl.concat([pl.read_parquet(p,columns=['pair_id','table_id']+chosen) for p in sorted((E/f'fold{f}-prefix60').glob('batch_*-pairs.parquet'))],how='vertical_relaxed').sort('pair_id')
 held=scores.select('pair_id','table_id').join(raw,on=['pair_id','table_id'],how='left',validate='1:1',maintain_order='left');assert held.height==scores.height and held.select('pair_id','table_id').equals(scores.select('pair_id','table_id'))
 Xh=held.select(chosen).to_numpy().astype(np.float32);Xh[~np.isfinite(Xh)]=np.nan
 np.savez_compressed(C/f'train_fold{f}.npz',X_train=X[tr][:,ix],y_train=y[tr],feature_name=np.array(chosen,dtype='U'))
 shards=[]
 for n,start in enumerate(range(0,len(Xh),SHARD)):
  dest=C/f'query_fold{f}_{n:03}.npz';np.savez_compressed(dest,X_held=Xh[start:start+SHARD],pair_id=held['pair_id'].to_numpy()[start:start+SHARD].astype('U'),table_id=held['table_id'].to_numpy()[start:start+SHARD].astype('U'));shards.append({'file':dest.name,'rows':min(SHARD,len(Xh)-start),'sha256':sha(dest)})
 # Truth is local only and is deliberately outside vm_bundle.
 ev=scores.select('pair_id','table_id').join(lab.select('pair_id','label','behavior_family'),on='pair_id',how='left',validate='1:1',maintain_order='left')
 np.savez_compressed(O/f'evaluator_fold{f}.npz',pair_id=ev['pair_id'].to_numpy().astype('U'),table_id=ev['table_id'].to_numpy().astype('U'),known=ev['label'].is_not_null().to_numpy(),label=ev['label'].fill_null(-1).to_numpy().astype(np.int8),family=np.array([FAMS.get(v,-1) for v in ev['behavior_family'].to_list()],dtype=np.int8))
 receipt['folds'].append({'fold':f,'held_rows':len(Xh),'held_tables':len(held_tables),'train_rows':int(tr.sum()),'train_positive':int((y[tr]>0).sum()),'train_tables':d.filter(pl.Series(tr))['table_id'].n_unique(),'table_overlap':0,'player_overlap':0,'features':chosen,'train_sha256':sha(C/f'train_fold{f}.npz'),'shards':shards})
(O/'context_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'folds':[{k:v for k,v in x.items() if k not in ('features','shards')} for x in receipt['folds']]},indent=2))
