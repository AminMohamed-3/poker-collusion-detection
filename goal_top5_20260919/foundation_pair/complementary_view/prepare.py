import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
import site, json, hashlib, time
from pathlib import Path
site.addsitedir(r'C:\Data\Projects\Money Making\work\poker\.venv\Lib\site-packages')
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier

ROOT=Path(r'C:\Data\Projects\Money Making'); OUT=Path(__file__).resolve().parent
SCREEN=OUT.parent/'screen'; LAB=ROOT/'work/poker/data/development_labels.csv'
FULL=ROOT/'work/poker/Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet'
P60=ROOT/'work/poker/Claude Version/experiments/pu_pairs_20260911'
SEED=35935; SALT='foundation-pf35-screen-v1:'
FAMS={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fold_of(t):return int(hashlib.sha256((SALT+t).encode()).hexdigest()[:16],16)%2
def model(seed):
 return LGBMClassifier(objective='multiclass',num_class=4,n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=25,reg_lambda=10,colsample_bytree=.8,random_state=seed,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
def load_prefix():
 frames=[]
 for f in (1,2):
  frames.append(pl.concat([pl.read_parquet(p) for p in sorted((P60/f'fold{f}-prefix60').glob('batch_*-pairs.parquet'))],how='vertical_relaxed'))
 return pl.concat(frames,how='vertical_relaxed').unique('pair_id',keep='first',maintain_order=True)
def main():
 assert (OUT/'PROTOCOL.md').exists(); t=time.time(); OUT.mkdir(exist_ok=True)
 x=pl.read_parquet(FULL); lab=pl.read_csv(LAB)
 d=x.join(lab.select('pair_id','label','behavior_family'),on='pair_id',validate='1:1')
 assert d.height==1860 and d['label'].sum()==372
 y=np.array([FAMS[v] if z==1 else 0 for z,v in zip(d['label'],d['behavior_family'])],dtype=np.int8)
 tids=np.asarray(d['table_id'].to_list(),dtype='U'); ids=np.asarray(d['pair_id'].to_list(),dtype='U')
 folds=np.array([fold_of(v) for v in tids],dtype=np.int8)
 feat=[c for c,dt in d.schema.items() if c not in {'pair_id','table_id','label','behavior_family'} and dt.is_numeric()]
 assert len(feat)==653
 X=d.select(feat).to_numpy().astype('float32');X[~np.isfinite(X)]=np.nan
 p60=load_prefix().select(['pair_id']+feat).join(d.select('pair_id'),on='pair_id',how='inner',validate='1:1')
 pids=np.asarray(p60['pair_id'].to_list(),dtype='U'); pmap={v:i for i,v in enumerate(pids)}
 PX=p60.select(feat).to_numpy().astype('float32');PX[~np.isfinite(PX)]=np.nan
 manifest=[]
 for f in (0,1):
  tr=folds!=f;va=~tr; sm=model(SEED+f);sm.fit(X[tr],y[tr])
  imp=sm.booster_.feature_importance(importance_type='gain');order=sorted(range(len(feat)),key=lambda j:(-float(imp[j]),feat[j]))
  first=np.array(order[:128]);ix=np.array(order[128:256])
  old=np.load(SCREEN/f'context_fold{f}.npz',allow_pickle=False)
  assert np.array_equal(np.asarray([feat[j] for j in first],dtype='U'),old['feature_name']), 'regenerated original selector mismatch'
  keys=ids[va]
  np.savez_compressed(OUT/f'context_fold{f}.npz',X_train=X[tr][:,ix],y_train=y[tr],X_held=X[va][:,ix],pair_id=keys,feature_name=np.asarray([feat[j] for j in ix],dtype='U'))
  np.savez_compressed(OUT/f'evaluator_fold{f}.npz',pair_id=keys,y=y[va],table_id=tids[va])
  hidx=np.array([i for i,v in enumerate(ids) if va[i] and v in pmap]);hp=np.array([pmap[ids[i]] for i in hidx])
  np.savez_compressed(OUT/f'context_fold{f}_prefix60.npz',X_train=X[tr][:,ix],y_train=y[tr],X_held=PX[hp][:,ix],pair_id=ids[hidx],feature_name=np.asarray([feat[j] for j in ix],dtype='U'))
  np.savez_compressed(OUT/f'evaluator_fold{f}_prefix60.npz',pair_id=ids[hidx],y=y[hidx],table_id=tids[hidx])
  manifest.append({'fold':f,'selector_seed':SEED+f,'train_rows':int(tr.sum()),'held_rows':int(va.sum()),'prefix_rows':len(hidx),'original_first128_sha256':hashlib.sha256('\n'.join(feat[j] for j in first).encode()).hexdigest(),'complementary_next128_sha256':hashlib.sha256('\n'.join(feat[j] for j in ix).encode()).hexdigest(),'features':[feat[j] for j in ix]})
 receipt={'frozen_protocol_sha256':sha(OUT/'PROTOCOL.md'),'source_sha256':sha(FULL),'labels_sha256':sha(LAB),'salt':SALT,'raw_feature_count':653,'excluded':['pair_id','table_id','label','behavior_family'],'selector':'original exact fold seeds 35935+fold; ranks 129-256','folds':manifest,'seconds':time.time()-t}
 (OUT/'prep_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
