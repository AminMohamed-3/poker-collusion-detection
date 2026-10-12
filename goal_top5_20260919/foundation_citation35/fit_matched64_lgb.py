import os,sys,time,json,hashlib,site
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[k]='2'
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import numpy as np,lightgbm as lgb
O=Path(__file__).resolve().parent;z=np.load(O/'context.npz');X=np.ascontiguousarray(np.nan_to_num(z['xfull'],nan=0.,posinf=0.,neginf=0.));y=np.ascontiguousarray(z['yfull'].astype(int));Q=np.ascontiguousarray(np.nan_to_num(z['xquery'],nan=0.,posinf=0.,neginf=0.))
PAR=dict(objective='binary',n_estimators=400,num_leaves=15,max_depth=4,min_child_samples=30,learning_rate=.05,subsample=.8,subsample_freq=1,colsample_bytree=.5,n_jobs=2,verbosity=-1)
seeds=(11,23,37,51,67,79,93,107);p=np.zeros(len(Q));t=time.perf_counter()
for sd in seeds:p+=lgb.LGBMClassifier(random_state=sd,**PAR).fit(X,y).predict_proba(Q)[:,1]
p/=len(seeds);np.save(O/'matched64_lgb.npy',p);r={'rows':len(X),'positives':int(y.sum()),'features':X.shape[1],'query_rows':len(Q),'seeds':seeds,'params':PAR,'seconds':time.perf_counter()-t,'sha256':hashlib.sha256((O/'matched64_lgb.npy').read_bytes()).hexdigest()};(O/'matched64_lgb_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
