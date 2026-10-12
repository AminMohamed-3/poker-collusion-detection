"""Training-only, class-conditional source diagnostics; never a utility gate."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import hashlib,json
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from prepare_trial import dataset,LABELS
H=Path(__file__).resolve().parent; OLD=H.parent/'history_realism_20260910'
original=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz')
meta=json.loads((H/'analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs']
views,sm=dataset([OLD/'persistent_sessions',OLD/'session_confirmation'])
names=json.loads((OLD/'persistent_sessions/features/metadata.json').read_text())['features']
x=np.r_[original['train_pair'][:1403],views[0]]
source=np.r_[np.zeros(1403),np.ones(len(sm[0]))]
labels=np.r_[original['train_y'][:1403],[LABELS[r['behavior_family']] for r in sm[0]]]
groups=np.array([r['table_id'] for r in meta]+[r['table_id'] for r in sm[0]])
results={}
for name,value in LABELS.items():
    take=labels==value;xx=x[take];y=source[take];g=groups[take]
    pred=np.full(len(y),np.nan);gain=np.zeros(x.shape[1]);folds=[]
    for i,(tr,te) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=19083).split(xx,y,g)):
        assert not set(g[tr])&set(g[te]) and len(set(y[tr]))==len(set(y[te]))==2
        m=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=30,class_weight='balanced',learning_rate=.05,n_jobs=2,verbosity=-1,random_state=19083+i)
        m.fit(xx[tr],y[tr]);pred[te]=m.predict_proba(xx[te])[:,1]
        folds.append(float(roc_auc_score(y[te],pred[te])));gain+=m.booster_.feature_importance(importance_type='gain')
    results[name]={'source_auc':float(roc_auc_score(y,pred)),'fold_auc':folds,'original_pairs':int(sum(y==0)),'synthetic_pairs':int(sum(y==1)),
        'top_features':[{'feature':names[i],'original_median':float(np.median(xx[y==0,i])),'synthetic_median':float(np.median(xx[y==1,i]))} for i in np.argsort(-gain)[:8]]}
out=H/'class-source-diagnostic.json';assert not out.exists()
out.write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results,'original_training_only':True,'exploratory_not_utility_or_admission':True,'limitations':'Few synthetic positive pairs; related policy sources; class-conditional discrimination does not establish label error or downstream harm.'},indent=2))
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='top_features'} for k,v in results.items()},indent=2))
