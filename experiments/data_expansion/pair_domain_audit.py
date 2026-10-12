"""Supplemental source discrimination at the detector's pair-feature level."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import hashlib,json
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'augmentation'
a=np.load(O/'synthetic/tree-arrays.npz');m=json.loads((O/'metadata.json').read_text());old=json.loads((R/'reports/mil-pair-experiment/fold0/metadata.json').read_text())
n=m['original_pairs'];sn=m['synthetic_pairs'];o=m['original_views']
x=np.r_[a['train_pair'][:n],a['train_pair'][o:o+sn]];y=np.r_[np.zeros(n),np.ones(sn)];groups=np.array([p['table_id'] for p in old['train_full']['pairs']]+[p['table_id'] for p in m['synthetic_views'][0]])
none=np.r_[a['train_y'][:n],a['train_y'][o:o+sn]]==0
x=x[none];y=y[none];groups=groups[none];p=np.full(len(y),np.nan);folds=[];importance=np.zeros(x.shape[1])
for i,(tr,te) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=19083).split(x,y,groups)):
    assert not set(groups[tr])&set(groups[te])
    model=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=30,class_weight='balanced',learning_rate=.05,n_jobs=2,verbosity=-1,random_state=19083+i)
    model.fit(x[tr],y[tr]);p[te]=model.predict_proba(x[te])[:,1];folds.append(float(roc_auc_score(y[te],p[te])));importance+=model.booster_.feature_importance(importance_type='gain')
names=json.loads((O/'features.json').read_text())['pair1217'];top=np.argsort(-importance)[:15]
result={'auc':float(roc_auc_score(y,p)),'fold_auc':folds,'original_none_pairs':int(sum(y==0)),'synthetic_none_pairs':int(sum(y==1)),'top_source_features':[{'feature':names[i],'gain':float(importance[i]),'original_median':float(np.median(x[y==0,i])),'synthetic_median':float(np.median(x[y==1,i]))} for i in top],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_held_labels_or_evaluation_rows':True,'interpretation':'Supplemental training-only diagnostic at the full pair representation. Hand-level plausibility does not imply pair-level equivalence. This does not modify the already frozen utility trial or justify claiming a useful gain.'}
(H/'learned/pair-domain-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
