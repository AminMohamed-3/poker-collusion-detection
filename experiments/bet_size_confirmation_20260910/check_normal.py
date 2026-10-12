"""Protect ordinary-pair realism when changing ordinary raise sizing."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
H=Path(__file__).resolve().parent
def main():
    original=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz')['train_pair'][:1403];om=json.loads((H.parent/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs'];keep=np.array([r['behavior_family']=='none' for r in om]);ox=original[keep];og=np.array([r['table_id'] for r in om])[keep];results={}
    for name,d in [('b1_control',H.parent/'softplay_capacity_20260910/b1_capacity'),('b1_sized',H/'b1_sized'),('fresh_sized',H/'fresh_sized')]:
        m=json.loads((d/'features/metadata.json').read_text())['views'][0];keep=np.array([r['behavior_family']=='none' for r in m]);sx=np.load(d/'features/synthetic-pairs.npz')['x'][:len(m)][keep];sg=np.array([r['table_id'] for r in m])[keep]
        x=np.r_[ox,sx];y=np.r_[np.zeros(len(ox)),np.ones(len(sx))];g=np.r_[og,sg];pred=np.full(len(y),np.nan)
        for k,(tr,te) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=19083).split(x,y,g)):
            assert not set(g[tr])&set(g[te]);model=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=30,class_weight='balanced',learning_rate=.05,n_jobs=2,verbosity=-1,random_state=19083+k).fit(x[tr],y[tr]);pred[te]=model.predict_proba(x[te])[:,1]
        auc=float(roc_auc_score(y,pred));results[name]={'source_auc':auc,'real_pairs':len(ox),'synthetic_pairs':len(sx),'passes_ceiling':auc<=.85};np.savez_compressed(H/(name+'-normal-source-oof.npz'),pred=pred,source=y,groups=g.astype(str))
    (H/'normal-diagnostic.json').write_text(json.dumps({'training_only':True,'results':results},indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
