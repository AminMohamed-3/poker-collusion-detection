"""Training-only soft-play action gaps and grouped source classification."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from collections import defaultdict
from pathlib import Path
import json,sys,hashlib
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from diagnose_actions import summary
H=Path(__file__).resolve().parent;R=H.parents[1];D=H/'repaired';sys.path.insert(0,str(R/'src'))
from conditional_inference_worker import extract_chunk
from conditional_action_features import action_class
def main():
    frames=[]
    for p in sorted((D/'pilot').glob('pool-*')):
        lab=pl.read_csv(p/'development_labels.csv').filter(pl.col('behavior_family')=='soft_play')
        if not lab.height:continue
        h=pl.read_parquet(p/'hands.parquet').with_columns(pl.col('started_at').cast(pl.String));s=pl.read_parquet(p/'seats.parquet');a=pl.read_parquet(p/'actions.parquet')
        counts=defaultdict(lambda:np.zeros(4,dtype=np.int64))
        for row in a.iter_rows(named=True):counts[row['player_id']][action_class(row)]+=1
        keys=s.select('hand_id','player_id').join(lab,left_on='player_id',right_on='player_1').join(s.select('hand_id',pl.col('player_id').alias('player_2')),on=['hand_id','player_2']).select('pair_id','hand_id','player_id','player_2').rename({'player_id':'player_1'})
        f=extract_chunk(keys,h,s,a,counts).with_columns(pl.lit(json.loads((p/'provenance.json').read_text())['spec']['soft_variant']).alias('soft_variant'))
        frames.append(f)
    actions=pl.concat(frames);actions.write_parquet(H/'new-soft-actions.parquet')
    sm=json.loads((D/'features/metadata.json').read_text())['views'][0];syn=np.load(D/'features/synthetic-pairs.npz')['x'][:len(sm)];keep=np.array([r['behavior_family']=='soft_play' for r in sm]);syn=syn[keep];sg=np.array([r['table_id'] for r in sm])[keep]
    om=json.loads((H.parent/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs'];orig=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz')['train_pair'][:len(om)];ok=np.array([r['behavior_family']=='soft_play' for r in om]);orig=orig[ok];og=np.array([r['table_id'] for r in om])[ok]
    x=np.r_[orig,syn];y=np.r_[np.zeros(len(orig)),np.ones(len(syn))];g=np.r_[og,sg];pred=np.full(len(y),np.nan);folds=[]
    for i,(tr,te) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=19083).split(x,y,g)):
        assert not set(g[tr])&set(g[te])
        m=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=30,class_weight='balanced',learning_rate=.05,n_jobs=2,verbosity=-1,random_state=19083+i).fit(x[tr],y[tr])
        pred[te]=m.predict_proba(x[te])[:,1];folds.append(float(roc_auc_score(y[te],pred[te])))
    old=json.loads((H/'action-diagnosis.json').read_text())['results'];new=summary(actions)
    gaps={}
    for context,kind in [('preflop',2),('postflop',1)]:
        key='equal_pair_mean_fold_check_call_raise';real=old['soft_play'][context][key][kind];previous=old['old_synthetic_soft_play'][context][key][kind];now=new[context][key][kind]
        gaps[context]={'original':real,'old_synthetic':previous,'repaired':now,'gap_before':abs(real-previous),'gap_after':abs(real-now),'improved':abs(real-now)<abs(real-previous)}
    auc=float(roc_auc_score(y,pred));oldauc=json.loads((H.parent/'history_augmentation_20260910/class-source-diagnostic.json').read_text())['results']['soft_play']['source_auc'];audit=json.loads((D/'pilot-audit.json').read_text())
    result={'training_only':True,'original_soft_pairs':len(orig),'new_soft_pairs':len(syn),'source_auc':auc,'old_source_auc':oldauc,'fold_auc':folds,'gaps':gaps,'new_action_summary':new,'by_variant':{str(v):summary(actions.filter(pl.col('soft_variant')==v)) for v in range(4)},'realism_gate_passed':bool(auc<=.85 and auc<oldauc and all(r['improved'] for r in gaps.values()) and audit['positive_pairs']>=12 and audit['policy_families']==4),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':['Only 12 new and 10 old synthetic soft pairs; uncertainty is substantial.','Old and new simulations use different seeds, so this is not a paired generator ablation.','Repeated source diagnostics are development checks, not independent confirmation.']}
    (H/'realism.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['new_action_summary','by_variant']},indent=2))
if __name__=='__main__':main()
