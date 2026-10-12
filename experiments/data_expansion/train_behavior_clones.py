"""Fit ordinary-action/size policies using confirmed non-target training pools only."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import hashlib,json
import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier,LGBMRegressor

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
INPUTS=['street_index','own_strength','own_high_rank','own_low_rank','own_pair','own_suited','own_flush_count','own_straight_window','board_high_rank','board_pairs','board_flush_count','position_from_button','price','log_pot_bb','log_stack_bb','players_active','prior_actions','own_prior_actions','own_prior_aggression','own_street_aggression','log_own_prior_amount_bb',*['own_style_'+x for x in ['fold','check','call','raise']]]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dest=HERE/'learned';dest.mkdir(exist_ok=True);assert not (dest/'models.joblib').exists()
    ref=ROOT/'reports/conditional-action-experiment';old=json.loads((ref/'policy-plan.json').read_text());cal=json.loads((HERE/'training-calibration.json').read_text());pools=set(cal['training_pools'])
    paths=sorted((ref/'action-cache').glob('*.parquet'))
    assert all(sha(p)==old['files'][p.relative_to(ref).as_posix()] for p in paths)
    labels=pl.read_csv(ROOT/'data/development_labels.csv').filter((pl.col('label')==0)&(pl.col('label_status')=='confirmed_non_target')).select('pair_id')
    frame=pl.read_parquet(paths).join(labels,on='pair_id',how='semi').filter(pl.col('table_id').is_in(sorted(pools))).select('table_id','pair_id','hand_id','direction','action_no','action_class','observed_log_amount_bb',*INPUTS)
    # The same acting player can occur through multiple labelled pairs. Own/public
    # fields and target must agree, and duplicate action observations count once.
    frame=frame.unique(subset=['hand_id','action_no'],keep='first').with_columns(pl.col('hand_id').hash(seed=50191).alias('_sample')).sort('_sample','action_no').drop('_sample')
    names=sorted(pools);rng=np.random.default_rng(50191);rng.shuffle(names);folds=np.array_split(names,4)
    models=[];receipts=[]
    styles=frame.select([c for c in INPUTS if c.startswith('own_style')]).unique().to_numpy()
    for i in range(4):
        train=frame.filter(~pl.col('table_id').is_in(folds[i].tolist())).head(150000)
        test=frame.filter(pl.col('table_id').is_in(folds[i].tolist()))
        assert not set(train['table_id'])&set(test['table_id'])
        params=dict(n_estimators=140,learning_rate=.05,num_leaves=[15,23,31,15][i],min_child_samples=80,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=50191+i,deterministic=True,force_col_wise=True)
        x=train.select(INPUTS).to_numpy();y=train['action_class'].to_numpy()
        classifier=LGBMClassifier(**params).fit(x,y)
        mask=y==3;regressor=LGBMRegressor(**params,objective='quantile',alpha=.5).fit(x[mask],train['observed_log_amount_bb'].to_numpy()[mask])
        from sklearn.metrics import log_loss
        loss=float(log_loss(test['action_class'].to_numpy(),classifier.predict_proba(test.select(INPUTS).to_numpy()),labels=[0,1,2,3]))
        models.append({'classifier':classifier.booster_,'regressor':regressor.booster_})
        receipts.append({'family':i,'training_pools':sorted(set(train['table_id'])),'inner_scored_pools':sorted(set(test['table_id'])),'training_actions':train.height,'inner_actions':test.height,'inner_log_loss':loss})
        print(json.dumps({k:v for k,v in receipts[-1].items() if not k.endswith('pools')}),flush=True)
    joblib.dump({'models':models,'inputs':INPUTS,'styles':styles},dest/'models.joblib')
    receipt={'source_sha256':sha(__file__),'models_sha256':sha(dest/'models.joblib'),'training_only':True,'confirmed_negative_actions':frame.height,'outer_held_excluded':True,'families':receipts,'limitation':'Four bootstrapped/leave-partition-out behavior clones are related policies, not four independent sources or exact organizer agents.'}
    (dest/'fit.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
