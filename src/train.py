"""Reproducible pool-held-out baseline and evidence classifier."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT/'data', ROOT/'outputs'
FAMILIES = ['none','directed_transfer','soft_play','coordinated_isolation']
LABELS = {s:i for i,s in enumerate(FAMILIES)}


def matrix(frame, columns):
    return np.nan_to_num(frame.select(columns).to_numpy().astype(np.float32), nan=0, posinf=0, neginf=0)


def classifier(trees=800, hand=False):
    return lgb.LGBMClassifier(n_estimators=trees, learning_rate=.03,
        num_leaves=15 if not hand else 23, max_depth=-1,
        min_child_samples=20 if not hand else 40, colsample_bytree=.8,
        reg_lambda=5, n_jobs=4, verbosity=-1, random_state=7329,
        deterministic=True, force_col_wise=True)


def train():
    labels = pl.read_csv(DATA/'development_labels.csv')
    pairdata = pl.read_parquet(OUT/'pairs_development.parquet').join(labels, on='pair_id', how='inner')
    features = [c for c in pairdata.columns if c not in ['pair_id','table_id','player_1','player_2','label','label_status','behavior_family']]
    X = matrix(pairdata, features)
    y = np.array([LABELS[x] for x in pairdata['behavior_family']])
    groups = pairdata['table_id'].to_numpy()
    results, iterations = [], []
    splits = list(GroupShuffleSplit(n_splits=3, test_size=.25, random_state=2901).split(X,y,groups))
    for fold,(itr,iva) in enumerate(splits):
        assert not set(groups[itr]) & set(groups[iva])
        model = classifier()
        model.fit(X[itr],y[itr],eval_set=[(X[iva],y[iva])],callbacks=[lgb.early_stopping(60, verbose=False)])
        prob = model.predict_proba(X[iva])
        pred = np.argmax(prob[:,1:],axis=1)+1
        risk = 1-prob[:,0]
        metrics = {'fold':fold,'train_pairs':len(itr),'validation_pairs':len(iva),
            'pair_ap':float(average_precision_score(y[iva]>0,risk)),
            'behavior_map':float(np.mean([average_precision_score(y[iva]==k,risk*(pred==k)) for k in [1,2,3]])),
            'best_iteration':int(model.best_iteration_)}
        print(json.dumps(metrics),flush=True)
        results.append(metrics); iterations.append(model.best_iteration_)
        if fold==0:
            held_tables=set(groups[iva])
            held_predictions=pairdata[iva.tolist()].select('pair_id','behavior_family').with_columns(
                pl.Series('risk_score',risk),pl.Series('predicted_behavior',[FAMILIES[k] for k in pred]))
            held_predictions.write_csv(OUT/'validation_pairs.csv')
    trees=max(50,int(np.median(iterations)))
    model=classifier(trees)
    model.fit(X,y)
    joblib.dump({'model':model,'features':features},OUT/'pair_model.joblib')
    pl.DataFrame({'feature':features,'importance':model.feature_importances_}).sort('importance',descending=True).write_csv(OUT/'feature_importance.csv')

    # Evidence labels enumerate only a subset of planted hands. Train against
    # confirmed non-target pairs; do not turn unknown positive hands negative.
    raw=pl.scan_parquet(OUT/'hands_development'/'*.parquet').collect()
    evidence=pl.read_csv(DATA/'development_evidence.csv').select('pair_id','hand_id',pl.col('behavior_family').alias('evidence_family'))
    handdata=raw.join(labels.select('pair_id','label'),on='pair_id').join(evidence,on=['pair_id','hand_id'],how='left')
    handdata=handdata.filter((pl.col('label')==0)|pl.col('evidence_family').is_not_null())
    handfeatures=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]
    HX=matrix(handdata,handfeatures)
    hy=np.array([LABELS[x] if x is not None else 0 for x in handdata['evidence_family']])
    hval=np.array([x in held_tables for x in handdata['table_id']])
    handmodel=classifier(hand=True)
    handmodel.fit(HX[~hval],hy[~hval],eval_set=[(HX[hval],hy[hval])],callbacks=[lgb.early_stopping(60,verbose=False)])
    hp=handmodel.predict_proba(HX[hval])
    hand_ap=float(average_precision_score(hy[hval]>0,1-hp[:,0]))
    # Report retrieval on all hands from held-out labeled target pairs.
    held=raw.filter(pl.col('table_id').is_in(list(held_tables)))
    ep=handmodel.predict_proba(matrix(held,handfeatures))
    held=held.with_columns([pl.Series(FAMILIES[k],ep[:,k]) for k in [1,2,3]])
    target=labels.filter(pl.col('label')==1).join(held.select('pair_id').unique(),on='pair_id')
    aps=[]
    actual={key[0]:set(group['hand_id']) for key,group in evidence.group_by('pair_id')}
    for row in target.iter_rows(named=True):
        candidates=held.filter(pl.col('pair_id')==row['pair_id']).sort(row['behavior_family'],descending=True)['hand_id'].head(5).to_list()
        truth=actual.get(row['pair_id'],set()); hits=0; ap=0.
        for rank,hand in enumerate(candidates,1):
            if hand in truth:
                hits+=1; ap+=hits/rank
        aps.append(ap/min(5,len(truth)) if truth else 0.)
    evidence_map=float(np.mean(aps)) if aps else 0.
    results_summary={'pair_validation':results,'hand_evidence_vs_confirmed_negative_ap':hand_ap,
        'evidence_map5_given_true_family':evidence_map,'retrieval_target_pairs':len(aps),
        'note':'Evidence MAP conditions on true behavior; it is a diagnostic, not the official combined competition score.',
        'pair_trees':trees,'hand_trees':int(handmodel.best_iteration_)}
    print(json.dumps(results_summary),flush=True)
    finalhand=classifier(max(50,int(handmodel.best_iteration_)),hand=True)
    finalhand.fit(HX,hy)
    joblib.dump({'model':finalhand,'features':handfeatures},OUT/'hand_model.joblib')
    (OUT/'validation_metrics.json').write_text(json.dumps(results_summary,indent=2)+'\n')


def predict():
    bundle=joblib.load(OUT/'pair_model.joblib')
    pairs=pl.read_parquet(OUT/'pairs_evaluation.parquet')
    p=bundle['model'].predict_proba(matrix(pairs,bundle['features']))
    family=np.argmax(p[:,1:],axis=1)+1
    scores=pairs.select('pair_id').with_columns(pl.Series('risk_score',1-p[:,0]),
        pl.Series('predicted_behavior',[FAMILIES[k] for k in family]))
    hb=joblib.load(OUT/'hand_model.joblib')
    all_evidence=[]
    for path in sorted((OUT/'hands_evaluation').glob('*.parquet')):
        hands=pl.read_parquet(path).join(scores.select('pair_id','predicted_behavior'),on='pair_id')
        prob=hb['model'].predict_proba(matrix(hands,hb['features']))
        chosen=np.array([LABELS[s] for s in hands['predicted_behavior']])
        hands=hands.with_columns(pl.Series('evidence_score',prob[np.arange(len(prob)),chosen]))
        top=hands.sort(['pair_id','evidence_score','hand_id'],descending=[False,True,False]).group_by('pair_id',maintain_order=True).agg(pl.col('hand_id').head(5))
        all_evidence.append(top)
        print(f'predict evidence {path.name}: {hands.height} hands',flush=True)
    ev=pl.concat(all_evidence)
    ev=ev.with_columns([pl.col('hand_id').list.get(i,null_on_oob=True).fill_null('NO_EVIDENCE').alias(f'evidence_hand_{i+1}') for i in range(5)]).drop('hand_id')
    template=pl.read_csv(DATA/'sample_submission.csv')
    result=template.select('pair_id').join(scores,on='pair_id',how='left').join(ev,on='pair_id',how='left')
    result=result.with_columns([pl.col(f'evidence_hand_{i}').fill_null('NO_EVIDENCE') for i in range(1,6)])
    assert result.height==template.height
    assert result['risk_score'].null_count()==0, 'Pairs with no gameplay summary need investigation'
    result.select(template.columns).write_csv(OUT/'submission_baseline.csv')
    print(f'wrote {result.height} rows to outputs/submission_baseline.csv',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['train','predict']); a=p.parse_args()
    train() if a.mode=='train' else predict()
