"""One frozen pair/evidence screen of conditional wager-size residuals."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import json
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,aggregate,thin,TRAIN_VIEWS,HELDOUT_VIEW,MODEL_PARAMS,LABELS
from tail_pair_experiment import tails
from evaluate_conditional_action_screen import extra_aggregate,metrics
from evidence_capacity_experiment import load_sequence,fit_variant,compare
from relative_evidence_experiment import records,summarize
from rank_evidence import context,family_columns,PARAMS
from equity_evidence_features import FEATURES as EQ

OUT=ROOT/'reports/conditional-sizing-experiment'
SOURCE=ROOT/'reports/conditional-action-experiment'


def main():
    dest=OUT/'screen'; dest.mkdir(exist_ok=False)
    summary=json.loads((OUT/'policy-screen/summary.json').read_text())
    assert summary['features_sha256']==sha(OUT/'policy-screen/hand-features.parquet')
    (dest/'identity.json').write_text(json.dumps({'source_sha256':sha(__file__),'plan_sha256':sha(OUT/'plan.json'),'size_features_sha256':summary['features_sha256']},indent=2)+'\n')
    size=pl.read_parquet(OUT/'policy-screen/hand-features.parquet'); sf=summary['features']
    ca=pl.read_parquet(SOURCE/'policy-screen/hand-features.parquet'); caf=[c for c in ca.columns if c.startswith('ca_')]
    assert ca.select('pair_id','hand_id').equals(size.select('pair_id','hand_id'))
    raw=pl.read_parquet(sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet')))
    names=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]; events=[c for c in names if c.startswith('v2_')]
    labels=pl.read_csv(ROOT/'data/development_labels.csv')
    combined=raw.join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').join(size.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(pl.col(caf+sf).fill_null(0))
    full=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').join(tails(raw,events),on=['pair_id','table_id']).join(extra_aggregate(combined,caf+sf),on=['pair_id','table_id']).join(labels,on='pair_id').sort('pair_id')
    views={}
    for key in TRAIN_VIEWS+[HELDOUT_VIEW]:
        part=thin(combined,*key)
        views[key]=aggregate(part,names).join(tails(part,events),on=['pair_id','table_id']).join(extra_aggregate(part,caf+sf),on=['pair_id','table_id']).join(labels,on='pair_id').select(full.columns).sort('pair_id')
    heldpools=pl.read_csv(ROOT/'reports/tabicl-screen/held-pairs.csv')['table_id'].unique().to_list()
    training=pl.concat([full.filter(~pl.col('table_id').is_in(heldpools))]+[views[k].filter(~pl.col('table_id').is_in(heldpools)) for k in TRAIN_VIEWS])
    frames={'full':full.filter(pl.col('table_id').is_in(heldpools)),'shortened':views[HELDOUT_VIEW].filter(pl.col('table_id').is_in(heldpools))}
    baseline=joblib.load(SOURCE/'screen-results/conditional-pair.joblib')
    features=baseline['features']+[c+s for s in ['_mean','_std','_max','_q95'] for c in sf]; assert len(features)==953
    model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(training,features),np.array([LABELS[f] for f in training['behavior_family']]),sample_weight=np.full(training.height,1/3))
    joblib.dump({'model':model,'features':features},dest/'pair-model.joblib')
    old=pl.read_csv(SOURCE/'screen-results/pair-predictions.csv').filter(pl.col('candidate')=='conditional')
    pair_results=[]; pair_predictions=[]
    for history,frame in frames.items():
        for candidate,bundle in [('conditional',baseline),('sizing',{'model':model,'features':features})]:
            risk=1-bundle['model'].predict_proba(matrix(frame,bundle['features']))[:,0]
            if candidate=='conditional': np.testing.assert_array_equal(risk,old.filter(pl.col('history')==history).sort('pair_id')['risk'].to_numpy())
            pair_results.append({'history':history,'candidate':candidate,**metrics(frame['label'].to_numpy(),risk)})
            pair_predictions.extend({'pair_id':p,'history':history,'candidate':candidate,'risk':float(v)} for p,v in zip(frame['pair_id'],risk))
    pl.DataFrame(pair_predictions).write_csv(dest/'pair-predictions.csv')
    data,current,truths,roles,pools=load_sequence(); assert set(pools)==set(heldpools)
    data=context(data.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
    data=context(data.join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').join(size.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf+sf).fill_null(0)),caf+sf)
    train=data.filter(~pl.col('table_id').is_in(pools)); held=data.filter(pl.col('table_id').is_in(pools)).join(roles.select('pair_id','predicted_behavior'),on='pair_id',validate='m:1').sort('pair_id','hand_id')
    base=joblib.load(SOURCE/'screen-results/conditional-evidence.joblib'); ef=base['features']+sf+['context_'+c for c in sf]; assert len(ef)==410
    ranker=fit_variant(train,ef,PARAMS); joblib.dump({'model':ranker,'features':ef},dest/'evidence-model.joblib')
    erows=[]; evidence_results=[]
    for candidate,bundle in [('conditional',base),('sizing',{'model':ranker,'features':ef})]:
        score=bundle['model'].predict(matrix(family_columns(held,'predicted_behavior'),bundle['features']),num_threads=2)
        if candidate=='conditional':
            expected=pl.read_parquet(SOURCE/'screen-results/conditional-evidence-scores.parquet')
            assert expected.select('pair_id','hand_id').equals(held.select('pair_id','hand_id'))
            np.testing.assert_array_equal(score,expected['score'].to_numpy())
        rows=records(held,score,truths,'screen',candidate); erows.extend(rows)
        evidence_results.append({'candidate':candidate,**summarize(rows)})
        held.select('pair_id','hand_id').with_columns(pl.Series('score',score)).write_parquet(dest/(candidate+'-scores.parquet'))
    lookup={(r['history'],r['candidate']):r for r in pair_results}
    ordinary=[lookup[h,'sizing']['pair_ap']-lookup[h,'conditional']['pair_ap'] for h in frames]
    sparse=[lookup[h,'sizing']['sparse_ap']-lookup[h,'conditional']['sparse_ap'] for h in frames]
    ecompare=compare(erows,'sizing',baseline='conditional')
    advance_pair=np.mean(ordinary)>=0 and min(ordinary)>=-.003 and np.mean(sparse)>=.003 and min(sparse)>=-.03
    advance_evidence=ecompare['delta']>=.005 and min(ecompare['by_family_delta'].values())>=-.025
    pl.DataFrame(erows).write_csv(dest/'evidence-pairs.csv')
    result={'pair_results':pair_results,'pair_ordinary_deltas':ordinary,'pair_sparse_deltas':sparse,'evidence_results':evidence_results,'evidence_comparison':ecompare,
        'advance_pair':bool(advance_pair),'advance_evidence':bool(advance_evidence),'all_controls_exact':True}
    (dest/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
