"""Frozen objective-only screen against the conditional ranker with v12 roles."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
from datetime import datetime,timezone
import json
import time
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,LABELS
from evidence_capacity_experiment import load_sequence,fit_variant,compare
from relative_evidence_experiment import records,summarize
from rank_evidence import PARAMS,context,family_columns
from equity_evidence_features import FEATURES as EQ
from conditional_evidence_classifier import fusion
from map5_objective import map5_lambdarank

OUT=ROOT/'reports/map5-objective-experiment'
SCREEN=ROOT/'reports/conditional-action-experiment'


def main():
    OUT.mkdir(exist_ok=False)
    testlog=ROOT/'reports/map5-objective-tests.log'
    assert testlog.read_text().rstrip().endswith('OK')
    params={k:v for k,v in PARAMS.items() if k not in ['objective','lambdarank_truncation_level','label_gain']}
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
        'objective_source_sha256':sha(ROOT/'src/map5_objective.py'),'fusion_source_sha256':sha(ROOT/'src/conditional_evidence_classifier.py'),'math_test_log_sha256':sha(testlog),
        'hypothesis':'Exact AP@5-swap weights may align retrieval training better with the competition metric than NDCG weights.',
        'parameters':params,'objective':'Metric-weighted RankNet, sigmoid 1, exact binary AP5 swap changes, soft query normalization log2(1+sum_abs_gradient)/sum_abs_gradient, stable initial ties.',
        'variants':['map5','map5_equal_rrf_k10'],'ensemble':'Equal reciprocal rank sum, offset 10, conditional NDCG ranker plus MAP5 ranker.',
        'scope':'One reused fold0 screen. Same 382 features, true training family, v12 predicted held family. All folds and coherent history needed before any evaluation candidate.',
        'screen_gate':'MAP gain >=.005 and no family loss beyond .025; select highest MAP among passing fixed variants, no subsequent weight tuning.',
        'confirmation_gate':'Pooled unique-pair known-fold MAP gain >=.005, each known-fold delta >=-.005, pooled family deltas >=-.025. Omitted-family mean delta >=-.005 and minimum >=-.03.',
        'coherent_gate':'For each of 65% and 40% histories: pooled gain >0, each fold delta >=-.005, pooled family delta >=-.03.',
        'limits':'No changes to risk or behavior; unannotated target hands are noisy ranking preferences, not confirmed negatives. Reused partitions cannot establish private leaderboard performance.',
        'references':['https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html','https://www.microsoft.com/en-us/research/publication/on-using-simultaneous-perturbation-stochastic-approximation-for-learning-to-rank-and-the-empirical-optimality-of-lambdarank/']}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    started=time.perf_counter()
    data,_,truths,_,pools=load_sequence()
    ca=pl.read_parquet(SCREEN/'policy-screen/hand-features.parquet'); caf=[c for c in ca.columns if c.startswith('ca_')]
    data=context(data.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
    data=context(data.join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0)),caf)
    train=data.filter(~pl.col('table_id').is_in(pools)); held=data.filter(pl.col('table_id').is_in(pools)).sort('pair_id','hand_id')
    pairpred=pl.read_csv(ROOT/'reports/conditional-behavior-experiment/fold0/pair-predictions.csv'); inverse={v:k for k,v in LABELS.items()}
    roles=pairpred.select('pair_id').with_columns(pl.Series('predicted_behavior',[inverse[int(x)] for x in pairpred['conditional_class']]))
    held=family_columns(held.join(roles,on='pair_id',validate='m:1',maintain_order='left'),'predicted_behavior')
    baseline=joblib.load(SCREEN/'screen-results/conditional-evidence.joblib'); features=baseline['features']; assert len(features)==382
    base_score=baseline['model'].predict(matrix(held,features),num_threads=2)
    expected=pl.read_parquet(ROOT/'reports/conditional-behavior-experiment/fold0/conditional_behavior-scores.parquet')
    assert held.select('pair_id','hand_id').equals(expected.select('pair_id','hand_id')); np.testing.assert_array_equal(base_score,expected['score'].to_numpy())
    model=fit_variant(train,features,{**params,'objective':map5_lambdarank,'metric':'map'})
    joblib.dump({'model':model,'features':features},OUT/'map5-model.joblib')
    prediction=model.predict(matrix(held,features),num_threads=2)
    rows=[]; summaries=[]
    for candidate,score in [('conditional',base_score),('map5',prediction),('map5_equal_rrf_k10',fusion(held,base_score,prediction))]:
        assert np.isfinite(score).all()
        r=records(held,score,truths,'screen',candidate); rows.extend(r); summaries.append({'candidate':candidate,**summarize(r)})
        held.select('pair_id','hand_id').with_columns(pl.Series('score',score)).write_parquet(OUT/(candidate+'-scores.parquet'))
    comparisons={c:compare(rows,c,baseline='conditional') for c in plan['variants']}
    passing=[c for c,r in comparisons.items() if r['delta']>=.005 and min(r['by_family_delta'].values())>=-.025]
    selected=max(passing,key=lambda c:comparisons[c]['delta']) if passing else None
    pl.DataFrame(rows).write_csv(OUT/'evidence-pairs.csv')
    metrics={'results':summaries,'comparisons':comparisons,'selected':selected,'baseline_exact':True,'seconds':time.perf_counter()-started,'model_sha256':sha(OUT/'map5-model.joblib')}
    (OUT/'screen.json').write_text(json.dumps(metrics,indent=2)+'\n'); print(json.dumps(metrics),flush=True)


if __name__=='__main__': main()
