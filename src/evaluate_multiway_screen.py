"""One frozen multiway feature screen; preserve risk, family and ranker recipe."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
from datetime import datetime,timezone
import json
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,LABELS
from rank_evidence import PARAMS,context,family_columns
from evidence_capacity_experiment import load_sequence,fit_variant,compare
from relative_evidence_experiment import records,summarize
from equity_evidence_features import FEATURES as EQ
from multiway_evidence_features import FEATURES as MW

OUT=ROOT/'reports/multiway-evidence-experiment'
SCREEN=ROOT/'reports/conditional-action-experiment'


def main():
    dest=OUT/'screen'; dest.mkdir(exist_ok=False)
    package=OUT/'package'; plan=json.loads((OUT/'plan.json').read_text())
    assert sha(OUT/'plan.json')==sha(package/'config.json')
    assert all(sha(package/p)==h for p,h in plan['files'].items())
    receipts=[]
    for i in range(8):
        p=package/'results'/f'batch_{i:03}.json'; r=json.loads(p.read_text())
        assert r['config_sha256']==sha(package/'config.json') and r['output_sha256']==sha(p.with_suffix('.parquet'))
        receipts.append(sha(p))
    mw=pl.read_parquet(sorted((package/'results').glob('batch_*.parquet'))).sort('pair_id','hand_id')
    assert mw.height==45129 and mw.select(pl.struct('pair_id','hand_id').n_unique()).item()==45129
    mw.write_parquet(OUT/'development.parquet')
    identity={'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'plan_sha256':sha(OUT/'plan.json'),'batch_receipts':receipts,'features_sha256':sha(OUT/'development.parquet')}
    (dest/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    data,_,truths,_,pools=load_sequence()
    ca=pl.read_parquet(SCREEN/'policy-screen/hand-features.parquet'); caf=[c for c in ca.columns if c.startswith('ca_')]
    data=context(data.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
    data=context(data.join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0)),caf)
    data=context(data.join(mw,on=['pair_id','hand_id'],validate='1:1'),MW)
    train=data.filter(~pl.col('table_id').is_in(pools)); held=data.filter(pl.col('table_id').is_in(pools)).sort('pair_id','hand_id')
    pairpred=pl.read_csv(ROOT/'reports/conditional-behavior-experiment/fold0/pair-predictions.csv'); inverse={v:k for k,v in LABELS.items()}
    roles=pairpred.select('pair_id').with_columns(pl.Series('predicted_behavior',[inverse[int(x)] for x in pairpred['conditional_class']]))
    held=family_columns(held.join(roles,on='pair_id',validate='m:1',maintain_order='left'),'predicted_behavior')
    baseline=joblib.load(SCREEN/'screen-results/conditional-evidence.joblib'); features=baseline['features']+MW+['context_'+c for c in MW]; assert len(features)==430
    control=baseline['model'].predict(matrix(held,baseline['features']),num_threads=2)
    expected=pl.read_parquet(ROOT/'reports/conditional-behavior-experiment/fold0/conditional_behavior-scores.parquet')
    assert held.select('pair_id','hand_id').equals(expected.select('pair_id','hand_id')); np.testing.assert_array_equal(control,expected['score'].to_numpy())
    model=fit_variant(train,features,PARAMS); joblib.dump({'model':model,'features':features},dest/'multiway-model.joblib')
    scores=model.predict(matrix(held,features),num_threads=2); rows=[]; summaries=[]
    for candidate,score in [('conditional',control),('multiway',scores)]:
        r=records(held,score,truths,'screen',candidate); rows.extend(r); summaries.append({'candidate':candidate,**summarize(r)})
        held.select('pair_id','hand_id').with_columns(pl.Series('score',score)).write_parquet(dest/(candidate+'-scores.parquet'))
    comparison=compare(rows,'multiway',baseline='conditional'); advance=comparison['delta']>=.005 and min(comparison['by_family_delta'].values())>=-.025
    pl.DataFrame(rows).write_csv(dest/'evidence-pairs.csv')
    metrics={'results':summaries,'comparison':comparison,'advance':bool(advance),'baseline_exact':True,
        'nonzero_feature_rows':mw.filter(pl.any_horizontal(pl.col(MW)>0)).height,'feature_means':mw.select(pl.col(MW).mean()).to_dicts()[0]}
    (dest/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n'); print(json.dumps(metrics),flush=True)


if __name__=='__main__': main()
