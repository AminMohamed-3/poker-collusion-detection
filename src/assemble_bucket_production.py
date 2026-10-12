"""Assemble the admitted risk-only blend and verify the entire submission."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import json
from datetime import datetime,timezone
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix
from tail_pair_experiment import tails
from validation import validate_submission

OUT=ROOT/'outputs_v15_bucket'; OLD=ROOT/'outputs_v11_conditional'; SOURCE=ROOT/'outputs_v13_coverage/submission_conditional_coverage.csv'
def main():
    plan=json.loads((OUT/'plan.json').read_text()); fit=json.loads((OUT/'fit.json').read_text()); assert fit['plan_sha256']==sha(OUT/'plan.json')
    assert all(sha(ROOT/'src'/n)==h for n,h in plan['sources'].items()); assert sha(SOURCE)==plan['incumbent_submission_sha256']
    assert sha(OUT/'pair-model.joblib')==fit['pair_model_sha256'] and sha(OLD/'pair-model.joblib')==plan['incumbent_pair_model_sha256']
    old=joblib.load(OLD/'pair-model.joblib'); new=joblib.load(OUT/'pair-model.joblib'); source=pl.read_csv(SOURCE,infer_schema=False)
    pairs=pl.read_parquet(ROOT/'outputs_v2/pairs_evaluation.parquet'); predictions=[]; receipts=[]
    for path in sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet')):
        name=path.stem; r=json.loads((OUT/'batches'/(name+'.json')).read_text()); cbpath=OUT/'batches'/(name+'-pairs.parquet'); assert r['plan_sha256']==sha(OUT/'plan.json') and sha(cbpath)==r['features_sha256'] and r['original_conditional_aggregates_exact']
        capath=OLD/'inference-package/results'/(name+'-pairs.parquet'); cr=json.loads((OLD/'inference-package/results'/(name+'.json')).read_text()); assert sha(capath)==cr['files'][capath.name]
        raw=pl.read_parquet(path); frame=pairs.join(tails(raw,old['events']),on=['pair_id','table_id'],validate='1:1').join(pl.read_parquet(capath),on=['pair_id','table_id'],validate='1:1').join(pl.read_parquet(cbpath),on=['pair_id','table_id'],validate='1:1').sort('pair_id')
        assert frame.height==r['pairs']; baseline=1-old['model'].predict_proba(matrix(frame,old['features']),num_threads=2)[:,0]
        expected=source.join(frame.select('pair_id'),on='pair_id',how='semi').sort('pair_id')['risk_score'].cast(pl.Float64).to_numpy(); np.testing.assert_array_equal(baseline,expected)
        risk=1-new['model'].predict_proba(matrix(frame,new['features']),num_threads=2)[:,0]; blended=(baseline+risk)/2
        assert np.isfinite(blended).all() and np.all((blended>=0)&(blended<=1))
        predictions.append(frame.select('pair_id').with_columns(pl.Series('new_risk',blended),pl.Series('bucket_risk',risk),pl.Series('incumbent_risk',baseline))); receipts.append(r)
    assert len(receipts)==20; predictions=pl.concat(predictions); assert predictions.height==predictions['pair_id'].n_unique()==112540
    predictions.write_parquet(OUT/'predictions.parquet'); result=source.join(predictions.select('pair_id','new_risk'),on='pair_id',validate='1:1',maintain_order='left').with_columns(pl.col('new_risk').cast(pl.String).alias('risk_score')).select(source.columns)
    unchanged=[c for c in source.columns if c!='risk_score']; assert result.select(unchanged).equals(source.select(unchanged))
    target=OUT/'submission_bucket_risk_blend.csv'; assert not target.exists(); result.write_csv(target); assert pl.read_csv(target,infer_schema=False).equals(result)
    validation=validate_submission(target,ROOT/'data/evaluation_pairs.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/sample_submission.csv').to_dict()
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n'); assert validation['valid']
    receipt={'created_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(OUT/'plan.json'),'submission_sha256':sha(target),'bytes':target.stat().st_size,'rows':result.height,'all_original_feature_and_risk_controls_exact':True,'behavior_and_evidence_unchanged':True,'validation':validation,'batches':20,'pair_hands':sum(r['pair_hands'] for r in receipts),'risk_blend':'0.5 incumbent + 0.5 bucket on every pair','no_public_score_yet':True}
    (OUT/'ready-audit.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)

if __name__=='__main__': main()
