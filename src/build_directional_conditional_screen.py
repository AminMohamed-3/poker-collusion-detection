"""Apply existing pool-separated reference models; fit no new references."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import json
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT, sha
from directional_conditional_features import directional_features
from run_conditional_policy_screen import hand_features

OUT=ROOT/'reports/directional-conditional-experiment'
SOURCE=ROOT/'reports/conditional-action-experiment'


def main():
    plan=OUT/'plan.json'
    assert plan.exists()
    dest=OUT/'policy-screen'; dest.mkdir(exist_ok=False)
    data=pl.read_parquet(sorted((SOURCE/'action-cache').glob('*.parquet'))).sort('pair_id','hand_id','direction','action_no')
    pieces=[]
    for name in ['inner0','inner1','inner2','held']:
        reference=json.loads((SOURCE/'policy-screen'/(name+'.json')).read_text())
        modelpath=SOURCE/'policy-screen'/(name+'-models.joblib')
        assert sha(modelpath)==reference['model_sha256']
        assert not set(reference['training_pools'])&set(reference['scored_pools'])
        scored=data.filter(pl.col('table_id').is_in(reference['scored_pools']))
        bundle=joblib.load(modelpath); X=scored.select(bundle['inputs']).to_numpy().astype(np.float32)
        probability=bundle['classifier'].predict_proba(X,num_threads=2)
        strength=bundle['regressor'].predict(X,num_threads=2)
        control=hand_features(scored,probability,strength).sort('pair_id','hand_id')
        expected=pl.read_parquet(SOURCE/'policy-screen'/(name+'.parquet')).sort('pair_id','hand_id')
        assert control.equals(expected), 'Original hand residuals do not reproduce'
        result=directional_features(scored,probability,strength); path=dest/(name+'.parquet'); result.write_parquet(path); pieces.append(result)
        # Exchange actor orientation for a bounded, real-data invariance check.
        probe=scored.head(256); swapped=probe.with_columns((1-pl.col('direction')).alias('direction'))
        assert directional_features(probe,probability[:256],strength[:256]).equals(directional_features(swapped,probability[:256],strength[:256]))
        receipt={'plan_sha256':sha(plan),'model_sha256':sha(modelpath),'features_sha256':sha(path),'rows':result.height,'old_features_exact':True,'actor_swap_invariant':True}
        (dest/(name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps({'part':name,**receipt}),flush=True)
    combined=pl.concat(pieces).sort('pair_id','hand_id')
    assert combined.height==223749 and combined.select(pl.struct('pair_id','hand_id').n_unique()).item()==combined.height
    path=dest/'hand-features.parquet'; combined.write_parquet(path)
    summary={'features':[c for c in combined.columns if c.startswith('cd_')],'features_sha256':sha(path),'plan_sha256':sha(plan)}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__': main()
