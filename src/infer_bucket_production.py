"""Extract bucket features with an exact original-feature control in every batch."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import argparse
import gc
import json
import time
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix
from conditional_inference_worker import extract_chunk,pair_aggregate
from conditional_action_features import INPUTS,card_features
from conditional_bucket_features import bucket_features
from run_conditional_policy_screen import hand_features

OUT=ROOT/'outputs_v15_bucket'; OLD=ROOT/'outputs_v11_conditional'; PACKAGE=OLD/'inference-package'
def main():
    p=argparse.ArgumentParser(); p.add_argument('--batches',type=int,nargs='+',required=True); args=p.parse_args()
    plan=json.loads((OUT/'plan.json').read_text()); fit=json.loads((OUT/'fit.json').read_text()); assert fit['plan_sha256']==sha(OUT/'plan.json')
    assert all(sha(ROOT/'src'/n)==h for n,h in plan['sources'].items())
    config=json.loads((PACKAGE/'config.json').read_text()); assert all(sha(PACKAGE/n)==h for n,h in config['files'].items())
    assert sha(OUT/'policy/final.joblib')==fit['policy_receipts'][-1]['model_sha256']
    old=joblib.load(PACKAGE/'final-models.joblib'); new=joblib.load(OUT/'policy/final.joblib'); assert old['inputs']==new['inputs']==INPUTS
    hands=pl.read_parquet(PACKAGE/'data/hands.parquet'); pairs=pl.read_csv(PACKAGE/'data/evaluation_pairs.csv').select('pair_id','player_1','player_2')
    counts={r[0]:np.array(r[1:],dtype=np.int64) for r in pl.read_parquet(PACKAGE/'style-counts.parquet').iter_rows()}
    dest=OUT/'batches'; dest.mkdir(exist_ok=True)
    for index in args.batches:
        name=f'batch_{index:03}'; receipt=dest/(name+'.json'); path=dest/(name+'-pairs.parquet')
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['plan_sha256']==sha(OUT/'plan.json') and saved['features_sha256']==sha(path); continue
        assert not path.exists(); started=time.perf_counter(); keys=pl.read_parquet(PACKAGE/'keys'/(name+'.parquet')).join(pairs,on='pair_id',validate='m:1')
        pools=sorted(keys['table_id'].unique()); controls=[]; results=[]; actions_count=0; missing=0
        for offset in range(0,len(pools),2):
            k=keys.filter(pl.col('table_id').is_in(pools[offset:offset+2])); ids=k.select('hand_id').unique()
            h=hands.join(ids,on='hand_id',how='semi'); s=pl.scan_parquet(PACKAGE/'data/seats.parquet').join(ids.lazy(),on='hand_id',how='semi').collect(); a=pl.scan_parquet(PACKAGE/'data/actions.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
            frame=extract_chunk(k,h,s,a,counts); actions_count+=frame.height; x=matrix(frame,INPUTS); ap=old['classifier'].predict_proba(x,num_threads=1)
            ca=hand_features(frame,ap,old['regressor'].predict(x,num_threads=1)); cb=bucket_features(frame,ap,new['model'].predict_proba(x,num_threads=1))
            caf=config['conditional_features']; cbf=[c for c in cb.columns if c.startswith('cb_')]
            missing+=k.select('pair_id','hand_id').join(cb.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='anti').height
            for f,names,collection in [(ca,caf,controls),(cb,cbf,results)]:
                filled=k.select('pair_id','hand_id','table_id').join(f.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(names).fill_null(0)); assert filled.height==k.height
                collection.append(pair_aggregate(filled,names))
            del k,h,s,a,frame,x,ca,cb,filled; card_features.cache_clear(); gc.collect()
        oldreceipt=json.loads((PACKAGE/'results'/(name+'.json')).read_text()); oldpath=PACKAGE/'results'/(name+'-pairs.parquet'); assert sha(oldpath)==oldreceipt['files'][oldpath.name]
        control=pl.concat(controls).sort('pair_id'); expected=pl.read_parquet(oldpath).sort('pair_id'); assert control.equals(expected),'Original conditional aggregate replay failed'
        result=pl.concat(results).sort('pair_id'); assert result['pair_id'].to_list()==expected['pair_id'].to_list(); result.write_parquet(path)
        saved={'plan_sha256':sha(OUT/'plan.json'),'features_sha256':sha(path),'original_control_receipt_sha256':sha(PACKAGE/'results'/(name+'.json')),'original_conditional_aggregates_exact':True,'pairs':result.height,'pair_hands':keys.height,'actions':actions_count,'missing_eligible_hands':missing,'seconds':time.perf_counter()-started}
        receipt.write_text(json.dumps(saved,indent=2)+'\n'); print(json.dumps({'completed_batch':index,**saved}),flush=True)

if __name__=='__main__': main()
