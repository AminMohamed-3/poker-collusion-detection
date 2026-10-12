"""Verify the remote archive, numerical host parity, and import unique batches."""
from pathlib import Path
import json
import shutil
import tarfile
import numpy as np
import polars as pl
import joblib
from robust_pair_experiment import ROOT,sha,matrix
from tail_pair_experiment import tails

OUT=ROOT/'outputs_v11_conditional'


def main():
    archive=OUT/'remote-results.tar.gz'
    assert sha(archive)=='0d19daf18d59070a6cadaa09f019e1617bbf36ef8a8d48b0b8f635b8ed784f98'
    destination=OUT/'remote-returned'; destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as t: t.extractall(destination,filter='data')
    assert sha(destination/'config.json')==sha(OUT/'inference-package/config.json')
    local=OUT/'inference-package/results'; remote=destination/'results'
    for index in [0,*range(6,20)]:
        receipt=remote/f'batch_{index:03}.json'; saved=json.loads(receipt.read_text())
        assert saved['plan_sha256']==sha(destination/'config.json')
        assert all(sha(remote/n)==h for n,h in saved['files'].items())
    parity={}
    for name,keys in [('pairs',['pair_id','table_id']),('evidence',['pair_id','hand_id','table_id'])]:
        a=pl.read_parquet(local/('batch_000-'+name+'.parquet')).sort(keys)
        b=pl.read_parquet(remote/('batch_000-'+name+'.parquet')).sort(keys)
        assert a.columns==b.columns and a.select(keys).equals(b.select(keys))
        features=[c for c in a.columns if c not in keys]
        av=a.select(features).to_numpy(); bv=b.select(features).to_numpy()
        np.testing.assert_allclose(av,bv,rtol=2e-6,atol=2e-6)
        parity[name]={'exact':a.equals(b),'max_absolute_difference':float(np.max(np.abs(av-bv))),'rows':a.height}
    bundle=joblib.load(OUT/'pair-model.joblib')
    raw=pl.read_parquet(ROOT/'outputs_v2/hands_evaluation/batch_000.parquet')
    base=pl.read_parquet(ROOT/'outputs_v2/pairs_evaluation.parquet').join(tails(raw,bundle['events']),on=['pair_id','table_id'])
    predictions=[]
    for folder in [local,remote]:
        frame=base.join(pl.read_parquet(folder/'batch_000-pairs.parquet'),on=['pair_id','table_id']).sort('pair_id')
        predictions.append(bundle['model'].predict_proba(matrix(frame,bundle['features'])))
    np.testing.assert_array_equal(*predictions)
    parity['pair_predictions_exact']=True
    for index in range(6,20):
        receipt=remote/f'batch_{index:03}.json'; saved=json.loads(receipt.read_text())
        for p in [receipt,*[remote/n for n in saved['files']]]:
            target=local/p.name
            if target.exists(): assert sha(target)==sha(p)
            else: shutil.copy2(p,target)
    for index in range(20):
        saved=json.loads((local/f'batch_{index:03}.json').read_text())
        assert saved['plan_sha256']==sha(destination/'config.json')
        assert all(sha(local/n)==h for n,h in saved['files'].items())
    result={'archive_sha256':sha(archive),'host_parity':parity,'all_20_batches_verified':True,
        'remote_batch0_used_for_parity_only':True,'imported_batches':list(range(6,20))}
    (OUT/'remote-import.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
