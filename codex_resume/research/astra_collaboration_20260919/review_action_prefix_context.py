"""Verify prefix feature assembly from saved probabilities, without fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
import polars as pl
HERE=Path(__file__).resolve().parent
F=HERE.parents[2]/'goal_top5_20260919/foundation_pair'
H=F/'action_policy_fast_prefix'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=json.loads((H/'SCORE_MANIFEST.json').read_text());r=json.loads((H/'SCORE_RECEIPT.json').read_text())
    assert r['score_manifest_sha256']==sha(H/'SCORE_MANIFEST.json')
    for path,v in m['sources'].items():assert sha(Path(path))==v,path
    reports=[]
    for rec in r['folds']:
        f=rec['fold'];cp=H/f'context_scored_prefix_fold{f}.npz';pp=H/f'policy_probabilities_prefix_fold{f}.parquet'
        assert sha(cp)==rec['context_sha256'] and sha(pp)==rec['probabilities_sha256']
        new=np.load(cp,allow_pickle=False);base=np.load(F/'screen'/f'context_fold{f}_prefix60.npz',allow_pickle=False);train=np.load(F/'action_policy_fast'/f'fast129_context_fold{f}.npz',allow_pickle=False)
        assert np.array_equal(new['X_train'],train['X_train'],equal_nan=True)
        assert np.array_equal(new['y_train'],train['y_train']) and np.array_equal(new['feature_name'],train['feature_name'])
        assert np.array_equal(new['X_held'][:,:128],base['X_held'],equal_nan=True) and np.array_equal(new['pair_id'],base['pair_id'])
        a=pl.read_parquet(pp);obs=[]
        assert a.select('pair_id','direction','hand_id','action_no').unique().height==a.height
        for role in ['target','normal']:
            p=a.select([f'{role}_p{k}' for k in range(4)]).to_numpy()
            assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-6)
            obs.append(p[np.arange(a.height),a['action_class'].to_numpy()])
        llr=np.log(np.maximum(obs[0],1e-12))-np.log(np.maximum(obs[1],1e-12))
        d=a.select('pair_id','direction').with_columns(pl.Series('v',llr)).group_by('pair_id','direction').agg(pl.mean('v')).group_by('pair_id').agg(pl.mean('v'),pl.len().alias('directions'))
        assert (d['directions']==2).all() and d.height==len(base['pair_id'])
        mapping=dict(d.select('pair_id','v').iter_rows());v=np.array([mapping[k] for k in base['pair_id']],dtype=np.float32)
        assert np.array_equal(v,new['X_held'][:,128])
        reports.append({'fold':f,'pairs':len(v),'action_rows':a.height,'training129_exact':True,'prefix128_exact':True,'prefix_llr_reconstructed_exact':True})
    out=HERE/'action_policy_prefix_context_review.json';assert not out.exists()
    out.write_text(json.dumps({'folds':reports,'new_fits':0,'quality_labels_read':False},indent=2)+'\n');print(out)
if __name__=='__main__':main()
