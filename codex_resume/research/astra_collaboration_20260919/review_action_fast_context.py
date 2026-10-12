"""Independently reconstruct nested policy features; no fits/quality read."""
import hashlib,json
from pathlib import Path
import numpy as np
import polars as pl
HERE=Path(__file__).resolve().parent;P=HERE.parents[2]
H=P/'goal_top5_20260919/foundation_pair/action_policy_fast'
S=H.parent/'screen'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def fold(t):return int(hashlib.sha256(('foundation-pf35-screen-v1:'+t).encode()).hexdigest()[:16],16)%2
def main():
    manifest=json.loads((H/'MANIFEST.json').read_text());receipt=json.loads((H/'CPU_RECEIPT.json').read_text())
    assert receipt['manifest_sha256']==sha(H/'MANIFEST.json')
    for k,v in manifest['sources'].items():assert sha(Path(k))==v,k
    for k,v in manifest['artifacts'].items():assert sha(H/k)==v,k
    raw=pl.read_parquet(P/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet').select('pair_id','table_id')
    raw=raw.with_columns(pl.Series('fold',[fold(t) for t in raw['table_id']]))
    reports=[]
    for rec in receipt['folds']:
        o=rec['outer'];cp=H/f'fast129_context_fold{o}.npz';assert sha(cp)==rec['context_sha256']
        old=np.load(S/f'context_fold{o}.npz',allow_pickle=False);new=np.load(cp,allow_pickle=False)
        for key in ['X_train','X_held']:assert np.array_equal(new[key][:,:128],old[key],equal_nan=True)
        for key in ['y_train','pair_id']:assert np.array_equal(new[key],old[key])
        assert np.array_equal(new['feature_name'][:128],old['feature_name']) and new['feature_name'][128]=='action_policy_llr'
        feature_parts=[]
        outerheld=set(raw.filter(pl.col('fold')==o)['table_id'])
        for j,member in enumerate(rec['memberships']):
            ap=H/f'action_predictions_outer{o}_pipeline{j}.parquet';assert sha(ap)==member['action_predictions_sha256']
            a=pl.read_parquet(ap)
            for role in ['target','normal']:
                mp=H/f'train_membership_outer{o}_pipeline{j}_{role}.parquet';assert sha(mp)==member[role]['sha256']
                t=pl.read_parquet(mp)
                assert not outerheld.intersection(t['table_id'])
                for key in ['pair_id','table_id','actor_id','hand_id']:assert not set(t[key]).intersection(a[key])
                w=t['fit_weight'].to_numpy();assert np.isfinite(w).all() and (w>0).all() and np.isclose(w.mean(),1)
                mass=t.group_by('pair_id','direction').agg(pl.sum('fit_weight'))['fit_weight'].to_numpy();assert np.allclose(mass,mass[0])
            obs=[]
            for role in ['target','normal']:
                q=a.select([f'{role}_p{k}' for k in range(4)]).to_numpy()
                assert np.isfinite(q).all() and (q>=0).all() and (q<=1).all() and np.allclose(q.sum(1),1,atol=1e-6)
                obs.append(q[np.arange(a.height),a['action_class'].to_numpy()])
            v=np.log(np.maximum(obs[0],1e-12))-np.log(np.maximum(obs[1],1e-12))
            z=a.select('pair_id','direction').with_columns(pl.Series('v',v)).group_by('pair_id','direction').agg(pl.mean('v')).group_by('pair_id').agg(pl.mean('v'),pl.len().alias('directions'))
            assert (z['directions']==2).all()
            feature_parts.append(z.select('pair_id','v'))
        for scope,part,ids in [('train',pl.concat(feature_parts[:2]),raw.filter(pl.col('fold')!=o)['pair_id'].to_numpy()),('held',feature_parts[2],old['pair_id'])]:
            assert part['pair_id'].n_unique()==part.height
            mapping=dict(part.iter_rows());expected=np.array([mapping[k] for k in ids],dtype=np.float32)
            assert np.array_equal(new[f'X_{scope}'][:,128],expected)
        reports.append({'outer':o,'original128_exact':True,'labels_keys_exact':True,'new_feature_reconstructed_exact':True,'nested_training_disjoint':True})
    out=HERE/'action_policy_fast_context_review.json';assert not out.exists()
    out.write_text(json.dumps({'folds':reports,'new_fits':0,'quality_labels_read':False},indent=2)+'\n');print(out)
if __name__=='__main__':main()
