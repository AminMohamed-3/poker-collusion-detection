"""Verify nested family-policy features from saved action probabilities."""
import json,hashlib
from pathlib import Path
import numpy as np
import polars as pl
HERE=Path(__file__).resolve().parent;P=HERE.parents[2]
H=P/'goal_top5_20260919/foundation_pair/action_policy_family_fast';S=H.parent/'screen'
FAMS=['directed_transfer','soft_play','coordinated_isolation']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def fold(t):return int(hashlib.sha256(('foundation-pf35-screen-v1:'+t).encode()).hexdigest()[:16],16)%2
def main():
    m=json.loads((H/'MANIFEST.json').read_text());r=json.loads((H/'CPU_RECEIPT.json').read_text())
    assert r['manifest_sha256']==sha(H/'MANIFEST.json') and r['new_family_fits']==18
    for k,v in m['sources'].items():assert sha(Path(k))==v,k
    raw=pl.read_parquet(P/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet').select('pair_id','table_id')
    raw=raw.with_columns(pl.Series('fold',[fold(t) for t in raw['table_id']]))
    reports=[]
    for rec in r['folds']:
        o=rec['fold'];path=H/f'fast131_context_fold{o}.npz';assert sha(path)==rec['context_sha256']
        new=np.load(path,allow_pickle=False);old=np.load(S/f'context_fold{o}.npz',allow_pickle=False)
        for n in ['X_train','X_held']:assert np.array_equal(new[n][:,:128],old[n],equal_nan=True)
        for n in ['y_train','pair_id']:assert np.array_equal(new[n],old[n])
        assert list(new['feature_name'][-3:])==['llr_dt','llr_sp','llr_ci']
        parts=[];outerheld=set(raw.filter(pl.col('fold')==o)['table_id'])
        for j in range(3):
            receipts=rec['fits'][j*4:(j+1)*4];pp=H/f'family_action_probabilities_outer{o}_pipeline{j}.parquet'
            assert sha(pp)==receipts[3]['probabilities_sha256']
            assert sha(H/f'family_models_outer{o}_pipeline{j}.joblib')==receipts[3]['model_sha256']
            a=pl.read_parquet(pp);obs={}
            for name in ['normal',*FAMS]:
                q=a.select([f'{name}_p{k}' for k in range(4)]).to_numpy()
                assert np.isfinite(q).all() and (q>=0).all() and (q<=1).all() and np.allclose(q.sum(1),1,atol=1e-6)
                obs[name]=q[np.arange(a.height),a['action_class'].to_numpy()]
            out=None
            for i,fam in enumerate(FAMS):
                member=H/f'membership_outer{o}_pipeline{j}_{fam}.parquet';assert sha(member)==receipts[i]['membership_sha256']
                t=pl.read_parquet(member);assert not outerheld.intersection(t['table_id'])
                for key in ['pair_id','table_id','hand_id']:assert not set(t[key]).intersection(a[key])
                w=t['weight'].to_numpy();assert np.isfinite(w).all() and (w>0).all() and np.isclose(w.mean(),1)
                mass=t.group_by('pair_id','direction').agg(pl.sum('weight'))['weight'].to_numpy();assert np.allclose(mass,mass[0])
                v=np.log(np.maximum(obs[fam],1e-12))-np.log(np.maximum(obs['normal'],1e-12))
                z=a.select('pair_id','direction').with_columns(pl.Series(fam,v)).group_by('pair_id','direction').agg(pl.mean(fam)).group_by('pair_id').agg(pl.mean(fam),pl.len().alias('nd'))
                assert (z['nd']==2).all();z=z.drop('nd');out=z if out is None else out.join(z,on='pair_id',validate='1:1')
            parts.append(out)
        for scope,part,ids in [('train',pl.concat(parts[:2]),raw.filter(pl.col('fold')!=o)['pair_id'].to_numpy()),('held',parts[2],old['pair_id'])]:
            assert part['pair_id'].n_unique()==part.height
            lookup={x[0]:x[1:] for x in part.select('pair_id',*FAMS).iter_rows()}
            expected=np.array([lookup[k] for k in ids],dtype=np.float32)
            assert np.array_equal(expected,new[f'X_{scope}'][:,128:])
        reports.append({'fold':o,'original128_exact':True,'three_features_reconstructed_exact':True,'nested_memberships_disjoint':True})
    dest=HERE/'family_context_review.json';assert not dest.exists()
    dest.write_text(json.dumps({'folds':reports,'new_fits':0,'quality_labels_read':False},indent=2)+'\n');print(dest)
if __name__=='__main__':main()
