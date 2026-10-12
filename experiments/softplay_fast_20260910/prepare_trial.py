"""Prepare all five inputs and freeze sources before any utility fits."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];OLD=H.parent/'history_realism_20260910'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def synset(paths):
    views=[[],[],[]];meta=[]
    for p in paths:
        m=json.loads((p/'features/metadata.json').read_text())['views'][0];x=np.load(p/'features/synthetic-pairs.npz')['x'];n=len(m)
        take=np.flatnonzero([r['behavior_family']=='soft_play' for r in m]);meta.extend([m[i] for i in take])
        for j in range(3):views[j].append(x[take+j*n])
    return [np.concatenate(v) for v in views],meta
def main():
    plan=json.loads((H/'study-plan.json').read_text());d=H/'trial';d.mkdir(exist_ok=False)
    assert all(sha(R/p)==h for p,h in plan['protected'].items())
    real=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz');x=real['train_pair'][:4209];y=real['train_y'][:4209];n=1403
    ids=np.load(H/'selected-original-indices.npy');control=np.concatenate([x[ids+j*n] for j in range(3)]);cy=np.concatenate([y[ids+j*n] for j in range(3)]);rv=np.load(H/'real-views.npz');np.testing.assert_array_equal(rv['y'],cy)
    features=json.loads((OLD/'persistent_sessions/features/metadata.json').read_text())['features'];assert x.shape==(4209,1217)
    assert np.array_equal(y[:n],y[n:2*n]) and np.array_equal(y[:n],y[2*n:])
    originalmeta=json.loads((H.parent/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs']
    pools={r['table_id'] for r in originalmeta};cal=json.loads((OLD/'persistent_sessions/training-calibration.json').read_text())
    assert pools==set(cal['training_pools']) and len(pools)==297
    references={}
    for folder,name in [('conditional-action-experiment','held-models.joblib'),('conditional-bucket-experiment','held.joblib')]:
        p=R/'reports'/folder/'policy-screen';r=json.loads((p/'held.json').read_text());assert set(r['training_pools'])==pools and not pools&set(r['scored_pools']);assert sha(p/name)==r['model_sha256'];references[folder]=sha(p/name)
    inputs={};selection={};rng=np.random.default_rng(20719)
    for arm in plan['arms']:
        extra=control.copy()
        if arm in ['old_soft','repaired_soft']:
            paths=[OLD/'persistent_sessions',OLD/'session_confirmation'] if arm=='old_soft' else [H/'repaired']
            views,meta=synset(paths);assert len(meta)>=8
            order=[]
            while len(order)<40:order.extend(rng.permutation(len(meta)).tolist())
            order=np.array(order[:40]);soft=np.flatnonzero(cy[:len(ids)]==2);assert len(soft)==40
            for j in range(3):extra[soft+j*len(ids)]=views[j][order]
            selection[arm]={'unique_pairs':len(set(order.tolist())),'draws':40,'max_reuse':int(np.bincount(order).max()),'pair_ids':[meta[i]['pair_id'] for i in order]}
        if arm=='real_history_views':extra=rv['x']
        xx=x if arm=='original' else np.r_[x,extra];yy=y if arm=='original' else np.r_[y,cy]
        assert np.isfinite(xx).all()
        np.savez_compressed(d/(arm+'.npz'),train_x=xx,train_y=yy,full_x=real['full_pair'],shortened_x=real['shortened_pair'])
        inputs[arm]={'sha256':sha(d/(arm+'.npz')),'rows':len(xx),'label_counts':np.bincount(yy,minlength=4).tolist()}
    receipt={'created_at':datetime.now(timezone.utc).isoformat(),'study_plan_sha256':sha(H/'study-plan.json'),'inputs':inputs,'synthetic_selection':selection,'reference_hashes':references,'original_training_pools':len(pools),'source_hashes':{n:sha(H/n) for n in ['prepare_trial.py','train_trial.py','evaluate_trial.py']},'held_metadata_sha256':sha(R/'experiments/mhc_local/fold0/held-metadata.json'),'held_metadata_read_for_hash_only':True,'features':features}
    (d/'freeze.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ['features','synthetic_selection']},indent=2))
if __name__=='__main__':main()
