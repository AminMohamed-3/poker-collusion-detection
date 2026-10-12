"""Freeze paired cohorts and three controls without loading validation labels."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def select_balanced(ids,families,n):
    buckets={k:sorted([p for p in ids if families[p]==k],key=lambda p:hashlib.sha256(('replication-cohort-v1:'+p).encode()).hexdigest()) for k in range(4)}
    result=[]
    while len(result)<n:
        for k in range(4):
            if buckets[k] and len(result)<n:result.append(buckets[k].pop(0))
    return result
def main():
    plan=json.loads((H/'study-plan.json').read_text());d=H/'trial';d.mkdir(exist_ok=False)
    base=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz');bx=base['train_pair'][:4209];by=base['train_y'][:4209]
    original=np.load(H.parent/'mhc_local/fold0/train-arrays.npz');np.testing.assert_array_equal(original['train_pair'],bx[:,:897]);np.testing.assert_array_equal(original['train_y'],by)
    datasets={};maps={};common={};families={}
    for arm in plan['generator_arms']:
        assert sha(H/arm/'pilot-plan.json')==plan['generator_plan_sha256'][arm]
        for audit in ['pilot-audit.json','intervention-replay-audit.json']:assert (H/arm/audit).exists()
        m=json.loads((H/arm/'features/metadata.json').read_text())['views'][0];x=np.load(H/arm/'features/synthetic-pairs.npz')['x']
        datasets[arm]=(x,m);maps[arm]={r['pair_id']:i for i,r in enumerate(m) if r['behavior_family']=='soft_play'}
    for b in plan['batches']:
        bid=str(b['batch']);common[bid]=sorted(set.intersection(*[set(maps[a]) for a in b['arms']]))
        families[bid]={p:json.loads((H/b['arms'][0]/'pilot'/f'pool-{int(p[3:7]):04d}'/'provenance.json').read_text())['spec']['policy_family'] for p in common[bid]}
    n=min(map(len,common.values()))
    if n<12 or any(len(set(v.values()))<4 for v in families.values()):
        (d/'cohort-failure.json').write_text(json.dumps({'common':common,'families':families,'no_fits_allowed':True},indent=2));raise RuntimeError('Predeclared supported-cohort requirement failed')
    selected={b:select_balanced(v,families[b],n) for b,v in common.items()}
    realsoft=np.flatnonzero(by[:1403]==2);assert len(realsoft)==92
    control_ids={f'resampled_{i}':np.random.default_rng(seed).permutation(realsoft)[:n] for i,seed in enumerate(plan['resampling_seeds'])}
    assert all(len(set(v))==n for v in control_ids.values())
    assert 3*n<=len(realsoft)
    control_ids.update({f'resampled_pool_{i}':np.random.default_rng(seed).permutation(realsoft)[:3*n] for i,seed in enumerate(plan['resampling_seeds'])})
    assert all(len(set(control_ids[f'resampled_pool_{i}']))==3*n for i in range(3))
    inputs={}
    for arm in plan['detector_arms']:
        if arm=='original':extra=np.empty((0,bx.shape[1]),dtype=bx.dtype)
        elif arm in control_ids:extra=np.concatenate([bx[control_ids[arm]+j*1403] for j in range(3)])
        elif arm=='pooled':
            chunks=[]
            for batch in plan['batches']:
                name=batch['arms'][0];xx,mm=datasets[name];ii=np.array([maps[name][pair] for pair in selected[str(batch['batch'])]])
                chunks.extend(xx[ii+j*len(mm)] for j in range(3))
            extra=np.concatenate(chunks)
            assert len(extra)==9*n
        else:
            b=str(next(b['batch'] for b in plan['batches'] if arm in b['arms']));x,m=datasets[arm];ii=np.array([maps[arm][p] for p in selected[b]])
            extra=np.concatenate([x[ii+j*len(m)] for j in range(3)])
        xx=np.concatenate([bx,extra]);yy=np.r_[by,np.full(len(extra),2,dtype=by.dtype)]
        assert np.isfinite(xx).all();np.testing.assert_array_equal(xx[:4209],bx);np.testing.assert_array_equal(yy[:4209],by)
        np.savez_compressed(d/(arm+'.npz'),train_x=xx,train_y=yy,full_x=base['full_pair'],shortened_x=base['shortened_pair'])
        inputs[arm]={'sha256':sha(d/(arm+'.npz')),'rows':len(xx),'unique_extra_pairs':len(extra)//3,'label_counts':np.bincount(yy,minlength=4).tolist()}
    pools=set(json.loads((H/plan['generator_arms'][0]/'training-calibration.json').read_text())['training_pools']);fit=json.loads((H.parent/'softplay_capacity_20260910/controller-fit.json').read_text())
    assert len(pools)==297 and set(fit['training_pools'])==pools
    assert sha(H/'controller.joblib')==plan['controller_sha256']==fit['controller_sha256']
    for folder in ['conditional-action-experiment','conditional-bucket-experiment']:
        ref=json.loads((R/'reports'/folder/'policy-screen/held.json').read_text());assert set(ref['training_pools'])==pools and not pools&set(ref['scored_pools'])
    assert all(sha(R/p)==v for p,v in plan['protected'].items())
    sources=['prepare_trial.py','train_trial.py','evaluate_trial.py','replication_statistics.py','diagnostics.py']
    analysis=json.loads((H/'analysis-freeze.json').read_text())
    assert all(sha(H/k)==v for k,v in analysis['sources'].items())
    freeze={'created_at':datetime.now(timezone.utc).isoformat(),'study_plan_sha256':sha(H/'study-plan.json'),'inputs':inputs,'common_target_pairs':common,'selected_target_pairs':selected,'selected_families':{b:{p:families[b][p] for p in ids} for b,ids in selected.items()},'common_dose':n,'control_original_indices':{k:v.tolist() for k,v in control_ids.items()},'held_metadata_sha256':sha(H.parent/'mhc_local/fold0/held-metadata.json'),'source_hashes':{n:sha(H/n) for n in sources},'analysis_freeze_sha256':sha(H/'analysis-freeze.json'),'no_held_labels_read':True,'training_pools':297}
    (d/'freeze.json').write_text(json.dumps(freeze,indent=2));print(json.dumps({'cohort_size':n,'common_counts':{b:len(v) for b,v in common.items()},'fits':plan['detector_fits']}),flush=True)
if __name__=='__main__':main()
