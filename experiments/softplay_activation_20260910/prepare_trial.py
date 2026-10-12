"""Common supported targets across three activation arms and a matched control."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];P=H.parent/'softplay_conditional_20260910'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());d=H/'trial';d.mkdir(exist_ok=False);base=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz');bx=base['train_pair'][:4209];by=base['train_y'][:4209];datasets={};maps={}
    for mode in plan['generator_arms']:
        m=json.loads((H/mode/'features/metadata.json').read_text())['views'][0];x=np.load(H/mode/'features/synthetic-pairs.npz')['x'];datasets[mode]=(x,m);maps[mode]={r['pair_id']:i for i,r in enumerate(m) if r['behavior_family']=='soft_play'}
    common=sorted(set.intersection(*[set(v) for v in maps.values()]));families={}
    for mode in maps:families[mode]={json.loads((H/mode/'pilot'/f"pool-{int(pid[3:7]):04d}"/'provenance.json').read_text())['spec']['policy_family'] for pid in common}
    if len(common)<12 or any(len(v)<4 for v in families.values()):
        (d/'cohort-failure.json').write_text(json.dumps({'targets':len(common),'families':{k:sorted(v) for k,v in families.items()},'no_fits_allowed':True}));raise RuntimeError('Insufficient common supported cohort')
    original=np.load(H.parent/'mhc_local/fold0/train-arrays.npz');np.testing.assert_array_equal(original['train_pair'],bx[:,:897]);np.testing.assert_array_equal(original['train_y'],by)
    ids=np.random.default_rng(79193).permutation(np.flatnonzero(by[:1403]==2))[:len(common)];inputs={}
    for mode in plan['detector_arms']:
        if mode=='resampled':extra=np.concatenate([bx[ids+j*1403] for j in range(3)])
        else:
            x,m=datasets[mode];ii=np.array([maps[mode][p] for p in common]);extra=np.concatenate([x[ii+j*len(m)] for j in range(3)])
        xx=np.r_[bx,extra];yy=np.r_[by,np.full(len(extra),2,dtype=np.int64)];assert np.isfinite(xx).all()
        np.savez_compressed(d/(mode+'.npz'),train_x=xx,train_y=yy,full_x=base['full_pair'],shortened_x=base['shortened_pair']);inputs[mode]={'sha256':sha(d/(mode+'.npz')),'rows':len(xx),'unique_extra_pairs':len(common),'label_counts':np.bincount(yy,minlength=4).tolist()}
    fit=json.loads((P/'controller-fit.json').read_text());pools=set(json.loads((H/'gated/training-calibration.json').read_text())['training_pools']);assert set(fit['training_pools'])==pools and len(pools)==297
    assert sha(H/'controller.joblib')==plan['controller_sha256']==fit['model_sha256']
    for folder in ['conditional-action-experiment','conditional-bucket-experiment']:
        ref=json.loads((R/'reports'/folder/'policy-screen/held.json').read_text());assert set(ref['training_pools'])==pools and not pools&set(ref['scored_pools'])
    assert all(sha(R/p)==v for p,v in plan['protected'].items())
    freeze={'created_at':datetime.now(timezone.utc).isoformat(),'study_plan_sha256':sha(H/'study-plan.json'),'inputs':inputs,'common_target_pairs':common,'control_original_indices':ids.tolist(),'held_metadata_sha256':sha(R/'experiments/mhc_local/fold0/held-metadata.json'),'source_hashes':{n:sha(H/n) for n in ['prepare_trial.py','train_trial.py','evaluate_trial.py']},'no_held_labels_read':True,'calibration_training_pools':len(pools),'unique_targets_not_duplicated':True}
    (d/'freeze.json').write_text(json.dumps(freeze,indent=2));print('Frozen common targets:',len(common),flush=True)
if __name__=='__main__':main()
