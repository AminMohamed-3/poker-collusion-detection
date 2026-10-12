"""Nested unique synthetic-pair doses, matched real controls, no held labels."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());d=H/'trial';d.mkdir(exist_ok=False);base=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz');bx=base['train_pair'][:4209];by=base['train_y'][:4209]
    orig=np.load(H.parent/'mhc_local/fold0/train-arrays.npz');np.testing.assert_array_equal(orig['train_pair'],bx[:,:897]);np.testing.assert_array_equal(orig['train_y'],by)
    datasets={};maps={}
    for mode in ['strength','learned']:
        m=json.loads((H/mode/'features/metadata.json').read_text())['views'][0];x=np.load(H/mode/'features/synthetic-pairs.npz')['x'];datasets[mode]=(x,m);maps[mode]={r['pair_id']:i for i,r in enumerate(m) if r['behavior_family']=='soft_play'}
    common=sorted(set(maps['strength'])&set(maps['learned']));low=[pid for pid in common if int(pid[3:7])<400]
    families={}
    for mode in maps:
        families[mode]=set(json.loads((H/mode/'pilot'/f"pool-{int(pid[3:7]):04d}"/'provenance.json').read_text())['spec']['policy_family'] for pid in common)
    if len(low)<6 or len(common)<12 or any(len(v)<4 for v in families.values()):
        (d/'cohort-failure.json').write_text(json.dumps({'low':len(low),'high':len(common),'policy_families':{k:sorted(v) for k,v in families.items()},'no_detector_fit_allowed':True},indent=2));raise RuntimeError('Insufficient supported dose cohort; stop.')
    soft=np.flatnonzero(by[:1403]==2);rng=np.random.default_rng(39791);selected=rng.permutation(soft)[:len(common)];inputs={}
    for dose,pids in [('low',low),('high',common)]:
        count=len(pids);ids=selected[:count];control=np.concatenate([bx[ids+j*1403] for j in range(3)])
        for mode in ['resampled','strength','learned']:
            if mode=='resampled':extra=control
            else:
                x,m=datasets[mode];idx=np.array([maps[mode][p] for p in pids]);extra=np.concatenate([x[idx+j*len(m)] for j in range(3)])
            arm=mode+'_'+dose;xx=np.r_[bx,extra];yy=np.r_[by,np.full(count*3,2,dtype=np.int64)];assert np.isfinite(xx).all()
            np.savez_compressed(d/(arm+'.npz'),train_x=xx,train_y=yy,full_x=base['full_pair'],shortened_x=base['shortened_pair']);inputs[arm]={'sha256':sha(d/(arm+'.npz')),'rows':len(xx),'unique_extra_pairs':count,'label_counts':np.bincount(yy,minlength=4).tolist()}
    for p,v in plan['protected'].items():assert sha(R/p)==v
    # Verify final action models and every reused normal reference exclude outer pools.
    cal=json.loads((H/'strength/training-calibration.json').read_text());pools=set(cal['training_pools']);fit=json.loads((H/'controller-fit.json').read_text());assert set(fit['training_pools'])==pools and sha(H/'controller.joblib')==fit['model_sha256']
    for folder in ['conditional-action-experiment','conditional-bucket-experiment']:
        ref=json.loads((R/'reports'/folder/'policy-screen/held.json').read_text());assert set(ref['training_pools'])==pools and not pools&set(ref['scored_pools'])
    freeze={'created_at':datetime.now(timezone.utc).isoformat(),'study_plan_sha256':sha(H/'study-plan.json'),'inputs':inputs,'cohorts':{'low':low,'high':common},'real_control_original_indices':selected.tolist(),'held_metadata_sha256':sha(R/'experiments/mhc_local/fold0/held-metadata.json'),'source_hashes':{n:sha(H/n) for n in ['prepare_trial.py','train_trial.py','evaluate_trial.py']},'no_held_labels_read':True,'calibration_training_pool_count':len(pools),'no_repeated_synthetic_target_within_a_dose':True,'paired_low_subset_of_high':set(low)<=set(common)}
    (d/'freeze.json').write_text(json.dumps(freeze,indent=2));print('Frozen cohort sizes:',len(low),len(common),flush=True)
if __name__=='__main__':main()
