"""Freeze matched training inputs using training-only availability and audits."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
import numpy as np
H=Path(__file__).resolve().parent; R=H.parents[1]; O=H/'trial'
OLD=H.parent/'data_expansion/augmentation'; REAL=H.parent/'history_realism_20260910'
LABELS={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def dataset(folders):
    xs=[[] for _ in range(3)]; ms=[[] for _ in range(3)]
    for d in folders:
        a=np.load(d/'features/synthetic-pairs.npz');m=read(d/'features/metadata.json');n=len(m['views'][0])
        assert a['x'].shape==(3*n,1217)
        for i in range(3):
            assert [r['pair_id'] for r in m['views'][i]]==[r['pair_id'] for r in m['views'][0]]
            assert np.array_equal(a['y'][i*n:(i+1)*n],[LABELS[r['behavior_family']] for r in m['views'][i]])
            xs[i].append(a['x'][i*n:(i+1)*n]);ms[i]+=m['views'][i]
    return [np.concatenate(x) for x in xs],ms

def balanced_select(meta,label,count,rng):
    groups={f:[] for f in range(4)}
    for i,row in enumerate(meta):
        if LABELS[row['behavior_family']]==label:
            pool=int(row['table_id'].replace('SSPOOL',''));groups[(pool%16)//4].append(i)
    for g in groups.values():rng.shuffle(g)
    chosen=[]
    while len(chosen)<count:
        active=[f for f,g in groups.items() if g];rng.shuffle(active)
        assert active
        for f in active:
            if len(chosen)<count:chosen.append(groups[f].pop())
    return chosen

def main():
    assert not O.exists();O.mkdir()
    study=read(H/'study-plan.json');vp=H/'varied'
    vchecks={'hand_source':not read(vp/'pilot-audit.json')['domain_discrepancy_flag'],
             'pair_source':read(vp/'pair-domain-audit.json')['auc']<=.8,
             'temporal_source':read(vp/'temporal-audit.json')['passed'],
             'balance_ledger':read(vp/'balance-audit.json')['all_stack_transitions_verified'],
             'seat_ledger':read(vp/'seat-schedule-audit.json')['all_seat_event_ledger_entries_verified'],
             'replay':read(vp/'replay-audit.json')['all_serialized_actions_replayed_legally']}
    calibrated=[REAL/n for n in study['datasets']['calibrated']]
    for d in calibrated:
        assert read(d/'pair-domain-audit.json')['auc']<=.8 and read(d/'temporal-audit.json')['passed']
        assert not read(d/'pilot-audit.json')['domain_discrepancy_flag']
        assert read(d/'replay-audit.json')['all_serialized_actions_replayed_legally']
    sets={'calibrated':dataset(calibrated)}
    if all(vchecks.values()):sets['varied']=dataset([vp])
    counts={label:min(cap,*[sum(LABELS[r['behavior_family']]==LABELS[label] for r in m[0]) for x,m in sets.values()]) for label,cap in study['class_pair_caps'].items()}
    assert min(counts[n] for n in LABELS if n!='none')>=4
    original=dict(np.load(OLD/'synthetic/tree-arrays.npz'))
    original['train_pair']=original['train_pair'][:4209];original['train_y']=original['train_y'][:4209]
    orig897=np.load(R/'experiments/mhc_local/fold0/train-arrays.npz')
    for tag in ['train','full','shortened']:np.testing.assert_array_equal(original[tag+'_pair'][:,:897],orig897[tag+'_pair'])
    np.testing.assert_array_equal(original['train_y'],orig897['train_y'])
    trainmeta=read(H/'analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json')['train_full']['pairs']
    assert len(trainmeta)==1403
    pools=set(read(H/'training-calibration.json')['training_pools']);assert {r['table_id'] for r in trainmeta}==pools
    yy=np.concatenate([np.full(counts[label],value,dtype=np.int64) for label,value in LABELS.items()]);n=len(yy)
    rng=np.random.default_rng(study['selection_seed']); extras={};selections={}
    for name,(views,metas) in sets.items():
        selected=np.array(sum([balanced_select(metas[0],value,counts[label],rng) for label,value in LABELS.items()],[]))
        assert len(selected)==len(set(selected))==n
        assert np.array_equal([LABELS[metas[0][i]['behavior_family']] for i in selected],yy)
        extras[name]=np.concatenate([view[selected] for view in views])
        selections[name]=[metas[0][i] for i in selected]
    picked=np.array([rng.choice(np.flatnonzero(original['train_y'][:1403]==v)) for v in yy])
    indices=np.r_[picked,picked+1403,picked+2806]
    extras['resampled_control']=original['train_pair'][indices]
    assert np.array_equal(original['train_y'][indices],np.tile(yy,3))
    arms=[a for a in study['arms'] if a in extras]
    for arm in arms:
        d=O/arm;d.mkdir();x=np.r_[original['train_pair'],extras[arm]];y=np.r_[original['train_y'],np.tile(yy,3)]
        np.savez_compressed(d/'train-arrays.npz',train_pair=x[:,:897],train_y=y,full_pair=original['full_pair'][:,:897],shortened_pair=original['shortened_pair'][:,:897],scaler_pair=original['train_pair'][:,:897])
        np.savez_compressed(d/'tree-arrays.npz',train_pair=x,train_y=y,full_pair=original['full_pair'],shortened_pair=original['shortened_pair'])
    for name in ['model.py','train.py']:shutil.copy2(OLD/name,O/name)
    tree=(OLD/'train_augmentation_trees.py').read_text().replace("for arm in ['resampled_control','synthetic']:","for arm in p['arms']:")
    (O/'train_augmentation_trees.py').write_text(tree)
    runner=(OLD/'run_remote.py').read_text().replace("'total':18","'total':len(p['variants'])*len(p['arms'])*len(p['seeds'])").replace('assert len(done)==18',"assert len(done)==len(p['variants'])*len(p['arms'])*len(p['seeds'])")
    (O/'run_remote.py').write_text(runner)
    metadata={'counts_by_class':counts,'extra_pairs':n,'extra_views':n*3,'total_rows':4209+n*3,'arms':arms,'varied_admission_checks':vchecks,'varied_admitted':all(vchecks.values()),'selections':selections,'control_original_full_view_indices':picked.tolist(),'no_original_held_labels_used':True,'original897_arrays_match_exactly':True,'source_inputs':{str(d.relative_to(R)/'features/synthetic-pairs.npz'):sha(d/'features/synthetic-pairs.npz') for d in calibrated+[vp]}}
    save(O/'metadata.json',metadata)
    old=read(OLD/'plan.json')
    plan={**old,'created_at':datetime.now(timezone.utc).isoformat(),'arms':arms,'study_plan_sha256':sha(H/'study-plan.json'),'parent_trial_sha256':sha(OLD/'plan.json'),'prepared_metadata_sha256':sha(O/'metadata.json'),
          'source_hashes':{n:sha(O/n) for n in ['model.py','train.py','train_augmentation_trees.py','run_remote.py']},
          'tree_inputs':{arm:sha(O/arm/'tree-arrays.npz') for arm in arms},
          'comparison':f'Each arm uses the same 4209 original views plus {n*3} extra views with exactly matched labels and training steps. Synthetic views remain grouped by complete pair. Original-only scaler. Calibrated data includes both repaired batches; varied admission governed by prespecified source checks.',
          'contrasts':[[arm,'resampled_control'] for arm in arms if arm!='resampled_control']+([['varied','calibrated']] if 'varied' in arms else []),
          'limitations':study['limitations'], 'preparer_sha256':sha(__file__),'evaluator_sha256':sha(H/'evaluate_trial.py'),
          'stress_test':{'dataset':'varied','eligible_only_if_not_used_for_training':True,'metrics':['pair_ap','multiclass_logloss'],'report_all_four_regimes':True,'interpretation':'Deliberate synthetic distribution shift; not a private-test proxy, production gate or evidence of real-world calibration.'}}
    # Drop inherited identifiers for the old, unrepaired synthetic pilot.
    for k in ['generator_plan_sha256','generator_audit_sha256','adapter_parity_sha256']:plan.pop(k,None)
    save(O/'plan.json',plan)
    for arm in arms:save(O/arm/'data-audit.json',{'plan_sha256':sha(O/'plan.json'),'train_arrays_sha256':sha(O/arm/'train-arrays.npz'),'held_labels_absent_from_training_arrays':True,'matched_rows':4209+n*3,'scaler_original_only':True})
    files=[O/n for n in ['plan.json','model.py','train.py','train_augmentation_trees.py','run_remote.py']]+[O/a/n for a in arms for n in ['train-arrays.npz','data-audit.json']]
    save(O/'remote-manifest.json',{p.relative_to(O).as_posix():sha(p) for p in files})
    with tarfile.open(O/'package.tar.gz','w:gz') as t:
        for p in files+[O/'remote-manifest.json']:t.add(p,arcname=p.relative_to(O))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['selections','control_original_full_view_indices','source_inputs']},indent=2))
if __name__=='__main__':main()
