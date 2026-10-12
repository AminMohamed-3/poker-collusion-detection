from pathlib import Path
import hashlib,json,shutil,tarfile
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;O=H/'augmentation';R=H.parents[1];OLD=R/'experiments/mhc_local'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (O/'plan.json').exists()
old=json.loads((OLD/'plan.json').read_text())
shutil.copy2(OLD/'model.py',O/'model.py')
s=(OLD/'train.py').read_text().replace("{'train_pair','train_y','full_pair','shortened_pair'}","{'train_pair','train_y','full_pair','shortened_pair','scaler_pair'}")
s=s.replace("train=arrays['train_pair']; center=", "train=arrays['scaler_pair']; center=")
s=s.replace('"""Local-only fixed training; no held labels or held-selected checkpoints."""','"""Matched augmentation trial; scaler fitted on the original training rows only."""')
(O/'train.py').write_text(s)
shutil.copy2(H/'train_augmentation_trees.py',O/'train_augmentation_trees.py')
variants={k:v for k,v in old['variants'].items() if k in ['residual_budget','mhc','single_pass']}
plan={'created_at':datetime.now(timezone.utc).isoformat(),'parent_architecture_plan_sha256':sha(OLD/'plan.json'),'generator_plan_sha256':sha(H/'learned/pilot-plan.json'),'generator_audit_sha256':sha(H/'learned/pilot-audit.json'),'adapter_parity_sha256':sha(H/'learned/adapter-parity.json'),'prepared_metadata_sha256':sha(O/'metadata.json'),
      'arms':['resampled_control','synthetic'],'variants':variants,'parameters':{k:old['parameters'][k] for k in variants},'seeds':old['seeds'],'training':old['training'],
      'source_hashes':{n:sha(O/n) for n in ['model.py','train.py','train_augmentation_trees.py']},
      'tree_parameters':{'n_estimators':226,'learning_rate':.03,'num_leaves':15,'max_depth':-1,'min_child_samples':20,'colsample_bytree':.8,'reg_lambda':5,'n_jobs':2,'verbosity':-1,'random_state':7329,'deterministic':True,'force_col_wise':True},
      'tree_inputs':{arm:sha(O/arm/'tree-arrays.npz') for arm in ['resampled_control','synthetic']},
      'comparison':'Both arms contain4209 original training views plus471 extra views with identical class counts and total optimizer steps. Control extras are resampled complete original training pairs, retaining their full/70%/40% views. Treatment replaces only those extra views with157 synthetic pairs. No held labels in any worker package. Normalization is identical, using original training rows only.',
      'primary_data_endpoints':['residual_budget','mhc','single_pass','tree897','tree1217'],
      'data_utility_gate':{'network':'Average paired log-loss reduction >=.001 with mean Pair AP delta>=0 and worst history mean AP delta>=-.001. All3 seeds and both histories reported.','tree':'Mean weighted pair/behavior delta>=.0001,worst>=-.003,mean sparse AP delta>=0.','comparison':'Synthetic versus matched resampling. Compare both arms with existing incumbent diagnostically. No best-seed or post-hoc blend search. Endpoints are multiple exploratory tests, not confirmatory statistical evidence.'},
      'progression':'Only passing endpoints justify a predeclared larger (~2x) diversity trial. No direct production admission. Any eventual candidate still needs existing omitted-family, history and audit gates. Do not proceed to10x solely because legal/source tests passed.',
      'limitations':['Only157 synthetic pairs; four related ordinary policy clones from one source.','Three soft-play targets had no qualifying intervention and were excluded.','Learned pilot has style/temperature variation but legacy tilt parameters are inactive. Additional innocent-behavior realism must be addressed before scaling.','Reused held pools; near-ceiling known-family metrics; no untouched estimate.'],
      'protected':old['protected']}
(O/'plan.json').write_text(json.dumps(plan,indent=2))
for arm in plan['arms']:
    audit={'plan_sha256':sha(O/'plan.json'),'train_arrays_sha256':sha(O/arm/'train-arrays.npz'),'held_labels_absent_from_training_arrays':True,'matched_rows':4680,'synthetic_rows':471 if arm=='synthetic' else 0,'scaler_original_only':True}
    (O/arm/'data-audit.json').write_text(json.dumps(audit,indent=2))
files=[O/n for n in ['plan.json','model.py','train.py']]+[O/arm/n for arm in plan['arms'] for n in ['train-arrays.npz','data-audit.json']]
m={p.relative_to(O).as_posix():sha(p) for p in files};(O/'remote-manifest.json').write_text(json.dumps(m,indent=2))
with tarfile.open(O/'package.tar.gz','w:gz') as t:
    for p in files+[O/'remote-manifest.json']:t.add(p,arcname=p.relative_to(O))
print(json.dumps({'plan_sha256':sha(O/'plan.json'),'package_sha256':sha(O/'package.tar.gz'),'bytes':(O/'package.tar.gz').stat().st_size}))
