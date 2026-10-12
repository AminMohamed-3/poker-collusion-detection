"""Prove local subset equivalence before generating a tiny transfer package."""
import argparse
import hashlib
import json
import shutil
import tarfile
import numpy as np
from robust_pair_experiment import ROOT,sha
from rebuild_mil_delta import select

p=argparse.ArgumentParser(); p.add_argument('--job',required=True); args=p.parse_args()
assert args.job.startswith('omit_')
report=ROOT/'reports/mil-pair-experiment'; parent=report/'fold0'; dest=report/args.job
oldmeta=json.loads((parent/'metadata.json').read_text()); newmeta=json.loads((dest/'metadata.json').read_text())
old=np.load(parent/'package/arrays.npz',allow_pickle=False); new=np.load(dest/'package/arrays.npz',allow_pickle=False)
indices={}
for group,tags in [('full',['full']),('shortened',['shortened']),('train',['train_full','train_70','train_40'])]:
    oldrows=[(t,r['pair_id']) for t in tags for r in oldmeta[t]['pairs']]; newrows=[(t,r['pair_id']) for t in tags for r in newmeta[t]['pairs']]; lookup={r:i for i,r in enumerate(oldrows)}
    indices[group]=np.array([lookup[r] for r in newrows],dtype=np.int64)
    rebuilt=select(old,group,indices[group])
    for k,v in rebuilt.items(): np.testing.assert_array_equal(v,new[k])
tiny=dest/'delta-package-v2'; tiny.mkdir(exist_ok=False); np.savez_compressed(tiny/'delta.npz',**indices)
plan={'parent_sha256':sha(parent/'package/arrays.npz'),'expected_file_sha256':sha(dest/'package/arrays.npz'),'key_order':new.files,'array_sha256':{k:hashlib.sha256(new[k].tobytes()).hexdigest() for k in new.files},'fortran_order':{k:bool(new[k].flags.f_contiguous and not new[k].flags.c_contiguous) for k in new.files},'all_local_arrays_match_exactly':True,'transfer_repair':'Preserve NPY Fortran-order metadata as well as exact numerical contents; no model/data changes.'}
(tiny/'delta-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for name in ['run_mil_pairs.py','mil_pair_model.py','worker-plan.json']: shutil.copy2(dest/'package'/name,tiny/name)
shutil.copy2(ROOT/'src/rebuild_mil_delta.py',tiny/'rebuild_mil_delta.py')
archive=dest/'delta-v2.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for path in tiny.iterdir(): t.add(path,arcname=path.name)
(dest/'delta-transfer-v2.json').write_text(json.dumps({'sha256':sha(archive),'bytes':archive.stat().st_size},indent=2)+'\n')
print(json.dumps({'job':args.job,'sha256':sha(archive),'bytes':archive.stat().st_size}))
