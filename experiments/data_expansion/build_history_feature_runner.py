"""Derive a read-only-feature build from the frozen learned-v2 extractor."""
from pathlib import Path
import hashlib
H=Path(__file__).resolve().parent
p=H/'prepare_history_features.py';assert not p.exists()
source=(H/'prepare_augmentation.py').read_text()
expected='507d4b4ee83c54a5fd539ac5fa8ee3716cb76cc337cf859f0a98ea2b65eea193'
assert hashlib.sha256((H/'prepare_augmentation.py').read_bytes()).hexdigest()==expected
source=source[:source.index('    original=dict(np.load')]
source=source.replace("D=H/'learned';O=H/'augmentation'","D=H/'history_repair';O=D/'features'")
source=source.replace(";assert audit['scale_up_admitted']", "")
source=source.replace('assert pair.height==157','assert pair.height==labels.height')
source+='''    np.savez_compressed(O/'synthetic-pairs.npz',x=np.concatenate(syn),y=np.tile(np.array([LABELS[r['behavior_family']] for r in synmeta[0]],dtype=np.int64),3))
    (O/'metadata.json').write_text(json.dumps({'views':synmeta,'features':names1217,'source_sha256':sha(__file__),'generator_plan_sha256':sha(D/'pilot-plan.json'),'no_held_labels_or_new_model_fits':True},indent=2))
    print('Built fresh history-repair features; no model fitting or held scoring.',flush=True)

if __name__=='__main__':main()
'''
p.write_text(source)
print(hashlib.sha256(p.read_bytes()).hexdigest())
