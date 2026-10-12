"""Label-blind returned-output verifier; run before evaluate_standard.py."""
import hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;B=H/'vm_bundle';O=B/'outputs';BUNDLE='4a9f11b35672cd48f1da3aa07ea435614c68f8ef66f6b84119fc0f6e656b5c2c';CK='4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert not (H/'standard_result.json').exists() and not (H/'returned_verified.json').exists();assert sha(H/'standard_citation_vm_bundle.zip')==BUNDLE
 m=json.loads((B/'MANIFEST.json').read_text());assert sha(B/'context.npz')==m['portable_context_sha256'] and m['checkpoint']['sha256']==CK
 r=json.loads((O/'runtime.json').read_text());assert r['checkpoint_sha256']==CK and r['context_sha256']==m['portable_context_sha256'] and r['seed']==2773 and r['members']==4 and r['batch']==64 and r['fit_mode']=='low_memory' and r['memory_saving_mode'] is True and r['peak_reserved_bytes']<16*1024**3
 g=json.loads((O/'timing_gate.json').read_text());assert g['projected_total_seconds']<=900 and g['peak_reserved_bytes']<16*1024**3
 z=np.load(B/'context.npz',allow_pickle=False);p=np.load(O/'standard_predictions.npz',allow_pickle=False);assert r['prediction_sha256']==sha(O/'standard_predictions.npz') and np.array_equal(p['pair_ids'],z['pair_ids']) and np.array_equal(p['hand_ids'],z['hand_ids']) and np.array_equal(p['classes'],np.array([0,1],dtype=np.int8)) and p['primary'].shape==(940,) and np.isfinite(p['primary']).all() and ((p['primary']>=0)&(p['primary']<=1)).all();rep=json.loads((O/'replay_first64.json').read_text());assert rep['exact'] and rep['max_abs']==0
 out={'verified':True,'pre_quality':True,'bundle_sha256':BUNDLE,'prediction_sha256':r['prediction_sha256'],'checkpoint_sha256':CK};(H/'returned_verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
