"""Pre-quality, label-blind verification of returned Global-PU inference."""
import hashlib,json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;B=O/'vm_bundle';C=B/'contexts';P=B/'outputs';EXPECTED_BUNDLE='5522436d8408e54e6e0aa6f42ccb2cf91b2e0d4d24c7c7330d58eada37504555';EXPECTED_CK='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert not (O/'global_pu_results.json').exists(),'Quality result already exists; verifier must precede label read'
 assert not (O/'returned_verified.json').exists(),'Refusing to overwrite verification sentinel'
 z=O/'global_pu_fast_vm_bundle.zip';assert sha(z)==EXPECTED_BUNDLE
 manifest=json.loads((B/'MANIFEST.json').read_text());assert manifest['checkpoint']['sha256']==EXPECTED_CK and not manifest['checkpoint']['included'] and not manifest['held_truth_included']
 for x in manifest['files']:
  p=B/x['path'];assert p.exists() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
 receipt=json.loads((O/'context_receipt.json').read_text());checked=[]
 for fr in receipt['folds']:
  f=fr['fold'];assert sha(C/f'train_fold{f}.npz')==fr['train_sha256'];rt=json.loads((P/f'runtime_fold{f}.json').read_text());assert rt['seed']==35935 and rt['members']==4 and rt['batch']==256 and rt['fit_mode']=='low_memory' and rt['memory_saving_mode'] is True and rt['checkpoint_sha256']==EXPECTED_CK and rt['train_sha256']==fr['train_sha256']
  expected={x['file'].replace('query_','pred_'):x for x in fr['shards']};seen={x['file']:x for x in rt['shards']};assert set(seen)==set(expected)
  total=0
  for name,e in expected.items():
   qname=name.replace('pred_','query_');q=np.load(C/qname,allow_pickle=False);dest=P/name;assert sha(C/qname)==e['sha256'];assert dest.exists();d=np.load(dest,allow_pickle=False);assert np.array_equal(d['pair_id'],q['pair_id']) and np.array_equal(d['table_id'],q['table_id']);assert np.array_equal(d['classes'],np.arange(4,dtype=np.int8));p=d['probabilities'];assert p.shape==(e['rows'],4) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-5);assert seen[name]['sha256']==sha(dest);total+=len(p)
  assert total==fr['held_rows'];rep=json.loads((P/f'replay_fold{f}.json').read_text());assert rep['fold']==f and rep['exact'] is True and rep['max_abs']==0.0;checked.append({'fold':f,'rows':total,'shards':len(expected),'runtime_sha256':sha(P/f'runtime_fold{f}.json'),'replay_sha256':sha(P/f'replay_fold{f}.json')})
 out={'verified':True,'pre_quality':True,'bundle_sha256':EXPECTED_BUNDLE,'checkpoint_sha256':EXPECTED_CK,'manifest_sha256':sha(B/'MANIFEST.json'),'folds':checked};(O/'returned_verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
