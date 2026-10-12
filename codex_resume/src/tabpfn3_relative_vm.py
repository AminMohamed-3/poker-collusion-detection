"""Offline VM inference for the frozen T versus T+RZ selection study."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import tabpfn3_hardware_resume as helper
HELPER_SHA='61738e64328370116de7f19686731704fe921d3fbea91164f49a61967dd44603'
assert helper.sha(Path(helper.__file__))==HELPER_SHA
sha=helper.sha;read=helper.read;write=helper.write

def assets(root):
 inv=read(root/'transfer-inventory.json')
 assert isinstance(inv,dict) and all(isinstance(h,str) and len(h)==64 for h in inv.values()),'new flat filename-to-SHA256 inventory schema'
 for n,h in inv.items():assert sha(root/n)==h,n
 ctx=read(root/'context-receipt.json');assert ctx['preflight_sha256']==sha(root/'preflight.json') and len(ctx['contexts'])==90 and ctx['held_labels_omitted']
 assert sha(root/'normal-reference-receipt.json')==ctx['normal_reference_receipt_sha256']
 for n,h in read(root/'normal-reference-receipt.json')['outputs'].items():assert sha(root/n)==h,n
 for m in ctx['contexts']:
  path=root/m['context_file'];assert sha(path)==m['context_sha256']
  with np.load(path,allow_pickle=False) as c:
   assert set(c.files)=={'xtrain','ytrain','xtest','held_indices','held_pair_ids','groups_train','groups_held'}
   assert c['xtrain'].dtype==c['xtest'].dtype==np.float64
   assert c['xtrain'].shape==tuple(m['train_shape']) and c['xtest'].shape==tuple(m['query_shape'])
   assert set(c['ytrain'])=={0,1} and not set(c['groups_train'])&set(c['groups_held'])
 assert sha(root/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==helper.CP_SHA
 return ctx

def freeze(root,out):
 ctx=assets(root);env=helper.runtime(root);out.mkdir(parents=True,exist_ok=True)
 replay_m=next(m for m in ctx['contexts'] if m['arm']=='RZ' and m['family']=='directed_transfer')
 assert replay_m['train_shape'][1]==replay_m['query_shape'][1]==599
 replay=replay_m['stem']
 write(out/'freeze.json',{'source_sha256':sha(Path(__file__)),'helper_sha256':HELPER_SHA,'inventory_sha256':sha(root/'transfer-inventory.json'),'context_receipt_sha256':sha(root/'context-receipt.json'),'checkpoint_sha256':helper.CP_SHA,'params':helper.PARAMS,'query_chunk':helper.CHUNK,'runtime':env,'replay_stem':replay,'no_predictions':True})
 print(json.dumps({'prepared':True,'contexts':len(ctx['contexts']),'gpu':env}),flush=True)

def check(root,out):
 f=read(out/'freeze.json');assert f['source_sha256']==sha(Path(__file__)) and f['helper_sha256']==HELPER_SHA
 assert f['inventory_sha256']==sha(root/'transfer-inventory.json') and f['context_receipt_sha256']==sha(root/'context-receipt.json')
 assert f['checkpoint_sha256']==helper.CP_SHA and f['params']==helper.PARAMS and f['query_chunk']==helper.CHUNK and f['runtime']==helper.runtime(root)
 return f,assets(root)

def benchmark(root,out):
 f,ctx=check(root,out);rng=np.random.default_rng(7329);n,q,w=ctx['max_context'],ctx['max_query'],ctx['max_width']
 x=rng.normal(size=(n,w));z=rng.normal(size=(q,w));y=np.arange(n,dtype=np.int64)%2;rng.shuffle(y);helper.network_off()
 p,info=helper.infer(root,{'xtrain':x,'ytrain':y,'xtest':z},f['runtime']['memory_ceiling_bytes'],out/'benchmark-diagnostic.json')
 write(out/'benchmark.json',{'passed':True,'synthetic_only':True,'freeze_sha256':sha(out/'freeze.json'),'runtime':info,'probabilities_sha256':hashlib.sha256(p.tobytes()).hexdigest()})
 print(json.dumps({'benchmark_pass':True,**info}),flush=True)

def fit(root,out):
 f,ctx=check(root,out);b=read(out/'benchmark.json');assert b['passed'] and b['freeze_sha256']==sha(out/'freeze.json');helper.network_off();dest=out/'predictions';dest.mkdir(exist_ok=True)
 for m in ctx['contexts']:
  pp=dest/(m['stem']+'.npy');rr=dest/(m['stem']+'.json');dd=dest/(m['stem']+'-diagnostic.json')
  if pp.exists() or rr.exists() or dd.exists():
   assert pp.exists() and rr.exists() and dd.exists(),'orphan/failed context; stop for review'
   r=read(rr);assert r['freeze_sha256']==sha(out/'freeze.json') and r['probabilities_sha256']==sha(pp) and r['diagnostic_sha256']==sha(dd) and r['context_sha256']==m['context_sha256'];continue
  with np.load(root/m['context_file'],allow_pickle=False) as c:p,info=helper.infer(root,c,f['runtime']['memory_ceiling_bytes'],dd)
  np.save(pp,p);write(rr,{'freeze_sha256':sha(out/'freeze.json'),'context_sha256':m['context_sha256'],'probabilities_sha256':sha(pp),'diagnostic_sha256':sha(dd),'runtime':info})
  print(json.dumps({'context':m['stem'],**info}),flush=True)
 write(out/'fit-complete.json',{'freeze_sha256':sha(out/'freeze.json'),'receipts':{m['stem']:sha(dest/(m['stem']+'.json')) for m in ctx['contexts']}})

def replay(root,out):
 f,ctx=check(root,out);done=read(out/'fit-complete.json');assert done['freeze_sha256']==sha(out/'freeze.json');m=next(m for m in ctx['contexts'] if m['stem']==f['replay_stem']);helper.network_off()
 assert m['train_shape'][1]==m['query_shape'][1]==599
 rr=out/'predictions'/(m['stem']+'.json');assert sha(rr)==done['receipts'][m['stem']];pp=out/'predictions'/(m['stem']+'.npy');assert sha(pp)==read(rr)['probabilities_sha256']
 with np.load(root/m['context_file'],allow_pickle=False) as c:p,info=helper.infer(root,c,f['runtime']['memory_ceiling_bytes'],out/'replay-diagnostic.json');ids=c['held_pair_ids']
 original=np.load(pp,allow_pickle=False);exact=all(np.array_equal(np.argsort(-p[ids==k],kind='stable'),np.argsort(-original[ids==k],kind='stable')) for k in np.unique(ids))
 np.save(out/'replay.npy',p);write(out/'replay.json',{'freeze_sha256':sha(out/'freeze.json'),'replay_sha256':sha(out/'replay.npy'),'ranking_exact':exact,'probabilities_exact':bool(np.array_equal(p,original)),'maximum_difference':float(np.max(np.abs(p-original))),'runtime':info});assert exact
 print(json.dumps({'replay_exact':exact,'max_diff':float(np.max(np.abs(p-original)))}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['freeze','benchmark','fit','replay']);a.add_argument('--root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);v=a.parse_args();globals()[v.mode](v.root.resolve(),v.out.resolve())
