"""Qualified pretrained T production, fixed contexts and offline inference only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
for k,v in {'TABPFN_DISABLE_TELEMETRY':'1','TABPFN_NO_BROWSER':'1','HF_HUB_OFFLINE':'1'}.items():os.environ[k]=v
import sys,json,time,gc,hashlib,argparse,subprocess
from pathlib import Path
import numpy as np
import tabpfn3_hardware_resume as baseline
CHUNK=256
GUARD=int(13.5*1024**3)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);assert not p.exists(),p;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(d,indent=2)+'\n')
def check(root):
 f=read(root/'production-freeze.json')
 for name,h in f['files'].items():assert sha(root/name)==h,name
 assert sha(root/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==baseline.CP_SHA==f['checkpoint_sha256']
 assert f['params']==baseline.PARAMS and f['chunk']==CHUNK and f['memory_guard']==GUARD
 import tabpfn
 installed=Path(tabpfn.__file__).resolve().parent
 for name,h in f['installed_tabpfn_source_sha256'].items():assert sha(installed/name)==h,name
 env=baseline.runtime(root);assert env['gpu_capacity_bytes']>=15*1024**3
 import torch
 torch.set_num_threads(2)
 baseline.network_off()
 return f,read(root/'production-recipe.json'),env

def model_fit(root,family,recipe):
 from tabpfn import TabPFNClassifier
 m=TabPFNClassifier(model_path=str(root/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt'),**baseline.PARAMS)
 c=next(c for c in recipe['training'] if c['family']==family)
 with np.load(root/c['file'],allow_pickle=False) as z:
  assert z['xtrain'].dtype==np.float64 and set(z['ytrain'])=={0,1}
  m.fit(z['xtrain'],z['ytrain'])
  assert tuple(m.executor_.X_train.shape)==tuple(z['xtrain'].shape)==tuple(c['shape'])
 assert m.n_estimators==m.n_estimators_==len(m.ensemble_configs_)==8
 return m

def predict(m,x):
 ix=int(np.flatnonzero(m.classes_==1)[0]);v=m.predict_proba(x)[:,ix]
 assert len(v)==len(x) and np.isfinite(v).all() and np.all((v>=0)&(v<=1))
 return v

def clean(m):
 import torch
 del m;gc.collect();torch.cuda.empty_cache()

def benchmark(root):
 import torch
 f,r,env=check(root);out=root/'run';out.mkdir(exist_ok=True)
 result={'freeze_sha256':sha(root/'production-freeze.json'),'runtime':env,'families':{},'no_quality_metrics':True}
 assert not (out/'benchmark.json').exists()
 for fam in r['families']:
  dest=out/'benchmark'/fam;dest.mkdir(parents=True,exist_ok=False)
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  assert torch.cuda.mem_get_info()[0]>=14*1024**3,'Another GPU job is using the VM'
  t=time.monotonic();m=model_fit(root,fam,r);fit_seconds=time.monotonic()-t
  fp=dest/'context.tabpfn_fit';m.save_fit_state(fp)
  q=next(c for c in r['queries'] if c['family']==fam)
  with np.load(root/q['file'],allow_pickle=False) as z:x=z['xtest'][:2048]
  t=time.monotonic();pr=np.concatenate([predict(m,x[i:i+CHUNK]) for i in range(0,len(x),CHUNK)]);seconds=time.monotonic()-t
  peak=torch.cuda.max_memory_reserved();pp=dest/'probabilities.npy';np.save(pp,pr)
  info={'query_file':q['file'],'query_sha256':q['sha256'],'query_rows':len(x),'fit_seconds':fit_seconds,'predict_seconds':seconds,'rows_per_second':len(x)/seconds,'peak_reserved_bytes':peak,'fit_state_sha256':sha(fp),'probabilities_sha256':sha(pp),'memory_pass':peak<=GUARD}
  write(dest/'receipt.json',info);print(json.dumps({'benchmark_family':fam,**info}),flush=True)
  assert info['memory_pass'],'Memory ceiling exceeded; diagnostic preserved'
  result['families'][fam]=info
  del m;gc.collect();torch.cuda.empty_cache()
 result['projected_inference_seconds']=sum(sum(c['shape'][0] for c in r['queries'] if c['family']==fam)/v['rows_per_second'] for fam,v in result['families'].items())
 result['passed']=True;write(out/'benchmark.json',result)
 print(json.dumps({'benchmark_passed':True,'projected_seconds':result['projected_inference_seconds']}),flush=True)

def replay(root):
 import torch
 from tabpfn import TabPFNClassifier
 f,r,env=check(root);out=root/'run';bench=read(out/'benchmark.json');assert bench['freeze_sha256']==sha(root/'production-freeze.json') and bench['passed']
 result={}
 for fam in r['families']:
  info=bench['families'][fam];dest=out/'benchmark'/fam
  assert sha(dest/'probabilities.npy')==info['probabilities_sha256'] and sha(dest/'context.tabpfn_fit')==info['fit_state_sha256']
  original=np.load(dest/'probabilities.npy',allow_pickle=False)
  with np.load(root/info['query_file'],allow_pickle=False) as z:x=z['xtest'][:info['query_rows']]
  checks={};torch.cuda.reset_peak_memory_stats()
  for mode in ('saved_fit','fresh_fit'):
   m=TabPFNClassifier.load_from_fit_state(dest/'context.tabpfn_fit',device='cuda:0') if mode=='saved_fit' else model_fit(root,fam,r)
   assert m.n_estimators==m.n_estimators_==8
   p=np.concatenate([predict(m,x[i:i+CHUNK]) for i in range(0,len(x),CHUNK)])
   checks[mode+'_exact']=bool(np.array_equal(p,original));checks[mode+'_maximum_difference']=float(np.max(np.abs(p-original)))
   del m;gc.collect();torch.cuda.empty_cache()
  checks['peak_reserved_bytes']=torch.cuda.max_memory_reserved();result[fam]=checks
  write(out/'replay'/f'{fam}.json',checks)
  assert checks['saved_fit_exact'] and checks['fresh_fit_exact'] and checks['peak_reserved_bytes']<=GUARD,checks
  print(json.dumps({'replay_family':fam,**checks}),flush=True)
 write(out/'replay.json',{'freeze_sha256':sha(root/'production-freeze.json'),'benchmark_sha256':sha(out/'benchmark.json'),'passed':True,'families':result})

def produce(root):
 import torch
 from tabpfn import TabPFNClassifier
 f,r,env=check(root);out=root/'run';bench=read(out/'benchmark.json');rep=read(out/'replay.json')
 assert bench['passed'] and rep['passed'] and rep['freeze_sha256']==sha(root/'production-freeze.json') and rep['benchmark_sha256']==sha(out/'benchmark.json')
 assert bench['projected_inference_seconds']<=9000,'Capacity estimate over frozen production budget; inspect before continuing'
 results={};started=time.monotonic()
 for fam in r['families']:
  saved=out/'benchmark'/fam/'context.tabpfn_fit';assert sha(saved)==bench['families'][fam]['fit_state_sha256']
  m=TabPFNClassifier.load_from_fit_state(saved,device='cuda:0');assert m.n_estimators==m.n_estimators_==8
  for q in (q for q in r['queries'] if q['family']==fam):
   batch=Path(q['file']).parent.name;dest=out/'predictions'/batch/fam;dest.mkdir(parents=True,exist_ok=True);rp=dest/'receipt.json';pp=dest/'probabilities.npy'
   if rp.exists():
    z=read(rp);assert z['freeze_sha256']==sha(root/'production-freeze.json') and z['query_sha256']==q['sha256'] and z['probabilities_sha256']==sha(pp)
    results[batch+'/'+fam]=sha(rp);continue
   assert not pp.exists(),'Incomplete final output requires review'
   torch.cuda.reset_peak_memory_stats();t=time.monotonic()
   with np.load(root/q['file'],allow_pickle=False) as z:x=z['xtest']
   assert list(x.shape)==q['shape'] and x.dtype==np.float64
   parts=[];chunks={}
   for i in range(0,len(x),CHUNK):
    chunk=dest/'chunks'/f'{i:06d}.npy';cr=chunk.with_suffix('.json')
    if chunk.exists() or cr.exists():
     assert chunk.exists() and cr.exists(),'Incomplete chunk requires review'
     z=read(cr);assert z['sha256']==sha(chunk) and z['query_sha256']==q['sha256'] and z['freeze_sha256']==sha(root/'production-freeze.json')
     v=np.load(chunk,allow_pickle=False)
    else:
     v=predict(m,x[i:i+CHUNK]);chunk.parent.mkdir(exist_ok=True);np.save(chunk,v)
     write(cr,{'sha256':sha(chunk),'query_sha256':q['sha256'],'freeze_sha256':sha(root/'production-freeze.json')})
    assert len(v)==min(CHUNK,len(x)-i);parts.append(v);chunks[chunk.name]=sha(chunk)
   p=np.concatenate(parts);np.save(pp,p)
   info={'freeze_sha256':sha(root/'production-freeze.json'),'query_sha256':q['sha256'],'probabilities_sha256':sha(pp),'fit_state_sha256':sha(saved),'rows':len(p),'chunks':chunks,'seconds':time.monotonic()-t,'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
   write(rp,info);assert info['peak_reserved_bytes']<=GUARD
   results[batch+'/'+fam]=sha(rp)
   print(json.dumps({'completed_batch':batch,'family':fam,'rows':len(p),'seconds':info['seconds'],'completed':len(results),'total':len(r['queries'])}),flush=True)
  del m;gc.collect();torch.cuda.empty_cache()
 assert len(results)==60
 assert sha(root/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==f['checkpoint_sha256']
 write(out/'complete.json',{'freeze_sha256':sha(root/'production-freeze.json'),'receipts':results,'query_rows':sum(c['shape'][0] for c in r['queries']),'completed_batches':60,'seconds':time.monotonic()-started,'all_risk_and_behavior_unchanged_by_design':True})
 print('PRODUCTION_COMPLETE',flush=True)

def all_modes(root):
 for mode in ('benchmark','replay','produce'):
  q=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),mode,'--root',str(root)])
  assert q.returncode==0,(mode,q.returncode)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['benchmark','replay','produce','all']);a.add_argument('--root',type=Path,required=True);args=a.parse_args()
 {'benchmark':benchmark,'replay':replay,'produce':produce,'all':all_modes}[args.mode](args.root.resolve())
