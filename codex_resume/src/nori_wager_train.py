"""Frozen matched conditional quantiles. Backends run sequentially in isolated runtimes."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['SYNTHEFY_DISABLE_PIPELINE_BATCHING']='1';os.environ['SYNTHEFY_FORBID_SUBSAMPLE']='1'
import sys,time,json,hashlib,gc
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'research/nori_wager_20260912';RUNTIME=HERE/'research/nori_runtime_20260912'
ALPHAS=np.array([.1,.5,.9]);POLICY={'cache':True,'allow_quantization':False,'allow_subsample':False,'offload_to_host':False,'elements_budget':87552}
TREE={'n_estimators':200,'learning_rate':.05,'num_leaves':31,'max_depth':5,'min_child_samples':20,'reg_lambda':2,'n_jobs':2,'random_state':72311,'verbosity':-1,'objective':'quantile'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,default=str)+'\n',encoding='utf-8')
def verify():
 cf=json.loads((OUT/'corpus-freeze.json').read_text());assert cf['status']=='complete'
 for p,h in cf['source_hashes'].items():assert sha(p)==h
 for p,h in cf['artifacts'].items():assert sha(OUT/p)==h
 tf=json.loads((OUT/'training-freeze.json').read_text())
 for p,h in tf['source_hashes'].items():assert sha(p)==h
 return cf

def freeze():
 assert not (OUT/'training-freeze.json').exists()
 src=[Path(__file__),OUT/'corpus-freeze.json',HERE/'research/nori_wager_protocol_20260912.md',RUNTIME/'checkpoint/nori.pt',RUNTIME/'runtime-receipt.json',RUNTIME/'benchmark-receipt.json',RUNTIME/'synthefy_nori-0.20.1-py3-none-any.whl']
 assert sha(RUNTIME/'checkpoint/nori.pt')=='a13b2bc31d8db24d17bae6d04844e0adf669e446087b0b7a34c7b05045d61323'
 write(OUT/'training-freeze.json',{'source_hashes':{str(p):sha(p) for p in src},'arms':['tree_quantiles','nori'],'primary':'nori','policy':POLICY,'tree_parameters':TREE,'alphas':ALPHAS.tolist(),'seed':72311,'context_rows':2048,'held_rows':1024,'no_outcomes_read':True})
 print('TRAINING FROZEN',flush=True)

def extract_quantiles(full):
 bank=np.asarray(full['quantiles'],dtype=float);taus=np.asarray(full['taus'],dtype=float)
 assert bank.ndim==2 and bank.shape[1]==len(taus)==999 and np.isfinite(bank).all() and np.all(np.diff(taus)>0) and taus[0]<.1 and taus[-1]>.9
 result=np.empty((len(bank),3))
 for j,a in enumerate(ALPHAS):
  i=int(np.searchsorted(taus,a));w=(a-taus[i-1])/(taus[i]-taus[i-1]);result[:,j]=bank[:,i-1]*(1-w)+bank[:,i]*w
 assert np.isfinite(result).all()
 return result

def diagnostics(y,p,upper):
 q=np.clip(np.sort(p,axis=1),0,upper[:,None]);r=y[:,None]-q
 return {'pinball':float(np.maximum(ALPHAS*r,(ALPHAS-1)*r).mean()),'pinball_by_quantile':np.maximum(ALPHAS*r,(ALPHAS-1)*r).mean(0).tolist(),'median_mae':float(np.abs(y-q[:,1]).mean()),'coverage80':float(((y>=q[:,0])&(y<=q[:,2])).mean()),'width80':float((q[:,2]-q[:,0]).mean())}

def run(arm):
 cf=verify();normal=np.load(OUT/'normal-corpus.npz',allow_pickle=False);score=np.load(OUT/'score-corpus.npz',allow_pickle=False);n=len(score['x'])
 if arm=='nori':
  import torch
  from synthefy_nori import NoriRegressor
  torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
  def deny_network(event,args):
   if event=='socket.connect':raise RuntimeError('Network connections disabled during local poker inference')
  sys.addaudithook(deny_network)
 else:
  import lightgbm as lgb
 for k in range(5):
  receipt=OUT/f'{arm}-{k}-receipt.json';assert not receipt.exists(),'write-once model receipts; inspect any partial run before resuming'
  t=time.perf_counter();tr=np.load(OUT/f'train-{k}-indices.npy');te=np.load(OUT/f'held-{k}-indices.npy');xt=normal['x'][tr];yt=normal['y'][tr];xq=np.concatenate([score['x'],normal['x'][te]],axis=0);info={}
  assert xt.shape==(2048,38) and len(te)==1024
  if arm=='tree_quantiles':
   pp=[]
   for a in ALPHAS:
    m=lgb.LGBMRegressor(**TREE,alpha=float(a)).fit(xt,yt);pred=m.predict(xq);path=OUT/f'tree-{k}-{a}.txt';m.booster_.save_model(str(path));replay=lgb.Booster(model_file=str(path)).predict(xq,num_threads=2);assert np.array_equal(pred,replay);pp.append(pred)
   p=np.column_stack(pp);info['model_hashes']={str(OUT/f'tree-{k}-{a}.txt'):sha(OUT/f'tree-{k}-{a}.txt') for a in ALPHAS}
  else:
   checks=[]
   for attempt in range(2):
    gc.collect();torch.cuda.empty_cache();torch.manual_seed(72311);np.random.seed(72311);torch.cuda.reset_peak_memory_stats()
    m=NoriRegressor(model_path=str(RUNTIME/'checkpoint/nori.pt'),device='cuda',token=False,augmentations=(),memory_policy=POLICY,categorical_columns=[])
    m.fit(xt,yt);raw=m.predict(xq,output_type='full');q=extract_quantiles(raw);mem=m.memory_report_
    assert mem['dropped_context_rows']==0 and mem['rung']=='resident_bf16' and mem['cache_dtype']=='bf16' and mem['query_chunk']==256
    assert len(mem['attempt_history'])==8 and set(sum([h['pipeline_ids'] for h in mem['attempt_history']],[]))==set(range(8))
    assert all(h['outcome']=='success' and h['dropped_context_rows']==0 and h['query_chunk']==256 and h['rung']=='resident_bf16' for h in mem['attempt_history'])
    checks.append({'memory_report':mem,'cuda_peak_mb':torch.cuda.max_memory_allocated()/2**20})
    if attempt==0:p=q.copy()
    else:assert np.array_equal(p,q),'New-model replay must match exactly before evidence evaluation'
    del m,raw,q;gc.collect();torch.cuda.empty_cache()
   info['runtime_checks']=checks
  assert p.shape==(n+1024,3) and np.isfinite(p).all()
  np.save(OUT/f'{arm}-{k}-quantiles.npy',p[:n],allow_pickle=False);np.save(OUT/f'{arm}-{k}-held-quantiles.npy',p[n:],allow_pickle=False)
  r={'arm':arm,'fold':k,'quantiles_sha256':sha(OUT/f'{arm}-{k}-quantiles.npy'),'held_quantiles_sha256':sha(OUT/f'{arm}-{k}-held-quantiles.npy'),'reload_exact':True,'train_rows':len(tr),'held_rows':len(te),'score_rows':n,'held_diagnostics':diagnostics(normal['y'][te],p[n:],normal['upper'][te]),'seconds':time.perf_counter()-t,**info}
  write(receipt,r);print(json.dumps({z:r[z] for z in ['arm','fold','seconds','held_diagnostics']}),flush=True)
 verify();print(arm,'COMPLETE',flush=True)

def finish():
 cf=verify();rr=[json.loads((OUT/f'{a}-{k}-receipt.json').read_text()) for a in ['tree_quantiles','nori'] for k in range(5)]
 for r in rr:
  a,k=r['arm'],r['fold'];assert r['reload_exact'] and sha(OUT/f'{a}-{k}-quantiles.npy')==r['quantiles_sha256']
  assert sha(OUT/f'{a}-{k}-held-quantiles.npy')==r['held_quantiles_sha256']
 assert not (OUT/'training-receipt.json').exists()
 report={'status':'complete','receipts':rr,'training_freeze_sha256':sha(OUT/'training-freeze.json'),'references':10,'tree_boosters':15,'nori_contexts':5,'no_evidence_outcomes_read':True,'held_diagnostics':{a:{metric:float(np.mean([r['held_diagnostics'][metric] for r in rr if r['arm']==a])) for metric in ['pinball','median_mae','coverage80','width80']} for a in ['tree_quantiles','nori']}}
 write(OUT/'training-receipt.json',report);print(json.dumps(report['held_diagnostics'],indent=2),flush=True)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='freeze':freeze()
 elif mode=='finish':finish()
 else:assert mode in ['tree_quantiles','nori'];run(mode)
