"""Offline label-free staged shape benchmark for the frozen citation recipe."""
import os,sys,time,json,gc
from pathlib import Path
for k,v in {'TABPFN_DISABLE_TELEMETRY':'1','TABPFN_NO_BROWSER':'1','HF_HUB_OFFLINE':'1','POLARS_MAX_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import numpy as np,torch
from tabpfn import TabPFNClassifier
OUT=Path(__file__).resolve().parent;CP=Path(r'C:/Data/Projects/Money Making/work/poker/Codex Resume/research/tabpfn3_runtime_20260912/checkpoint/tabpfn-v3-classifier-v3_default.ckpt')
def net(event,args):
 if event=='socket.connect':raise RuntimeError('network forbidden')
sys.addaudithook(net);torch.set_num_threads(2);r=np.random.default_rng(2773);results=[]
for n,e,limit,secs in [(256,1,2.75,60),(512,1,2.75,60),(1024,1,2.75,60),(1024,4,3.0,120)]:
 x=r.normal(size=(n,64));y=np.tile([0,1],n//2);z=r.normal(size=(256,64));torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 m=TabPFNClassifier(model_path=str(CP),n_estimators=e,auto_scale_n_estimators=False,device='cuda:0',fit_mode='low_memory',memory_saving_mode=True,random_state=2773,n_preprocessing_jobs=1,show_progress_bar=False)
 m.fit(x,y);p=m.predict_proba(z);dt=time.perf_counter()-t;peak=torch.cuda.max_memory_reserved();passed=peak<=limit*1024**3 and dt<=secs and np.isfinite(p).all()
 row={'train_rows':n,'features':64,'query_rows':256,'estimators':e,'seconds':dt,'peak_reserved_bytes':peak,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'passed':bool(passed)};results.append(row);print(json.dumps(row),flush=True)
 del m,x,y,z,p;gc.collect();torch.cuda.empty_cache()
 if not passed:break
(OUT/'shape-benchmark.json').write_text(json.dumps({'all_passed':len(results)==4 and all(x['passed'] for x in results),'results':results},indent=2)+'\n')
