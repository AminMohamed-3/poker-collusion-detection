"""Label-free bounded TabPFN3 feasibility preflight; no network and no poker data."""
import os,sys,time,json,gc
from pathlib import Path
for k,v in {'TABPFN_DISABLE_TELEMETRY':'1','TABPFN_NO_BROWSER':'1','HF_HUB_OFFLINE':'1','POLARS_MAX_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import numpy as np,torch
from tabpfn import TabPFNClassifier
cp=Path(r'C:/Data/Projects/Money Making/work/poker/Codex Resume/research/tabpfn3_runtime_20260912/checkpoint/tabpfn-v3-classifier-v3_default.ckpt')
def audit(event,args):
 if event=='socket.connect':raise RuntimeError('network forbidden')
sys.addaudithook(audit);torch.set_num_threads(2);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
r=np.random.default_rng(2773);x=r.normal(size=(64,64));y=np.tile([0,1],32);z=r.normal(size=(32,64));t=time.perf_counter()
m=TabPFNClassifier(model_path=str(cp),n_estimators=1,auto_scale_n_estimators=False,device='cuda:0',fit_mode='low_memory',memory_saving_mode=True,random_state=2773,n_preprocessing_jobs=1,show_progress_bar=False)
m.fit(x,y);p=m.predict_proba(z);out={'success':True,'shape_train':[64,64],'shape_query':[32,64],'n_estimators':1,'seconds':time.perf_counter()-t,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'finite':bool(np.isfinite(p).all())}
Path(__file__).with_name('synthetic-preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));del m;gc.collect();torch.cuda.empty_cache()
