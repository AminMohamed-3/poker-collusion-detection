"""Synthetic-only local Nori smoke check; no poker data or evidence outcomes."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,time,json,hashlib
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from synthefy_nori import NoriRegressor
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'research/nori_runtime_20260912'
def deny_network(event,args):
 if event=='socket.connect':raise RuntimeError('Network connections disabled for local smoke test')
sys.addaudithook(deny_network)
def main():
 t=time.perf_counter();torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(72311);torch.use_deterministic_algorithms(True)
 rng=np.random.default_rng(72311);x=rng.normal(size=(136,38)).astype(np.float32);y=(np.log1p(np.exp(x[:128,0]))+.05*rng.normal(size=128)).astype(np.float32)
 model=NoriRegressor(model_path=str(OUT/'checkpoint/nori.pt'),device='cuda',token=False,augmentations=(),memory_policy='exact',categorical_columns=[])
 model.fit(x[:128],y);first=model.predict(x[128:],output_type='full');second=model.predict(x[128:],output_type='full')
 assert isinstance(first,dict);info={}
 for k,v in first.items():
  a=np.asarray(v);assert np.isfinite(a).all();info[k]={'shape':list(a.shape),'repeat_exact':bool(np.array_equal(a,np.asarray(second[k]))),'max_repeat_error':float(np.max(np.abs(a-np.asarray(second[k]))))}
 assert info['mean']['shape']==[8] and all(v['repeat_exact'] for v in info.values())
 report={'status':'complete','synthetic_context_rows':128,'synthetic_query_rows':8,'features':38,'output':info,'memory_report':model.memory_report_,'cuda_peak_mb':torch.cuda.max_memory_allocated()/2**20,'seconds':time.perf_counter()-t,'network_connections_disabled':True,'poker_data_read':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'smoke-receipt.json').write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8');print(json.dumps(report,indent=2,default=str),flush=True)
if __name__=='__main__':main()
