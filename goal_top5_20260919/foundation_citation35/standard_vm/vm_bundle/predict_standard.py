"""One frozen inference run or one fresh first-64 replay; held truth is absent."""
import argparse,hashlib,json,os,time,gc
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TABPFN_NO_BROWSER='1')
import numpy as np
import torch
from tabpfn import TabPFNClassifier
H=Path(__file__).resolve().parent;CTX=H/'context.npz';MODEL=H/'tabpfn-v3.5-20260909_multiclass.safetensors';OUT=H/'outputs';OUT.mkdir(exist_ok=True)
EXPECTED='4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149';SEED=2773;BATCH=64
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def make():return TabPFNClassifier(model_path=str(MODEL),device='cuda',n_estimators=4,auto_scale_n_estimators=False,fit_mode='low_memory',memory_saving_mode=True,random_state=SEED,n_preprocessing_jobs=1)
def prob(m,x):
 j=int(np.flatnonzero(np.asarray(m.classes_)==1)[0]);return np.concatenate([m.predict_proba(x[i:i+BATCH])[:,j] for i in range(0,len(x),BATCH)])
def fit(z):
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.monotonic();m=make();m.fit(z['xfull'],z['yfull']);return m,time.monotonic()-t
def run():
 z=np.load(CTX,allow_pickle=False);m,fit_s=fit(z);t=time.monotonic();first=prob(m,z['xquery'][:64]);q_s=time.monotonic()-t;peak=int(torch.cuda.max_memory_reserved());projected=fit_s+q_s/64*len(z['xquery']);gate={'fit_seconds':fit_s,'first64_seconds':q_s,'projected_total_seconds':projected,'peak_reserved_bytes':peak,'time_limit_seconds':900,'memory_limit_bytes':16*1024**3}
 (OUT/'timing_gate.json').write_text(json.dumps(gate,indent=2)+'\n');print(gate,flush=True)
 if projected>900 or peak>=16*1024**3:raise RuntimeError('Frozen time/memory gate failed; no retry or tuning')
 t=time.monotonic();p=prob(m,z['xquery']);dest=OUT/'standard_predictions.npz';assert not dest.exists();np.savez_compressed(dest,primary=p,pair_ids=z['pair_ids'],hand_ids=z['hand_ids'],classes=np.asarray(m.classes_,dtype=np.int8));rec={'seconds_full_query':time.monotonic()-t,'fit_seconds':fit_s,'peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'checkpoint_sha256':sha(MODEL),'context_sha256':sha(CTX),'prediction_sha256':sha(dest),'seed':SEED,'members':4,'batch':BATCH,'fit_mode':'low_memory','memory_saving_mode':True};(OUT/'runtime.json').write_text(json.dumps(rec,indent=2)+'\n')
def replay():
 z=np.load(CTX,allow_pickle=False);old=np.load(OUT/'standard_predictions.npz',allow_pickle=False);assert np.array_equal(old['pair_ids'],z['pair_ids']) and np.array_equal(old['hand_ids'],z['hand_ids']);m,_=fit(z);p=prob(m,z['xquery'][:64]);check={'exact':bool(np.array_equal(p,old['primary'][:64])),'max_abs':float(np.max(np.abs(p-old['primary'][:64])))};(OUT/'replay_first64.json').write_text(json.dumps(check,indent=2)+'\n');assert check['exact'],check
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['run','replay']);z=a.parse_args();assert torch.cuda.is_available() and sha(MODEL)==EXPECTED;torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70);run() if z.mode=='run' else replay()
if __name__=='__main__':main()
