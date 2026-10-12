import os,sys,json,time,hashlib,gc,argparse
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
ROOT=Path(r'C:/Data/Projects/Money Making');OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/tabldm_screen';RT=ROOT/'work/poker/goal_top5_20260919/tabldm';CK=RT/'checkpoint/clf_default.ckpt'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=json.loads((OUT/'freeze.json').read_text());release=json.loads((OUT/'GPU_RELEASE.json').read_text()) if (OUT/'GPU_RELEASE.json').exists() else {};assert release.get('freeze_sha256')==sha(OUT/'freeze.json'),'hash-bound root GPU release absent'
for n,h in freeze['files'].items():assert sha(OUT/n)==h,(n,'hash mismatch')
runtime_paths={'PREFLIGHT':RT/'PREFLIGHT.md','synthetic_runtime':RT/'receipts/synthetic_runtime.json','provenance_manifest':RT/'receipts/provenance_manifest.json','classifier_source':RT/'overlay/tabldm/_sklearn/classifier.py'}
for n,p in runtime_paths.items():assert sha(p)==freeze['tabldm_runtime'][n],(n,'runtime provenance hash')
prov=json.loads((RT/'receipts/provenance_manifest.json').read_text());tracked=prov['official_source']['tracked_files_sha256']
for rel,h in tracked.items():
 if rel.startswith('tabldm/') and rel.endswith('.py'):assert sha(RT/'overlay'/rel)==h,(rel,'overlay source drift')
EXPECTED_PY=(ROOT/'work/poker/Codex Resume/.venv_tabpfn3/Scripts/python.exe').resolve();assert Path(sys.executable).resolve()==EXPECTED_PY,(sys.executable,'wrong interpreter');assert sha(CK)==freeze['checkpoint_sha256'];sys.path.insert(0,str(RT/'overlay'))
import numpy as np,torch,einops,sklearn,scipy
import tabldm
from tabldm import TabLDMClassifier
IMPORTED=Path(tabldm.__file__).resolve();assert RT/'overlay' in IMPORTED.parents,(IMPORTED,'wrong tabldm import')
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70);SEED=35935;CHUNK=256
def model():return TabLDMClassifier(n_estimators=4,batch_size=1,random_state=SEED,n_jobs=2,enhance_candidates=False,model_path=CK,allow_auto_download=False)
def predict(m,x,keys,name):
 assert len(x)==len(keys) and np.array_equal(keys,np.sort(keys));out=[];chunks=[]
 for i in range(0,len(x),CHUNK):
  j=min(i+CHUNK,len(x));q=x[i:j].copy();mask=np.flatnonzero(np.all(np.isnan(q),axis=0)).tolist();p=m.predict_proba(q);assert p.shape==(j-i,4) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=2e-6);out.append(p);chunks.append({'surface':name,'start':i,'stop':j,'first_pair_id':str(keys[i]),'last_pair_id':str(keys[j-1]),'all_nan_feature_indices':mask})
 return np.concatenate(out),chunks
def one(f):
 d=np.load(OUT/f'context_fold{f}.npz');p=np.load(OUT/f'context_fold{f}_prefix60.npz');assert np.array_equal(d['X_train'],p['X_train'],equal_nan=True) and np.array_equal(d['y_train'],p['y_train']);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.monotonic();m=model();m.fit(d['X_train'].copy(),d['y_train'].copy());assert np.array_equal(m.classes_,np.arange(4));a,ca=predict(m,d['X_held'],d['pair_id'],'full');b,cb=predict(m,p['X_held'],p['pair_id'],'prefix');peak=int(torch.cuda.max_memory_reserved());assert peak<3*1024**3;return a,b,ca+cb,time.monotonic()-t,peak
def main():
 a=argparse.ArgumentParser();a.add_argument('--replay',action='store_true');z=a.parse_args()
 if z.replay:
  assert (OUT/'predictions.npz').exists() and not (OUT/'replay.npz').exists() and not (OUT/'replay_receipt.json').exists();x,y,ch,s,pk=one(0);src=np.load(OUT/'predictions.npz');ef=np.array_equal(x,src['full_fold0']);ep=np.array_equal(y,src['prefix_fold0']);assert ef and ep;np.savez_compressed(OUT/'replay.npz',full=x,prefix=y);(OUT/'replay_receipt.json').write_text(json.dumps({'exact_full':ef,'exact_prefix':ep,'replay_sha256':sha(OUT/'replay.npz'),'predictions_sha256':sha(OUT/'predictions.npz'),'seconds':s,'peak_reserved':pk,'chunks':ch},indent=2)+'\n');return
 assert not (OUT/'predictions.npz').exists() and not (OUT/'runtime.json').exists();arr={};rec=[];start=time.monotonic()
 for f in (0,1):
  x,y,ch,s,pk=one(f);arr[f'full_fold{f}']=x;arr[f'prefix_fold{f}']=y;rec.append({'fold':f,'seconds':s,'peak_reserved':pk,'chunks':ch});gc.collect();torch.cuda.empty_cache()
  if len(rec)==1 and s*2>600:raise RuntimeError('projected runtime exceeds 10 minutes')
  if time.monotonic()-start>600:raise RuntimeError('cumulative runtime exceeds 10 minutes')
 for n,h in freeze['context_files'].items():assert sha(OUT/n)==h,(n,'source context mutated')
 np.savez_compressed(OUT/'predictions.npz',**arr);(OUT/'runtime.json').write_text(json.dumps({'predictions_sha256':sha(OUT/'predictions.npz'),'checkpoint_sha256':sha(CK),'python_executable':str(EXPECTED_PY),'pythonpath_overlay':str((RT/'overlay').resolve()),'imported_tabldm_module':str(IMPORTED),'einops_module':str(Path(einops.__file__).resolve()),'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'einops':einops.__version__},'verified_overlay_python_files':sum(k.startswith('tabldm/') and k.endswith('.py') for k in tracked),'enhance_candidates':False,'n_estimators':4,'batch_size':1,'seed':SEED,'n_jobs':2,'query_chunk':CHUNK,'memory_fraction':.70,'synthetic_256_vs_4x64_max_abs_diff':0.0013904571533203125,'total_seconds':time.monotonic()-start,'fits':rec},indent=2)+'\n')
if __name__=='__main__':main()
