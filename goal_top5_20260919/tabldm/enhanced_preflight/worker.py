import os,sys,json,time,hashlib,platform
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
ROOT=Path(r'C:/Data/Projects/Money Making');TD=ROOT/'work/poker/goal_top5_20260919/tabldm';OUT=TD/'enhanced_preflight';CK=TD/'checkpoint/clf_default.ckpt'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
man=json.loads((OUT/'MANIFEST.json').read_text());assert sha(CK)==man['checkpoint_sha256'];assert Path(sys.executable).resolve()==Path(man['python_executable']).resolve()
source_paths={'provenance_manifest':TD/'receipts/provenance_manifest.json','synthetic_runtime':TD/'receipts/synthetic_runtime.json','default_enhanced_attempt':TD/'receipts/default_enhanced_attempt.json','classifier':TD/'overlay/tabldm/_sklearn/classifier.py'}
for n,pth in source_paths.items():assert sha(pth)==man['source'][n],(n,'source drift')
prov=json.loads((TD/'receipts/provenance_manifest.json').read_text());verified=[]
for rel,h in prov['official_source']['tracked_files_sha256'].items():
 if rel.startswith('tabldm/') and rel.endswith('.py'):assert sha(TD/'overlay'/rel)==h,(rel,'overlay drift');verified.append(rel)
sys.path.insert(0,str(TD/'overlay'))
import numpy as np,torch,tabldm,einops,sklearn,scipy
from tabldm import TabLDMClassifier
assert sha(Path(tabldm.__file__).resolve())==man['tabldm_init_sha256'];torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70);assert torch.cuda.is_available()
rng=np.random.default_rng(35935+1000);xtr=rng.normal(size=(1000,128)).astype(np.float32);y=np.arange(1000,dtype=np.int64)%4;rng.shuffle(y);xq=rng.normal(size=(256,128)).astype(np.float32)
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter();print('WORKER_STAGE classifier_construct',flush=True)
m=TabLDMClassifier(n_estimators=4,batch_size=1,random_state=35935,n_jobs=2,enhance_candidates=True,model_path=CK,allow_auto_download=False)
print('WORKER_STAGE fit_begin',flush=True);m.fit(xtr,y);fit_s=time.perf_counter()-t0;print(f'WORKER_STAGE fit_end seconds={fit_s:.6f}',flush=True)
t1=time.perf_counter();p=m.predict_proba(xq.copy());pred_s=time.perf_counter()-t1;print(f'WORKER_STAGE predict_end seconds={pred_s:.6f}',flush=True)
peak_alloc=int(torch.cuda.max_memory_allocated());peak_res=int(torch.cuda.max_memory_reserved());assert peak_res<3*1024**3 and p.shape==(256,4) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=2e-6)
counts={'main':int(getattr(m,'_frozen_main_views_',4)),'quantile':int(getattr(m,'_frozen_q_views_',0)),'svd_ens':int(getattr(m,'_frozen_svd_ens_views_',0)),'adaptive_plus':int(getattr(m,'_frozen_ap_views_',0)),'gaussian_rank':int(getattr(m,'_frozen_gr_views_',0))};counts['total']=sum(counts.values());w=getattr(m,'nnls_weights_',None);cal=getattr(m,'calibration_params_',None)
rec={'status':'PASS','scope':'synthetic_only','config':{'n_train':1000,'n_features':128,'n_query':256,'n_estimators':4,'batch_size':1,'seed':35935,'n_jobs':2,'enhance_candidates':True,'enhancement_parameters':'official defaults'},'timing':{'fit_seconds':fit_s,'predict_seconds':pred_s,'total_seconds':time.perf_counter()-t0},'candidate_counts':counts,'verified_overlay_python_files':len(verified),'nnls':{'present':w is not None,'weight_count':int(len(w)) if w is not None else 0,'nonzero':int((w>0).sum()) if w is not None else 0,'sum':float(w.sum()) if w is not None else None},'calibration':{'present':cal is not None,'type':cal.get('type') if cal else None},'probability':{'shape':list(p.shape),'finite':True,'min':float(p.min()),'max':float(p.max()),'row_sum_max_abs_error':float(np.max(np.abs(p.sum(1)-1)))},'memory':{'torch_peak_allocated_bytes':peak_alloc,'torch_peak_reserved_bytes':peak_res},'runtime':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'einops':einops.__version__,'cuda_device':torch.cuda.get_device_name(0),'tabldm_module':str(Path(tabldm.__file__).resolve())},'hashes':{'checkpoint':sha(CK),'manifest':sha(OUT/'MANIFEST.json')}}
(OUT/'worker_result.json').write_text(json.dumps(rec,indent=2)+'\n');print('WORKER_STAGE receipt_written',flush=True)

