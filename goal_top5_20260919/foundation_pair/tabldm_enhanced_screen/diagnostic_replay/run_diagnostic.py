import os,sys,json,time,hashlib,importlib.util
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
ROOT=Path(r'C:/Data/Projects/Money Making'); PARENT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/tabldm_enhanced_screen'; OUT=PARENT/'diagnostic_replay'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
man=json.loads((OUT/'MANIFEST.json').read_text()); rel=json.loads((OUT/'DIAGNOSTIC_RELEASE.json').read_text()) if (OUT/'DIAGNOSTIC_RELEASE.json').exists() else {}; assert rel.get('manifest_sha256')==sha(OUT/'MANIFEST.json'),'diagnostic release absent'
for p,h in man['bound_files'].items():assert sha(ROOT/p)==h,(p,'hash mismatch')
for n in ('diagnostic_arrays.npz','diagnostic_receipt.json','comparison.json'):assert not (OUT/n).exists(),(n,'no overwrite')
spec=importlib.util.spec_from_file_location('frozen_runner',PARENT/'run_tabldm.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
import numpy as np,torch
from scipy.stats import rankdata,spearmanr
d=np.load(PARENT/'context_fold0.npz');p=np.load(PARENT/'context_fold0_prefix60.npz');torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.monotonic();m=mod.model();m.fit(d['X_train'].copy(),d['y_train'].copy());assert np.array_equal(m.classes_,np.arange(4));
counts={'main':int(getattr(m,'_frozen_main_views_',4)),'adaptive_plus':int(getattr(m,'_frozen_ap_views_',0)),'gaussian_rank':int(getattr(m,'_frozen_gr_views_',0))};assert counts=={'main':4,'adaptive_plus':16,'gaussian_rank':16}
full,cf=mod.predict(m,d['X_held'],d['pair_id'],'full');prefix,cp=mod.predict(m,p['X_held'],p['pair_id'],'prefix');w=np.asarray(m.nnls_weights_);mask=np.asarray(m.nnls_valid_candidate_mask_);cal=m.calibration_params_;assert cal['type']=='vector'
np.savez_compressed(OUT/'diagnostic_arrays.npz',full=full,prefix=prefix,nnls_weights=w,nnls_valid_candidate_mask=mask,calibration_scale=np.asarray(cal['scale']),calibration_bias=np.asarray(cal['bias']))
receipt={'seconds':time.monotonic()-t,'peak_reserved':int(torch.cuda.max_memory_reserved()),'candidate_counts':counts,'chunks':cf+cp,'arrays_sha256':sha(OUT/'diagnostic_arrays.npz'),'bound_manifest_sha256':sha(OUT/'MANIFEST.json')};(OUT/'diagnostic_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
saved=np.load(PARENT/'predictions.npz');fresh=np.load(OUT/'diagnostic_arrays.npz');result={'strict_original_gate_remains_failed':True,'thresholds':{'max_abs_probability':1e-5,'mean_abs_probability':1e-6,'argmax_changes':0},'surfaces':{}}
for name,key in [('full','full_fold0'),('prefix','prefix_fold0')]:
 a=np.asarray(saved[key],float);b=np.asarray(fresh[name],float);diff=np.abs(a-b);ra=1-a[:,0];rb=1-b[:,0];rka=rankdata(ra,method='average')/len(ra);rkb=rankdata(rb,method='average')/len(rb)
 cell={'shape':list(a.shape),'exact_mismatching_values':int(np.count_nonzero(a!=b)),'max_abs_probability':float(diff.max()),'mean_abs_probability':float(diff.mean()),'argmax_changes':int(np.count_nonzero(np.argmax(a,axis=1)!=np.argmax(b,axis=1))),'risk_rank_spearman':float(spearmanr(ra,rb).statistic),'max_normalized_risk_rank_movement':float(np.max(np.abs(rka-rkb)))};cell['within_prospective_noise_thresholds']=cell['max_abs_probability']<=1e-5 and cell['mean_abs_probability']<=1e-6 and cell['argmax_changes']==0;result['surfaces'][name]=cell
result['technical_noise_classification']=all(x['within_prospective_noise_thresholds'] for x in result['surfaces'].values());result['diagnostic_arrays_sha256']=sha(OUT/'diagnostic_arrays.npz');result['diagnostic_receipt_sha256']=sha(OUT/'diagnostic_receipt.json');(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
