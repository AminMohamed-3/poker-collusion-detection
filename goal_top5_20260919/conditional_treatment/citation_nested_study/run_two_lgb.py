"""Exactly two frozen CPU fits. Hard-blocked without root ADVANCE.json."""
import hashlib,json,os,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[k]='2'
import joblib,numpy as np,polars as pl
from lightgbm import LGBMClassifier
O=Path(__file__).resolve().parent;UP=O.parent/'citation_nested';K=['pair_id','hand_id']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert (O/'PREFLIGHT_OK.json').exists(),'Run clean preflight first';assert not (O/'predictions_frozen.json').exists() and not (O/'results.json').exists()
 pf=json.loads((O/'PREFLIGHT_OK.json').read_text());adv=json.loads((O/'ADVANCE.json').read_text());assert adv.get('root_approved') is True and adv['clean_manifest_sha256']==pf['clean_manifest_sha256'] and adv['artifact_sha256']==pf['artifact_sha256']
 assert sha(UP/'clean_manifest.json')==pf['clean_manifest_sha256'], 'Manifest changed after preflight'
 m=json.loads((UP/'clean_manifest.json').read_text());assert sha(UP/m['artifact'])==pf['artifact_sha256'], 'Artifact changed after preflight'
 d=pl.read_parquet(UP/m['artifact']);tr=d.filter(pl.col('split')=='train').sort(K);te=d.filter(pl.col('split')=='audit').sort(K);y=tr['target'].to_numpy().astype(np.int8);arms={'support_control':m['raw_features']+m['support_features'],'conditional_candidate':m['raw_features']+m['support_features']+m['residual_features']};receipt={'seed':11,'hand_weighting':'ordinary unit weight, STEP277 fit(X,y) parity','fits':{},'input_hashes':{'preflight':sha(O/'PREFLIGHT_OK.json'),'advance':sha(O/'ADVANCE.json'),'manifest':sha(UP/'clean_manifest.json'),'artifact':sha(UP/m['artifact']),'runner':sha(Path(__file__))}}
 for name,cols in arms.items():
  model=LGBMClassifier(objective='binary',n_estimators=400,learning_rate=.05,num_leaves=15,max_depth=4,min_child_samples=30,subsample=.8,subsample_freq=1,colsample_bytree=.5,random_state=11,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
  t=time.monotonic();model.fit(tr.select(cols).to_numpy().astype(np.float32),y);p=model.predict_proba(te.select(cols).to_numpy().astype(np.float32))[:,1];assert p.shape==(940,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all();np.save(O/f'{name}_probabilities.npy',p);joblib.dump(model,O/f'{name}_model.joblib');receipt['fits'][name]={'features':len(cols),'seconds':time.monotonic()-t,'probability_sha256':sha(O/f'{name}_probabilities.npy'),'model_sha256':sha(O/f'{name}_model.joblib')}
 np.savez_compressed(O/'predictions.npz',pair_id=te['pair_id'].to_numpy().astype('U'),hand_id=te['hand_id'].to_numpy().astype('U'),support_control=np.load(O/'support_control_probabilities.npy'),conditional_candidate=np.load(O/'conditional_candidate_probabilities.npy'));receipt['predictions_sha256']=sha(O/'predictions.npz');receipt['quality_reads']=0;receipt['fit_count']=2;(O/'predictions_frozen.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
