import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
H=Path(__file__).resolve().parent;P=H.parents[2];RT=P/'goal_top5_20260919/tabldm';CTX=H.parent/'full_fast/full_context.npz';CK=RT/'checkpoint/clf_default.ckpt';PROG=H/'chunk_progress.jsonl';BATCH=4096;CHUNK=256;SEED=35935;CAP=2100
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
man=json.loads((H/'MANIFEST.json').read_text());rel=json.loads((H/'GPU_RELEASE.json').read_text()) if (H/'GPU_RELEASE.json').exists() else {};assert rel.get('manifest_sha256')==sha(H/'MANIFEST.json'),'root GPU release absent'
for n,v in man['inputs'].items():assert sha(P/'goal_top5_20260919'/n)==v,(n,'drift')
for n,v in man['code_files'].items():assert sha(H/n)==v,(n,'code drift')
assert Path(sys.executable).resolve()==Path(man['python_executable']).resolve();sys.path.insert(0,str(P/'.venv/Lib/site-packages'));sys.path.insert(0,str(RT/'overlay'))
import numpy as np,polars as pl,torch,tabldm
from tabldm import TabLDMClassifier
assert RT/'overlay' in Path(tabldm.__file__).resolve().parents;torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70)
def model():return TabLDMClassifier(n_estimators=4,batch_size=1,random_state=SEED,n_jobs=2,enhance_candidates=False,kv_cache=False,model_path=CK,allow_auto_download=False)
def predict(m,X,keys,start,deadline,surface):
 out=[]
 for i in range(0,len(X),CHUNK):
  assert time.monotonic()<deadline;j=min(i+CHUNK,len(X));q=X[i:j].copy();mask=np.flatnonzero(np.all(np.isnan(q),axis=0)).tolist();p=m.predict_proba(q);assert p.shape==(j-i,4) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=2e-6);out.append(p);rec={'surface':surface,'start':start+i,'stop':start+j,'first_pair_id':str(keys[i]),'last_pair_id':str(keys[j-1]),'all_nan_feature_indices':mask};
  with PROG.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,separators=(',',':'))+'\n');f.flush()
  assert torch.cuda.max_memory_reserved()<3*1024**3,'3 GiB GPU cap exceeded'
 return np.concatenate(out)
def fitted(d):
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m=model();m.fit(d['X_train'].copy(),d['y_train'].copy());assert np.array_equal(m.classes_,np.arange(4));return m
def main():
 a=argparse.ArgumentParser();a.add_argument('--replay',action='store_true');z=a.parse_args();d=np.load(CTX,allow_pickle=False);keys=d['pair_id'];assert len(keys)==112540 and len(np.unique(keys))==len(keys) and np.array_equal(keys,np.sort(keys));deadline=time.monotonic()+CAP;s=time.monotonic();m=fitted(d)
 if z.replay:
  assert (H/'tabldm_probabilities.parquet').exists() and not (H/'replay_first1024.npz').exists() and not (H/'replay_receipt.json').exists();p=predict(m,d['X_eval'][:1024],keys[:1024],0,deadline,'replay_first1024');old=pl.read_parquet(H/'tabldm_probabilities.parquet').select('p_none','p_dt','p_sp','p_ci').head(1024).to_numpy();assert np.array_equal(p,old);np.savez_compressed(H/'replay_first1024.npz',p=p,pair_id=keys[:1024]);(H/'replay_receipt.json').write_text(json.dumps({'exact':True,'seconds':time.monotonic()-s,'peak_reserved':int(torch.cuda.max_memory_reserved()),'fit_count_this_process':1,'replay_sha256':sha(H/'replay_first1024.npz'),'probabilities_sha256':sha(H/'tabldm_probabilities.parquet'),'progress_sha256':sha(PROG)},indent=2)+'\n');return
 assert not (H/'tabldm_probabilities.parquet').exists() and not (H/'runtime.json').exists() and not PROG.exists() and not list(H.glob('batch_*.parquet')),'normal inference refuses partial or completed outputs'
 for start in range(0,len(keys),BATCH):
  dest=H/f'batch_{start//BATCH:03}.parquet'
  assert not dest.exists()
  stop=min(start+BATCH,len(keys));p=predict(m,d['X_eval'][start:stop],keys[start:stop],start,deadline,f'score_batch_{start//BATCH:03}');pl.DataFrame({'pair_id':keys[start:stop],'p_none':p[:,0],'p_dt':p[:,1],'p_sp':p[:,2],'p_ci':p[:,3]}).write_parquet(dest)
 parts=[pl.read_parquet(x) for x in sorted(H.glob('batch_*.parquet'))];allp=pl.concat(parts);assert len(parts)==28 and allp.height==len(keys) and np.array_equal(allp['pair_id'].to_numpy(),keys) and torch.cuda.max_memory_reserved()<3*1024**3;allp.write_parquet(H/'tabldm_probabilities.parquet');(H/'runtime.json').write_text(json.dumps({'scope':'inference only','seconds':time.monotonic()-s,'peak_reserved':int(torch.cuda.max_memory_reserved()),'fit_count_this_process':1,'batches':len(parts),'probabilities_sha256':sha(H/'tabldm_probabilities.parquet'),'progress_sha256':sha(PROG),'query_chunk':CHUNK,'batch_rows':BATCH,'resume_supported':False},indent=2)+'\n')
if __name__=='__main__':main()
