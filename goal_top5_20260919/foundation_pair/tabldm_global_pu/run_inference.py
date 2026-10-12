import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; GP=HERE.parent/'global_pu'; RT=ROOT/'goal_top5_20260919/tabldm'; CK=RT/'checkpoint/clf_default.ckpt'
PROGRESS=HERE/'chunk_progress.jsonl'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((HERE/'MANIFEST.json').read_text()); release=json.loads((HERE/'GPU_RELEASE.json').read_text()) if (HERE/'GPU_RELEASE.json').exists() else {}
assert release.get('manifest_sha256')==sha(HERE/'MANIFEST.json'),'hash-bound root GPU release absent'
for rel,h in manifest['inputs'].items():assert sha(ROOT/rel)==h,(rel,'input drift')
for name,h in manifest['code_files'].items():assert sha(HERE/name)==h,(name,'code drift')
assert Path(sys.executable).resolve()==Path(manifest['python_executable']).resolve();sys.path.insert(0,str(RT/'overlay'))
import numpy as np,torch,tabldm
from tabldm import TabLDMClassifier
assert RT/'overlay' in Path(tabldm.__file__).resolve().parents
SEED=35935;CHUNK=256;CAP=1500
def model():return TabLDMClassifier(n_estimators=4,batch_size=1,random_state=SEED,n_jobs=2,enhance_candidates=False,kv_cache=False,model_path=CK,allow_auto_download=False)
def load_query(f):
 parts=[]
 for p in sorted((GP/'contexts').glob(f'query_fold{f}_*.npz')):parts.append(np.load(p,allow_pickle=False))
 X=np.concatenate([z['X_held'] for z in parts]);keys=np.concatenate([z['pair_id'] for z in parts]);tables=np.concatenate([z['table_id'] for z in parts]);assert np.array_equal(keys,np.sort(keys));return X,keys,tables
def predict(m,X,keys,deadline,surface,limit=None):
 n=len(X) if limit is None else min(limit,len(X));out=[];chunks=[]
 for i in range(0,n,CHUNK):
  assert time.monotonic()<deadline,'25-minute deadline exceeded before query chunk';j=min(i+CHUNK,n);q=X[i:j].copy();mask=np.flatnonzero(np.all(np.isnan(q),axis=0)).tolist();p=m.predict_proba(q);assert p.shape==(j-i,4) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=2e-6);out.append(p);rec={'surface':surface,'start':i,'stop':j,'first_pair_id':str(keys[i]),'last_pair_id':str(keys[j-1]),'all_nan_feature_indices':mask};chunks.append(rec)
  with PROGRESS.open('a',encoding='utf-8') as log:log.write(json.dumps(rec,separators=(',',':'))+'\n');log.flush()
 return np.concatenate(out),chunks
def fit_fold(f,deadline,surface,limit=None):
 tr=np.load(GP/f'contexts/train_fold{f}.npz',allow_pickle=False);X,k,t=load_query(f);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();s=time.monotonic();m=model();m.fit(tr['X_train'].copy(),tr['y_train'].copy());assert np.array_equal(m.classes_,np.arange(4));p,ch=predict(m,X,k,deadline,surface,limit);peak=int(torch.cuda.max_memory_reserved());assert peak<3*1024**3;return p,k[:len(p)],t[:len(p)],ch,time.monotonic()-s,peak
def main():
 a=argparse.ArgumentParser();a.add_argument('--replay',action='store_true');z=a.parse_args()
 if z.replay:
  assert (HERE/'predictions.npz').exists() and not (HERE/'replay.npz').exists();p,k,t,ch,sec,peak=fit_fold(1,time.monotonic()+CAP,'replay_fold1_first1024',1024);old=np.load(HERE/'predictions.npz',allow_pickle=False);assert np.array_equal(k,old['pair_fold1'][:1024]) and np.array_equal(t,old['table_fold1'][:1024]) and np.array_equal(p,old['prob_fold1'][:1024]);np.savez_compressed(HERE/'replay.npz',prob=p,pair_id=k,table_id=t);(HERE/'replay_receipt.json').write_text(json.dumps({'exact':True,'array_key_match_first1024':True,'seconds':sec,'peak_reserved':peak,'chunks':ch,'replay_sha256':sha(HERE/'replay.npz'),'predictions_sha256':sha(HERE/'predictions.npz'),'progress_log_sha256':sha(PROGRESS)},indent=2)+'\n');return
 assert not (HERE/'predictions.npz').exists() and not (HERE/'runtime.json').exists() and not PROGRESS.exists();start=time.monotonic();deadline=start+CAP;arr={};fits=[]
 for f in (1,2):
  p,k,t,ch,sec,peak=fit_fold(f,deadline,f'score_fold{f}');arr[f'prob_fold{f}']=p;arr[f'pair_fold{f}']=k;arr[f'table_fold{f}']=t;fits.append({'fold':f,'seconds':sec,'peak_reserved':peak,'chunks':ch});gc.collect();torch.cuda.empty_cache();assert time.monotonic()<deadline
 np.savez_compressed(HERE/'predictions.npz',**arr);(HERE/'runtime.json').write_text(json.dumps({'scope':'inference only; no quality','predictions_sha256':sha(HERE/'predictions.npz'),'checkpoint_sha256':sha(CK),'query_chunk':CHUNK,'seed':SEED,'n_estimators':4,'batch_size':1,'enhance_candidates':False,'fits':fits,'total_seconds':time.monotonic()-start,'progress_log_sha256':sha(PROGRESS)},indent=2)+'\n')
if __name__=='__main__':main()
