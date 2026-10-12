import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
import site,json,hashlib,time
from pathlib import Path
site.addsitedir(r'C:\Data\Projects\Money Making\work\poker\.venv\Lib\site-packages')
import numpy as np
import polars as pl
ROOT=Path(r'C:\Data\Projects\Money Making');OUT=Path(__file__).resolve().parent;SCREEN=OUT.parent/'screen'
LAB=ROOT/'work/poker/data/development_labels.csv';FULL=ROOT/'work/poker/Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet';SALT='foundation-pf35-screen-v1:';BS='pf35-normal-bag-v1:'
FAMS={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def foldof(t):return int(hashlib.sha256((SALT+t).encode()).hexdigest()[:16],16)%2
def hv(s):return hashlib.sha256(s.encode()).hexdigest()
def main():
 assert (OUT/'PROTOCOL.md').exists();t=time.time();x=pl.read_parquet(FULL);lab=pl.read_csv(LAB);d=x.join(lab.select('pair_id','label','behavior_family','player_1','player_2'),on='pair_id',validate='1:1');assert d.height==1860
 y=np.array([FAMS[v] if z==1 else 0 for z,v in zip(d['label'],d['behavior_family'])],dtype=np.int8);ids=np.asarray(d['pair_id'].to_list(),dtype='U');tabs=np.asarray(d['table_id'].to_list(),dtype='U');fold=np.array([foldof(v) for v in tabs]);players=np.asarray(list(zip(d['player_1'],d['player_2'])),dtype='U')
 feat=[c for c,dt in d.schema.items() if c not in {'pair_id','table_id','label','behavior_family','player_1','player_2'} and dt.is_numeric()];assert len(feat)==653
 X=d.select(feat).to_numpy().astype('float32');X[~np.isfinite(X)]=np.nan;fi={v:i for i,v in enumerate(feat)};manifest=[]
 for f in (0,1):
  old=np.load(SCREEN/f'context_fold{f}.npz',allow_pickle=False);names=old['feature_name'];ix=np.array([fi[v] for v in names]);tr=fold!=f;held=~tr;pos=np.flatnonzero(tr&(y>0));norm=np.flatnonzero(tr&(y==0));K=len(pos);assert K in (170,202)
  by={}
  for i in norm:by.setdefault(tabs[i],[]).append(i)
  table_order=sorted(by,key=lambda z:hv(BS+z));
  for tb in by:by[tb].sort(key=lambda i:hv(BS+tabs[i]+':'+ids[i]))
  rr=[];r=0
  while len(rr)<len(norm):
   for tb in table_order:
    if r<len(by[tb]):rr.append(by[tb][r])
   r+=1
  assert set(rr)==set(norm.tolist())
  heldtables=set(tabs[held]);heldplayers=set(players[held].ravel());bagsets=[]
  for b in range(4):
   pick=np.array([rr[(b*K+j)%len(rr)] for j in range(K)]);bagsets.append(set(pick.tolist()));sel=np.sort(np.concatenate([pos,pick]));assert not (set(tabs[sel])&heldtables) and not (set(players[sel].ravel())&heldplayers)
   np.savez_compressed(OUT/f'context_fold{f}_bag{b}.npz',X_train=X[sel][:,ix],y_train=y[sel],X_held=old['X_held'],pair_id=old['pair_id'],feature_name=names)
   pre=np.load(SCREEN/f'context_fold{f}_prefix60.npz',allow_pickle=False);assert np.array_equal(pre['feature_name'],names)
   np.savez_compressed(OUT/f'context_fold{f}_bag{b}_prefix60.npz',X_held=pre['X_held'],pair_id=pre['pair_id'])
   np.savez_compressed(OUT/f'membership_fold{f}_bag{b}.npz',positive_pair_id=ids[pos],normal_pair_id=ids[pick],table_id=tabs[sel])
  pair_overlap=[[len(bagsets[a]&bagsets[b]) for b in range(4)] for a in range(4)]
  manifest.append({'fold':f,'positive_rows':K,'normal_available':len(norm),'rows_per_bag':2*K,'train_tables':len(set(tabs[tr])),'held_tables':len(heldtables),'held_player_overlap':0,'held_table_overlap':0,'normal_pairwise_overlap_matrix':pair_overlap,'normal_total_selections':4*K,'normal_unique_selected':len(set().union(*bagsets)),'feature_names_sha256':hashlib.sha256('\n'.join(names.tolist()).encode()).hexdigest(),'contexts':[{'bag':b,'sha256':sha(OUT/f'context_fold{f}_bag{b}.npz'),'membership_sha256':sha(OUT/f'membership_fold{f}_bag{b}.npz')} for b in range(4)]})
 (OUT/'prep_receipt.json').write_text(json.dumps({'protocol_sha256':sha(OUT/'PROTOCOL.md'),'source_sha256':sha(FULL),'labels_sha256':sha(LAB),'sampling_salt':BS,'ids_predictive_features':False,'feature_selector_refit':False,'folds':manifest,'seconds':time.time()-t},indent=2)+'\n')
if __name__=='__main__':main()
