import os
os.environ['POLARS_MAX_THREADS']='2'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import polars as pl
from within_hand_replay_capped import sha,write
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'research/within_hand_sequence_20260912'
def main():
 assert not (OUT/'shuffle-receipt.json').exists()
 sources=[Path(__file__),OUT/'prefit-shuffle-correction.md',OUT/'corpus-receipt.json']
 artifacts={};counts={}
 for name in ['normal','score']:
  p=OUT/f'{name}-corpus.npz';k=OUT/f'{name}-keys.parquet';sources.extend([p,k]);z=np.load(p,allow_pickle=False);x=z['x'];n=z['lengths'];keys=pl.read_parquet(k);assert len(keys)==len(x)
  xx=x.copy()
  for i,row in enumerate(keys.select('pair_id','hand_id','action_no').iter_rows()):
   text='within-hand-shuffle-71931|'+'|'.join(map(str,row));seed=int(hashlib.sha256(text.encode()).hexdigest()[:16],16);l=int(n[i]);perm=np.random.default_rng(seed).permutation(l);xx[i,:l]=x[i,:l][perm]
   assert np.array_equal(np.sort(xx[i,:l],axis=0),np.sort(x[i,:l],axis=0)) and np.count_nonzero(xx[i,l:])==0
  q=OUT/f'{name}-shuffle.npy';np.save(q,xx,allow_pickle=False);artifacts[q.name]=sha(q);counts[name]=int(np.any(xx!=x,axis=(1,2)).sum())
 write(OUT/'shuffle-receipt.json',{'status':'complete','before_any_model_fit':True,'source_hashes':{str(p):sha(p) for p in sources},'artifacts':artifacts,'different_rows':counts,'shuffle_key_omits_direction':True})
 print(json.dumps({'status':'complete','different_rows':counts}))
if __name__=='__main__':main()
