"""Fixed symmetric action-surprise summaries from frozen neural probabilities."""
import os
os.environ['POLARS_MAX_THREADS']='2'
import sys,json
from pathlib import Path
import numpy as np
import polars as pl
from within_hand_replay_capped import sha,write
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'research/within_hand_sequence_20260912'
ARMS=['query_only','shuffled','ordered'];COLS=[f'wh_summary_{i}' for i in range(26)]
def pair_summary(means,maxima,counts):
 assert means.shape==maxima.shape and means.shape[1:]==(2,8) and counts.shape==means.shape[:2]
 return np.column_stack([(means[:,0]+means[:,1])/2,np.abs(means[:,0]-means[:,1]),np.maximum(maxima[:,0],maxima[:,1]),counts.sum(1)/24,np.abs(counts[:,0]-counts[:,1])/24])
def aggregate(keys,actions,y,p):
 assert p.shape==(len(actions),4) and np.isfinite(p).all() and np.all(p>=0) and np.allclose(p.astype(float).sum(1),1,atol=1e-6,rtol=0)
 mapping={key:i for i,key in enumerate(keys.iter_rows())};ix=np.array([mapping[k] for k in actions.select('pair_id','hand_id').iter_rows()]);direction=actions['direction'].to_numpy()
 residual=np.eye(4)[y]-p;obs=p[np.arange(len(y)),y];v=np.column_stack([residual,-np.log(np.maximum(obs,1e-7)),np.sum(residual**2,axis=1),-np.sum(p*np.log(np.maximum(p,1e-7)),axis=1),obs]);assert v.shape[1]==8
 sums=np.zeros((len(keys),2,8));maxima=np.full_like(sums,-np.inf);counts=np.zeros((len(keys),2),dtype=int)
 np.add.at(sums,(ix,direction),v);np.add.at(counts,(ix,direction),1);np.maximum.at(maxima,(ix,direction),v)
 means=sums/np.maximum(counts[:,:,None],1);maxima[counts==0]=0;out=pair_summary(means,maxima,counts)
 assert np.array_equal(out,pair_summary(means[:,::-1],maxima[:,::-1],counts[:,::-1]))
 assert np.isfinite(out).all() and out.shape==(3240,26) and np.all(counts.sum(1)>0)
 return out

def main():
 assert not (OUT/'features-receipt.json').exists()
 training=json.loads((OUT/'training-receipt.json').read_text());assert training['status']=='complete' and training['models']==15
 keys=pl.read_parquet(OUT/'target-keys.parquet');actions=pl.read_parquet(OUT/'score-keys.parquet');z=np.load(OUT/'score-corpus.npz',allow_pickle=False);y=z['y'];assert len(y)==len(actions)
 sources=[Path(__file__),OUT/'training-receipt.json',OUT/'target-keys.parquet',OUT/'score-keys.parquet',OUT/'score-corpus.npz',HERE/'research/within_hand_sequence_protocol_20260912.md',OUT/'prefit-shuffle-correction.md']
 artifacts={}
 for r in training['receipts']:
  arm,k=r['arm'],r['fold'];p=OUT/f'{arm}-{k}-probabilities.npy';assert sha(p)==r['probabilities_sha256'];sources.append(p)
  out=aggregate(keys,actions,y,np.load(p,allow_pickle=False));q=OUT/f'{arm}-{k}-features.parquet';keys.with_columns(*[pl.Series(c,out[:,i]) for i,c in enumerate(COLS)]).write_parquet(q);artifacts[q.name]=sha(q)
 write(OUT/'features-receipt.json',{'status':'complete','source_hashes':{str(p):sha(p) for p in sources},'artifacts':artifacts,'columns':COLS,'pair_swap_exact':True,'zero_whole_hands':0,'fields':26,'rows':3240,'no_evidence_truth_read':True});print('15 feature tables complete; pair swap exact')
if __name__=='__main__':main()
