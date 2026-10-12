"""Fifteen frozen pool-excluded neural normal-action references, CPU only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time,copy
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from within_hand_model import QueryPrefixModel,configure_deterministic_cpu,train_final,predict_probabilities,load_plain_state_dict
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'research/within_hand_sequence_20260912'
ARMS=['query_only','shuffled','ordered']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def ce(y,p):return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-7)).mean())
def main():
 t=time.perf_counter();assert (OUT/'training-freeze.json').exists() and not (OUT/'training-receipt.json').exists(),'resume numerical check only'
 cf=json.loads((OUT/'corpus-freeze.json').read_text());cr=json.loads((OUT/'corpus-receipt.json').read_text());sr=json.loads((OUT/'shuffle-receipt.json').read_text())
 assert cr['status']==sr['status']=='complete' and cr['replay_violations']==0
 assert sha(OUT/'corpus-freeze.json')==cr['corpus_freeze_sha256']
 for p,d in cf['source_hashes'].items():assert sha(p)==d
 for p,d in cr['artifacts'].items():assert sha(OUT/p)==d
 for p,d in sr['source_hashes'].items():assert sha(p)==d
 for p,d in sr['artifacts'].items():assert sha(OUT/p)==d
 nz=np.load(OUT/'normal-corpus.npz',allow_pickle=False);sz=np.load(OUT/'score-corpus.npz',allow_pickle=False)
 normal={k:nz[k] for k in ['x','query','lengths','y','pool']};score={k:sz[k] for k in ['x','query','lengths','y','pool']};nz.close();sz.close()
 normal['shuffled']=np.load(OUT/'normal-shuffle.npy',allow_pickle=False);score['shuffled']=np.load(OUT/'score-shuffle.npy',allow_pickle=False)
 tf=json.loads((OUT/'training-freeze.json').read_text())
 for path,digest in tf['sources'].items():assert sha(path)==digest
 design=tf['design']
 configure_deterministic_cpu();torch.set_num_interop_threads(2);initial=torch.load(OUT/'initial-model.pt',map_location='cpu',weights_only=True)
 existing=[OUT/'query_only-0.pt',OUT/'query_only-0-normalizer.npz'];assert all(p.exists() for p in existing)
 assert not any((OUT/f'{arm}-{k}.pt').exists() for k in range(5) for arm in ARMS if (arm,k)!=('query_only',0))
 write(OUT/'resume-freeze.json',{'source_sha256':sha(Path(__file__)),'original_training_freeze_sha256':sha(OUT/'training-freeze.json'),'reused_completed_model':{str(p):sha(p) for p in existing},'correction':'Probability-sum check uses float64 accumulation and 1e-6 tolerance; no prediction normalization/change. Original float32 summation reached 2.384e-7 error (two machine eps), while sum of stored probabilities in float64 erred only 1.707e-7. Original threshold2e-7 was unsuitable for float32 accumulated sums. No evidence scores inspected.','models_retrained':0})
 receipts=[]
 for k in range(5):
  tr=np.load(OUT/f'train-{k}-indices.npy');te=np.load(OUT/f'held-{k}-indices.npy')
  for arm in ARMS:
   begin=time.perf_counter();model=QueryPrefixModel();model.load_state_dict(initial)
   use='shuffled' if arm=='shuffled' else 'x'
   if arm=='query_only' and k==0:
    model=load_plain_state_dict(model,OUT/'query_only-0.pt');norm=np.load(OUT/'query_only-0-normalizer.npz',allow_pickle=False);mean,std=norm['mean'],norm['std']
   else:
    mean,std=train_final(model,normal['query'][tr],normal[use][tr],normal['lengths'][tr],normal['y'][tr],arm)
   pp=predict_probabilities(model,score['query'],score[use],score['lengths'],arm,mean,std)
   hp=predict_probabilities(model,normal['query'][te],normal[use][te],normal['lengths'][te],arm,mean,std)
   tp=predict_probabilities(model,normal['query'][tr],normal[use][tr],normal['lengths'][tr],arm,mean,std)
   mp=OUT/f'{arm}-{k}.pt'
   if not mp.exists():torch.save(model.state_dict(),mp);np.savez(OUT/f'{arm}-{k}-normalizer.npz',mean=mean,std=std)
   rr=load_plain_state_dict(QueryPrefixModel(),mp);nn=np.load(OUT/f'{arm}-{k}-normalizer.npz',allow_pickle=False)
   rp=predict_probabilities(rr,score['query'],score[use],score['lengths'],arm,nn['mean'],nn['std']);assert np.array_equal(pp,rp)
   assert pp.shape==(16461,4) and np.isfinite(pp).all() and np.all(pp>=0) and np.allclose(pp.astype(float).sum(1),1,rtol=0,atol=1e-6)
   path=OUT/f'{arm}-{k}-probabilities.npy';np.save(path,pp,allow_pickle=False);np.save(OUT/f'{arm}-{k}-held-probabilities.npy',hp,allow_pickle=False)
   r={'arm':arm,'fold':k,'model_sha256':sha(mp),'probabilities_sha256':sha(path),'normalizer_sha256':sha(OUT/f'{arm}-{k}-normalizer.npz'),'held_probabilities_sha256':sha(OUT/f'{arm}-{k}-held-probabilities.npy'),'reload_exact':True,'train_ce':ce(normal['y'][tr],tp),'held_ce':ce(normal['y'][te],hp),'train_rows':len(tr),'held_rows':len(te),'seconds':time.perf_counter()-begin};receipts.append(r);write(OUT/f'{arm}-{k}-receipt.json',r)
   print(json.dumps(r),flush=True)
 for p,d in json.loads((OUT/'training-freeze.json').read_text())['sources'].items():assert sha(p)==d
 result={'status':'complete','models':15,'receipts':receipts,'training_freeze_sha256':sha(OUT/'training-freeze.json'),'held_ce_by_arm':{a:float(np.mean([r['held_ce'] for r in receipts if r['arm']==a])) for a in ARMS},'seconds':time.perf_counter()-t,'no_evidence_outcome_read':True,'resume_freeze_sha256':sha(OUT/'resume-freeze.json'),'first_completed_model_reused':True}
 write(OUT/'training-receipt.json',result);print('ALL REFERENCES COMPLETE',result['held_ce_by_arm'],result['seconds'],flush=True)
if __name__=='__main__':main()
