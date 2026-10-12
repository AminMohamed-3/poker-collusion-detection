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
 t=time.perf_counter();assert not (OUT/'training-freeze.json').exists(),'write-once study'
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
 design=[]
 for k in range(5):
  exclusion=cf['encoder_exclusions'][str(k)];tr=np.flatnonzero(~np.isin(normal['pool'],exclusion))[:50000];te=np.flatnonzero(np.isin(normal['pool'],exclusion))[:10000]
  tp=set(normal['pool'][tr]);hp=set(normal['pool'][te]);assert not tp&hp and not tp&set(exclusion) and not tp&set(cf['outer_pools'])
  assert len(tr)==50000 and len(te)>0 and set(normal['y'][tr])=={0,1,2,3}
  np.save(OUT/f'train-{k}-indices.npy',tr,allow_pickle=False);np.save(OUT/f'held-{k}-indices.npy',te,allow_pickle=False)
  design.append({'fold':k,'train_rows':len(tr),'held_rows':len(te),'training_pools':sorted(tp),'held_pools':sorted(hp),'excluded_pools':exclusion,'train_indices_sha256':sha(OUT/f'train-{k}-indices.npy'),'held_indices_sha256':sha(OUT/f'held-{k}-indices.npy')})
 configure_deterministic_cpu();torch.set_num_interop_threads(2);initial=QueryPrefixModel().state_dict();torch.save(initial,OUT/'initial-model.pt')
 sources=[Path(__file__),HERE/'src/within_hand_model.py',HERE/'research/within_hand_sequence_protocol_20260912.md',OUT/'prefit-shuffle-correction.md',OUT/'corpus-freeze.json',OUT/'corpus-receipt.json',OUT/'shuffle-receipt.json',OUT/'initial-model.pt',*[OUT/f'{n}-{s}.npz' for n in ['normal','score'] for s in ['corpus']],OUT/'normal-shuffle.npy',OUT/'score-shuffle.npy']
 write(OUT/'training-freeze.json',{'sources':{str(p):sha(p) for p in sources},'design':design,'arms':ARMS,'fits':15,'seed':71931,'epochs':6,'batch_size':512,'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.0001,'clip':5},'same_initial_state':True,'query_normalizer_training_only':True,'scoring_model_is_fold_specific':True,'final_epoch_only':True,'shuffle_correction_applied':True})
 receipts=[]
 for k in range(5):
  tr=np.load(OUT/f'train-{k}-indices.npy');te=np.load(OUT/f'held-{k}-indices.npy')
  for arm in ARMS:
   begin=time.perf_counter();model=QueryPrefixModel();model.load_state_dict(initial)
   use='shuffled' if arm=='shuffled' else 'x'
   mean,std=train_final(model,normal['query'][tr],normal[use][tr],normal['lengths'][tr],normal['y'][tr],arm)
   pp=predict_probabilities(model,score['query'],score[use],score['lengths'],arm,mean,std)
   hp=predict_probabilities(model,normal['query'][te],normal[use][te],normal['lengths'][te],arm,mean,std)
   tp=predict_probabilities(model,normal['query'][tr],normal[use][tr],normal['lengths'][tr],arm,mean,std)
   mp=OUT/f'{arm}-{k}.pt';torch.save(model.state_dict(),mp);np.savez(OUT/f'{arm}-{k}-normalizer.npz',mean=mean,std=std)
   rr=load_plain_state_dict(QueryPrefixModel(),mp);nn=np.load(OUT/f'{arm}-{k}-normalizer.npz',allow_pickle=False)
   rp=predict_probabilities(rr,score['query'],score[use],score['lengths'],arm,nn['mean'],nn['std']);assert np.array_equal(pp,rp)
   assert pp.shape==(16461,4) and np.isfinite(pp).all() and np.all(pp>=0) and np.allclose(pp.sum(1),1,rtol=0,atol=2e-7)
   path=OUT/f'{arm}-{k}-probabilities.npy';np.save(path,pp,allow_pickle=False);np.save(OUT/f'{arm}-{k}-held-probabilities.npy',hp,allow_pickle=False)
   r={'arm':arm,'fold':k,'model_sha256':sha(mp),'probabilities_sha256':sha(path),'normalizer_sha256':sha(OUT/f'{arm}-{k}-normalizer.npz'),'held_probabilities_sha256':sha(OUT/f'{arm}-{k}-held-probabilities.npy'),'reload_exact':True,'train_ce':ce(normal['y'][tr],tp),'held_ce':ce(normal['y'][te],hp),'train_rows':len(tr),'held_rows':len(te),'seconds':time.perf_counter()-begin};receipts.append(r);write(OUT/f'{arm}-{k}-receipt.json',r)
   print(json.dumps(r),flush=True)
 for p,d in json.loads((OUT/'training-freeze.json').read_text())['sources'].items():assert sha(p)==d
 result={'status':'complete','models':15,'receipts':receipts,'training_freeze_sha256':sha(OUT/'training-freeze.json'),'held_ce_by_arm':{a:float(np.mean([r['held_ce'] for r in receipts if r['arm']==a])) for a in ARMS},'seconds':time.perf_counter()-t,'no_evidence_outcome_read':True}
 write(OUT/'training-receipt.json',result);print('ALL REFERENCES COMPLETE',result['held_ce_by_arm'],result['seconds'],flush=True)
if __name__=='__main__':main()
