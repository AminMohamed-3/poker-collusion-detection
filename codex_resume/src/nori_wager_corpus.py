"""Build fixed fold-excluded continuous wager contexts; no evidence outcomes."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl
from within_hand_replay_capped import sha,write
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
OUT=HERE/'research/nori_wager_20260912';OLD=HERE/'research/within_hand_sequence_20260912';SOURCE=ROOT/'reports/conditional-action-experiment'
AK=['pair_id','hand_id','action_no','direction']
def main():
 t=time.perf_counter();OUT.mkdir(exist_ok=True);assert not (OUT/'corpus-freeze.json').exists()
 cf=json.loads((OLD/'corpus-freeze.json').read_text());cr=json.loads((OLD/'corpus-receipt.json').read_text());assert cr['status']=='complete' and sha(OLD/'corpus-freeze.json')==cr['corpus_freeze_sha256']
 inputs=cf['inputs'];assert len(inputs)==37
 policy=json.loads((SOURCE/'policy-plan.json').read_text());files=sorted((SOURCE/'action-cache').glob('batch_*.parquet'))
 for p in files:assert sha(p)==policy['files'][p.relative_to(SOURCE).as_posix()]
 for name in ['normal-keys.parquet','score-keys.parquet','target-keys.parquet']:assert sha(OLD/name)==cr['artifacts'][name]
 use=AK+['table_id','action_class','observed_log_amount_bb']+inputs
 data=pl.read_parquet(files,columns=use).filter(pl.col('table_id').is_in(cf['allowed_pools']))
 frames={};artifacts={}
 for name in ['normal','score']:
  keys=pl.read_parquet(OLD/f'{name}-keys.parquet').select(AK).sort(AK)
  f=keys.join(data,on=AK,how='left',validate='1:1').sort(AK);assert f.height==keys.height and f['action_class'].null_count()==0
  f=f.filter(pl.col('action_class')>=2)
  den=np.maximum(1,np.expm1(f['log_pot_bb'].to_numpy().astype(float)))
  y=np.log1p(np.expm1(f['observed_log_amount_bb'].to_numpy().astype(float))/den);upper=np.log1p(np.expm1(f['log_stack_bb'].to_numpy().astype(float))/den)
  x=np.column_stack([f.select(inputs).to_numpy(),(f['action_class']==3).to_numpy()]).astype(np.float32)
  assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(upper).all() and np.all(y>=0) and np.all(y<=upper+1e-5)
  assert not set(f['table_id'])&set(cf['outer_pools'])
  np.savez(OUT/f'{name}-corpus.npz',x=x,y=y,upper=upper)
  f.select(AK+['table_id']).write_parquet(OUT/f'{name}-keys.parquet');frames[name]=f
  for ext in ['corpus.npz','keys.parquet']:artifacts[f'{name}-{ext}']=sha(OUT/f'{name}-{ext}')
 target=pl.read_parquet(OLD/'target-keys.parquet').sort('pair_id','hand_id');assert target.height==3240;target.write_parquet(OUT/'target-keys.parquet');artifacts['target-keys.parquet']=sha(OUT/'target-keys.parquet')
 normal=frames['normal'];priority=np.array([hashlib.sha256(('nori-wager-72311|'+'|'.join(map(str,r))).encode()).hexdigest() for r in normal.select(AK).iter_rows()]);order=np.argsort(priority,kind='stable');pool=normal['table_id'].to_numpy();design=[]
 for k in range(5):
  excluded=set(cf['encoder_exclusions'][str(k)]);mask=np.isin(pool,list(excluded));tr=order[~mask[order]][:2048];te=order[mask[order]][:1024]
  assert len(tr)==2048 and len(te)==1024 and not set(tr)&set(te)
  assert not set(pool[tr])&excluded and not set(pool[tr])&set(cf['outer_pools'])
  np.save(OUT/f'train-{k}-indices.npy',tr,allow_pickle=False);np.save(OUT/f'held-{k}-indices.npy',te,allow_pickle=False)
  for tag in ['train','held']:artifacts[f'{tag}-{k}-indices.npy']=sha(OUT/f'{tag}-{k}-indices.npy')
  design.append({'fold':k,'train_pools':sorted(set(pool[tr])),'held_pools':sorted(set(pool[te])),'train_rows':len(tr),'held_rows':len(te),'train_indices_sha256':sha(OUT/f'train-{k}-indices.npy'),'held_indices_sha256':sha(OUT/f'held-{k}-indices.npy'),'train_pairs':normal[tr]['pair_id'].n_unique(),'held_pairs':normal[te]['pair_id'].n_unique()})
 sources=[Path(__file__),HERE/'research/nori_wager_protocol_20260912.md',OLD/'corpus-freeze.json',OLD/'corpus-receipt.json',SOURCE/'policy-plan.json',*[OLD/f'{name}-keys.parquet' for name in ['normal','score','target']],*files]
 freeze={'status':'complete','source_hashes':{str(p):sha(p) for p in sources},'artifacts':artifacts,'inputs':inputs+['is_raise'],'design':design,'encoder_exclusions':cf['encoder_exclusions'],'outer_pools':cf['outer_pools'],'allowed_pools':cf['allowed_pools'],'detector_design':cf['detector_design'],'normal_rows':normal.height,'score_rows':frames['score'].height,'target_hands':3240,'zero_eligible_hands':3240-frames['score'].select('pair_id','hand_id').n_unique(),'no_evidence_outcomes_read':True}
 write(OUT/'corpus-freeze.json',freeze);print(json.dumps({k:freeze[k] for k in ['normal_rows','score_rows','zero_eligible_hands']})+' seconds='+str(time.perf_counter()-t),flush=True)
if __name__=='__main__':main()
