"""Frozen normal-play query/prefix corpus and target actions; no evidence truth."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time,math,copy
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl
from within_hand_prefix import extract_prefix,_action_class
from within_hand_replay_capped import replay,sha,write
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
OUT=HERE/'research/within_hand_sequence_20260912';SOURCE=ROOT/'reports/conditional-action-experiment'
AK=['pair_id','hand_id','action_no','direction'];SK=AK+['table_id']
def priority(row,prefix='within-hand-71931|'):
 return hashlib.sha256((prefix+'|'.join(str(row[k]) for k in AK)).encode()).hexdigest()

def main():
 t=time.perf_counter();assert not (OUT/'corpus-freeze.json').exists(),'write-once corpus'
 oldpath=HERE/'research/real_history_encoder_20260912/corpus-freeze.json';old=json.loads(oldpath.read_text())
 protocol=HERE/'research/within_hand_sequence_protocol_20260912.md'
 policy=json.loads((SOURCE/'policy-plan.json').read_text());inputs=policy['inputs'];assert len(inputs)==37
 files=sorted((SOURCE/'action-cache').glob('batch_*.parquet'));assert len(files)==20
 for p in files:assert sha(p)==policy['files'][p.relative_to(SOURCE).as_posix()]
 labs=ROOT/'data/development_labels.csv';assert sha(labs)==policy['files']['development_labels.csv']
 data=pl.read_parquet(files).filter(pl.col('table_id').is_in(old['allowed_pools'])).sort(AK);assert data.select(AK).n_unique()==data.height
 labels=pl.read_csv(labs,columns=['pair_id','player_1','player_2','label','label_status'])
 normal=data.join(labels.select('pair_id','label','label_status'),on='pair_id',validate='m:1').filter(pl.col('label')==0)
 assert set(normal['label_status'])=={'confirmed_non_target'}
 normal=normal.with_columns(pl.Series('priority',[priority(row) for row in normal.select(AK).iter_rows(named=True)])).sort('pair_id','priority').group_by('pair_id',maintain_order=True).head(128).drop('label','label_status').sort('priority')
 keypath=HERE/'research/real_history_encoder_20260912/target-keys.parquet';keys=pl.read_parquet(keypath).sort('pair_id','hand_id');assert keys.height==3240
 score=data.join(keys,on=['pair_id','hand_id'],how='semi').sort(AK);assert score.height==16461 and score.select('pair_id','hand_id').unique().height==3240
 assert not set(normal['table_id'])&set(old['outer_pools']) and not set(score['table_id'])&set(old['outer_pools'])
 normal.select(SK+['priority']).write_parquet(OUT/'normal-keys.parquet');score.select(SK).write_parquet(OUT/'score-keys.parquet');keys.write_parquet(OUT/'target-keys.parquet')
 sources=[Path(__file__),HERE/'src/within_hand_prefix.py',HERE/'src/within_hand_replay_capped.py',protocol,oldpath,keypath,SOURCE/'policy-plan.json',OUT/'normal-keys.parquet',OUT/'score-keys.parquet',OUT/'target-keys.parquet',OUT/'capped-replay/replay-receipt.json',labs,*files,*[ROOT/'data'/f for f in ['hands.parquet','seats.parquet','actions.parquet']]]
 write(OUT/'corpus-freeze.json',{'source_hashes':{str(p):sha(p) for p in sources},'inputs':inputs,'normal_rows':normal.height,'score_rows':score.height,'allowed_pools':old['allowed_pools'],'outer_pools':old['outer_pools'],'encoder_exclusions':old['encoder_exclusions'],'detector_design':old['detector_design'],'priority_formula':'SHA256(within-hand-71931|pair_id|hand_id|action_no|direction)','rows_per_pair':128,'training_cap':50000,'held_cap':10000,'no_outer_evidence_read':True})
 allrows=pl.concat([normal.select(SK+inputs+['action_class']),score.select(SK+inputs+['action_class'])]).with_row_index('_row');need=allrows.partition_by('hand_id',as_dict=True)
 handkeys=allrows.select('hand_id').unique()
 hands=pl.scan_parquet(ROOT/'data/hands.parquet').select('hand_id','table_id','button_seat','small_blind','big_blind','board_cards','final_pot').join(handkeys.lazy(),on='hand_id',how='semi').collect().sort('hand_id')
 seats=pl.scan_parquet(ROOT/'data/seats.parquet').select('hand_id','player_id','seat_no','starting_stack','total_contribution','folded').join(handkeys.lazy(),on='hand_id',how='semi').collect().partition_by('hand_id',as_dict=True)
 actions=pl.scan_parquet(ROOT/'data/actions.parquet').join(handkeys.lazy(),on='hand_id',how='semi').collect().partition_by('hand_id',as_dict=True)
 pairs={p:(a,b) for p,a,b in labels.select('pair_id','player_1','player_2').iter_rows()}
 x=np.zeros((allrows.height,24,18),dtype=np.float32);n=np.zeros(allrows.height,dtype=np.int64);seen=np.zeros(allrows.height,dtype=np.int8);checks=Counter();fixtures=0
 for hand in hands.iter_rows(named=True):
  hid=hand['hand_id'];ss=seats[(hid,)].to_dicts();aa=sorted(actions[(hid,)].to_dicts(),key=lambda a:a['action_no']);em={a['action_no']:a for a in aa}
  cc,bad=replay(hand,ss,aa);assert not bad,(hid,bad);checks.update(cc)
  for row in need[(hid,)].iter_rows(named=True):
   i=row['_row'];pair=pairs[row['pair_id']];actor=pair[row['direction']];partner=pair[1-row['direction']];a=em[row['action_no']]
   assert actor==a['player_id'] and row['table_id']==hand['table_id']
   assert row['action_class']==_action_class(a,a['amount'],a['to_call'])[0]
   raw={'street_index':{'preflop':0,'flop':1,'turn':2,'river':3}[a['street']],'price':a['to_call']/max(1,a['pot_before']+a['to_call']),'log_pot_bb':math.log1p(a['pot_before']/hand['big_blind']),'log_stack_bb':math.log1p(a['stack_before']/hand['big_blind']),'players_active':a['players_active'],'prior_actions':sum(e['action_no']<a['action_no'] for e in aa)}
   for name,v in raw.items():assert np.float32(row[name])==np.float32(v),(name,row[name],v)
   z,l=extract_prefix(hand,ss,aa,actor,partner,a['action_no']);x[i]=z;n[i]=l;seen[i]+=1
   if fixtures<20:
    changed=copy.deepcopy(aa)
    for event in changed:
     if event['action_no']>=a['action_no']:event.update(action='unread_future',amount=-777,stack_before=-111)
    hc=dict(hand);past=[e for e in aa if e['action_no']<a['action_no']];nb=max([{'preflop':0,'flop':3,'turn':4,'river':5}[e['street']] for e in past],default=0)
    hc['board_cards']=' '.join((hand['board_cards'] or '').split()[:nb]+['ZZ']*(5-nb))
    mutated,ml=extract_prefix(hc,ss,changed,actor,partner,a['action_no']);assert ml==l and np.array_equal(mutated,z)
    fixtures+=1
 assert np.all(seen==1) and fixtures==20
 shuffled=x.copy()
 for row,l in zip(allrows.select(AK).iter_rows(named=True),n):
  pass
 for i,row in enumerate(allrows.select(AK).iter_rows(named=True)):
  l=int(n[i]);seed=int(priority(row,'within-hand-shuffle-71931|')[:16],16);perm=np.random.default_rng(seed).permutation(l);shuffled[i,:l]=x[i,:l][perm]
  assert np.array_equal(np.sort(shuffled[i,:l],axis=0),np.sort(x[i,:l],axis=0))
 query=allrows.select(inputs).to_numpy().astype(np.float32);y=allrows['action_class'].to_numpy().astype(np.int64);assert np.isfinite(query).all()
 for name,lo,hi in [('normal',0,normal.height),('score',normal.height,allrows.height)]:
  np.savez_compressed(OUT/f'{name}-corpus.npz',x=x[lo:hi],shuffled=shuffled[lo:hi],lengths=n[lo:hi],query=query[lo:hi],y=y[lo:hi],pool=np.asarray(allrows['table_id'].to_list()[lo:hi],dtype=str))
 filesout=['normal-corpus.npz','score-corpus.npz','normal-keys.parquet','score-keys.parquet','target-keys.parquet']
 result={'status':'complete','normal_rows':normal.height,'score_rows':score.height,'normal_pairs':normal['pair_id'].n_unique(),'normal_pools':normal['table_id'].n_unique(),'source_hands':hands.height,'source_actions':checks['pot_before'],'replay_checks':dict(checks),'replay_violations':0,'actual_future_fixtures':fixtures,'empty_prefixes':int(sum(n==0)),'max_prefix':int(n.max()),'different_shuffled_rows':int(np.any(x!=shuffled,axis=(1,2)).sum()),'corpus_freeze_sha256':sha(OUT/'corpus-freeze.json'),'artifacts':{p:sha(OUT/p) for p in filesout},'seconds':time.perf_counter()-t,'no_outer_evidence_read':True}
 for p,digest in json.loads((OUT/'corpus-freeze.json').read_text())['source_hashes'].items():assert sha(p)==digest
 write(OUT/'corpus-receipt.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
