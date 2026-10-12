"""Label-free deterministic public-state replay, outside the outer-pool union."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
import polars as pl
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
OUT=HERE/'research/within_hand_sequence_20260912'
STREETS={'preflop':0,'flop':1,'turn':2,'river':3}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def replay(hand,seats,events):
 checks=Counter();bad=[]
 def check(name,actual,expected,action_no=-1):
  checks[name]+=1
  if actual!=expected:bad.append({'check':name,'action_no':action_no,'actual':actual,'expected':expected})
 sm={s['player_id']:s for s in seats};check('six_unique_seats',sorted(s['seat_no'] for s in seats),list(range(6)))
 start={p:s['starting_stack'] for p,s in sm.items()};total={p:0 for p in sm};street={p:0 for p in sm}
 for seat,blind in [((hand['button_seat']+1)%6,hand['small_blind']),((hand['button_seat']+2)%6,hand['big_blind'])]:
  p=next(p for p,s in sm.items() if s['seat_no']==seat);total[p]=blind;street[p]=blind;check('blind_stack',start[p]>=blind,True)
 folded=set();allin={p for p in sm if start[p]==total[p]};prior=-1;previous_street=0
 check('unique_action_no',len(set(a['action_no'] for a in events)),len(events))
 for a in sorted(events,key=lambda a:a['action_no']):
  n=a['action_no'];p=a['player_id'];s=STREETS[a['street']]
  check('action_order',n>prior,True,n);check('street_order',s>=previous_street,True,n)
  if s!=previous_street:street={p:0 for p in sm}
  check('actor_seated',p in sm,True,n)
  if p not in sm:continue
  remaining=start[p]-total[p];tocall=max(street.values())-street[p]
  check('actor_not_folded',p not in folded,True,n);check('actor_not_allin',p not in allin and remaining>0,True,n)
  check('pot_before',a['pot_before'],sum(total.values()),n);check('stack_before',a['stack_before'],remaining,n)
  check('to_call',a['to_call'],tocall,n);check('amount_to',a['amount_to'],street[p]+a['amount'],n)
  check('players_active',a['players_active'],len(sm)-len(folded),n)
  check('amount_range',0<=a['amount']<=remaining,True,n)
  if a['action'] in ('fold','check'):check('zero_action_amount',a['amount'],0,n)
  if a['action']=='check':check('check_price',tocall,0,n)
  if a['action']=='call':check('call_amount',a['amount'],min(tocall,remaining),n)
  if a['action'] in ('bet','raise'):check('aggressive_amount',a['amount']>tocall,True,n)
  if a['action']=='bet':check('bet_price',tocall,0,n)
  if a['action']=='all_in':check('allin_amount',a['amount'],remaining,n)
  total[p]+=a['amount'];street[p]+=a['amount']
  if a['action']=='fold':folded.add(p)
  if total[p]==start[p]:allin.add(p)
  prior=n;previous_street=s
 check('final_pot',sum(total.values()),hand['final_pot'])
 for p,s in sm.items():
  check('total_contribution',total[p],s['total_contribution']);check('terminal_fold',p in folded,s['folded'])
 return checks,bad

def main():
 t=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'replay-freeze.json').exists()
 base=HERE/'research/real_history_encoder_20260912/corpus-freeze.json';allowed=json.loads(base.read_text())['allowed_pools']
 h=pl.scan_parquet(ROOT/'data/hands.parquet').filter((pl.col('phase')=='development')&pl.col('table_id').is_in(allowed)).select('hand_id','table_id','button_seat','small_blind','big_blind','final_pot').sort('table_id','hand_id').group_by('table_id',maintain_order=True).head(100).collect().sort('hand_id')
 h.select('hand_id','table_id').write_parquet(OUT/'replay-keys.parquet')
 sources=[Path(__file__),base,OUT/'replay-keys.parquet',*[ROOT/'data'/n for n in ('hands.parquet','seats.parquet','actions.parquet')]]
 write(OUT/'replay-freeze.json',{'sample':'first100 hand_ids per permitted development pool, no gameplay or label conditioning','source_hashes':{str(p):sha(p) for p in sources},'outer_union_excluded':True,'labels_read':False,'hands':h.height,'pools':h['table_id'].n_unique(),'convention':'amount incremental; amount_to cumulative on current street; to_call uncapped max(all street contributions)-own; active includes not-folded all-in players'})
 sm=pl.scan_parquet(ROOT/'data/seats.parquet').select('hand_id','player_id','seat_no','starting_stack','total_contribution','folded').join(h.lazy().select('hand_id'),on='hand_id',how='semi').collect().partition_by('hand_id',as_dict=True)
 am=pl.scan_parquet(ROOT/'data/actions.parquet').join(h.lazy().select('hand_id'),on='hand_id',how='semi').collect().partition_by('hand_id',as_dict=True)
 counts=Counter();bad=[];allinhands=0;maxactions=0;longhands=0
 for hand in h.iter_rows(named=True):
  hid=hand['hand_id'];events=am[(hid,)].to_dicts();c,b=replay(hand,sm[(hid,)].to_dicts(),events);counts.update(c)
  allinhands+=any(a['action']=='all_in' for a in events);maxactions=max(maxactions,len(events));longhands+=len(events)>24
  if b:bad.append({'hand_id':hid,'violations':b})
 result={'status':'complete','hands':h.height,'pools':h['table_id'].n_unique(),'actions':counts['pot_before'],'checks':dict(counts),'hands_with_violation':len(bad),'violation_counts':dict(Counter(v['check'] for b in bad for v in b['violations'])),'first_violations':bad[:10],'allin_hands':allinhands,'maximum_actions':maxactions,'hands_above24_actions':longhands,'seconds':time.perf_counter()-t,'no_evidence_truth_read':True,'freeze_sha256':sha(OUT/'replay-freeze.json')}
 write(OUT/'replay-receipt.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
