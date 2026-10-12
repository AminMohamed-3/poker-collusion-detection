from pathlib import Path
from collections import defaultdict
import os,json,time,joblib,numpy as np,polars as pl
os.environ.setdefault('POLARS_MAX_THREADS','2'); os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
import sys; sys.path.insert(0,str(ROOT/'src'))
from conditional_action_features import extract_hand,action_class,INPUTS,card_features

def main():
 t=time.time(); support=json.loads((OUT/'support.json').read_text()); assert support['selected_grid']=='medium' and not support['labels_read']
 labels=pl.read_csv(ROOT/'data/development_labels.csv')
 pairs=labels.select('pair_id','player_1','player_2')
 hands=pl.read_parquet(ROOT/'data/hands.parquet').filter(pl.col('phase')=='development').select('hand_id','table_id','board_cards','button_seat','big_blind')
 ids=hands.select('hand_id'); acts=pl.scan_parquet(ROOT/'data/actions.parquet').join(ids.lazy(),on='hand_id',how='semi').collect().sort('hand_id','action_no')
 seats=pl.scan_parquet(ROOT/'data/seats.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
 cls=pl.when(pl.col('action')=='fold').then(0).when(pl.col('action')=='check').then(1).when((pl.col('action')=='call')|((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call')))).then(2).otherwise(3)
 totals=acts.with_columns(cls.alias('kind')).group_by('player_id').agg(*[(pl.col('kind')==k).sum().alias('n'+str(k)) for k in range(4)])
 tl={r[0]:np.array(r[1:]) for r in totals.iter_rows()}
 # Actual aggressor, and only physical actions made by nominated actors.
 aa=acts.with_columns((pl.col('action').is_in(['bet','raise'])|((pl.col('action')=='all_in')&(pl.col('amount')>pl.col('to_call')))).alias('_agg')).with_columns(pl.when('_agg').then('player_id').otherwise(None).forward_fill().shift(1).over('hand_id','street').alias('target'))
 members=pl.concat([pairs.select(pl.col('player_1').alias('actor')),pairs.select(pl.col('player_2').alias('actor'))]).unique()
 eligible=aa.join(members,left_on='player_id',right_on='actor',how='semi').filter(pl.col('target').is_not_null()&(pl.col('target')!=pl.col('player_id'))).select('hand_id','action_no',pl.col('player_id').alias('actor'),'target').unique()
 relids=eligible.select('hand_id').unique(); hands=hands.join(relids,on='hand_id',how='semi'); acts=acts.join(relids,on='hand_id',how='semi'); seats=seats.join(relids,on='hand_id',how='semi')
 ep={(h,a): (actor,target) for h,a,actor,target in eligible.iter_rows()}
 hp=defaultdict(set)
 for h,a,actor,target in eligible.iter_rows(): hp[h].add((actor,target))
 hs=defaultdict(list); ha=defaultdict(list)
 for r in seats.iter_rows(named=True): hs[r['hand_id']].append(r)
 for r in acts.iter_rows(named=True): ha[r['hand_id']].append(r)
 rows=[]
 for hand in hands.filter(pl.col('hand_id').is_in(list(hp))).iter_rows(named=True):
  h=hand['hand_id']; observed=ha[h]; curr=defaultdict(lambda:np.zeros(4,dtype=int))
  for a in observed: curr[a['player_id']][action_class(a)]+=1
  styles={s['player_id']:((tl.get(s['player_id'],np.zeros(4,dtype=int))-curr[s['player_id']]+1)/(max(0,(tl.get(s['player_id'],np.zeros(4,dtype=int))-curr[s['player_id']]).sum())+4)).tolist() for s in hs[h]}
  synth=[(f'{h}|{a}|{b}',a,b) for a,b in hp[h]]
  for r in extract_hand(hand,hs[h],observed,synth,styles):
   actor,target=ep.get((h,r['action_no']),(None,None))
   if actor is not None and r['pair_id']==f'{h}|{actor}|{target}' and r['facing_partner']==1: r['actor']=actor; r['target']=target; rows.append(r)
 card_features.cache_clear()
 frame=pl.DataFrame(rows).with_columns(pl.col(INPUTS+['partner_strength','observed_log_amount_bb']).cast(pl.Float32))
 assert frame.select(pl.struct('hand_id','action_no','actor').n_unique()).item()==frame.height==eligible.height
 # Apply the exact saved whole-table OOF policy bundle.
 pieces=[]
 for name in ['inner0','inner1','inner2','held']:
  rec=json.loads((ROOT/f'reports/conditional-action-experiment/policy-screen/{name}.json').read_text()); part=frame.filter(pl.col('table_id').is_in(rec['scored_pools']))
  if not part.height: continue
  b=joblib.load(ROOT/f'reports/conditional-action-experiment/policy-screen/{name}-models.joblib'); X=part.select(b['inputs']).to_numpy().astype(np.float32); p=b['classifier'].predict_proba(X,num_threads=2); y=part['action_class'].to_numpy(); res=(np.eye(4)[y]-p)/np.sqrt(p*(1-p)+.05); fold=np.clip(res[:,0],-4,4)*part['own_strength'].to_numpy()*(1-part['price'].to_numpy())
  pieces.append(part.select('hand_id','action_no','table_id','actor','target','street_index','own_strength','price','log_pot_bb','log_stack_bb','players_active').with_columns(pl.Series('score',fold.astype(np.float32))))
 scored=pl.concat(pieces); assert scored.height==frame.height
 scored.write_parquet(OUT/'actual_target_residuals.parquet')
 # Expand once to original pair membership, select fixed medium bins, and contrast.
 mapping=pl.concat([pairs.select('pair_id',pl.col('player_1').alias('actor'),pl.col('player_2').alias('partner')).with_columns(pl.lit(0).alias('direction')),pairs.select('pair_id',pl.col('player_2').alias('actor'),pl.col('player_1').alias('partner')).with_columns(pl.lit(1).alias('direction'))])
 x=scored.join(mapping,on='actor',how='inner').with_columns((pl.col('target')==pl.col('partner')).alias('treated'))
 cuts={'own_strength':[.32499998807907104,.5375000238418579],'log_pot_bb':[1.7047480344772339,2.2512917518615723],'log_stack_bb':[4.574710845947266,4.832305908203125]}
 for c,cc in cuts.items(): x=x.with_columns(sum([(pl.col(c)>v).cast(pl.Int8) for v in cc],start=pl.lit(0)).alias(c+'_bin'))
 x=x.with_columns(((pl.col('price')>.20).cast(pl.Int8)+(pl.col('price')>.50).cast(pl.Int8)).alias('price_bin'),(pl.col('players_active')>=3).cast(pl.Int8).alias('multi'))
 keys=['pair_id','direction','actor','street_index','own_strength_bin','price_bin','log_pot_bb_bin','log_stack_bb_bin','multi']
 def contrasts(z):
  c=z.group_by(keys).agg(pl.col('treated').sum().alias('nt'),(~pl.col('treated')).sum().alias('nc'),pl.col('score').filter(pl.col('treated')).mean().alias('st'),pl.col('score').filter(~pl.col('treated')).mean().alias('sc')).filter((pl.col('nt')>0)&(pl.col('nc')>0))
  d=c.group_by('pair_id','direction').agg(pl.col('nt').sum(),pl.col('nc').sum(),pl.len().alias('cells'),((pl.col('st')-pl.col('sc'))*pl.col('nt')).sum().alias('num')).with_columns((pl.col('num')/pl.col('nt')).alias('contrast')).filter((pl.col('nt')>=4)&(pl.col('nc')>=8)&(pl.col('cells')>=2))
  return d.group_by('pair_id').agg((pl.col('contrast')*pl.col('nt')).sum().alias('num'),pl.col('nt').sum()).with_columns((pl.col('num')/pl.col('nt')).alias('contrast')).select('pair_id','contrast')
 result=contrasts(x).join(labels,on='pair_id',validate='1:1'); result.write_csv(OUT/'pair_contrasts.csv')
 def summary(r): return {'n':r.height,'mean':r['contrast'].mean(),'median':r['contrast'].median()}
 normal=result.filter(pl.col('label')==0); soft=result.filter(pl.col('behavior_family')=='soft_play')
 # Whole-table bootstrap of mean difference; each pair belongs to its source table.
 table=labels.select('pair_id','table_id').join(result.select('pair_id','contrast'),on='pair_id').join(labels.select('pair_id','label','behavior_family'),on='pair_id')
 tids=table['table_id'].unique().to_list(); rng=np.random.default_rng(20260919); boot=[]
 for _ in range(2000):
  draw=rng.choice(tids,len(tids),replace=True); parts=[table.filter(pl.col('table_id')==q) for q in draw]; b=pl.concat(parts); s=b.filter(pl.col('behavior_family')=='soft_play')['contrast']; n=b.filter(pl.col('label')==0)['contrast'];
  if len(s) and len(n): boot.append(float(s.mean()-n.mean()))
 post=contrasts(x.filter(pl.col('street_index')>0)).join(labels,on='pair_id')
 ps=post.filter(pl.col('behavior_family')=='soft_play'); pn=post.filter(pl.col('label')==0)
 # Rotate nominated-partner identity while keeping each action's exact actual-target policy residual fixed.
 table_players=seats.join(hands.select('hand_id','table_id'),on='hand_id').select('table_id','player_id').unique().group_by('table_id').agg(pl.col('player_id').sort().alias('players'))
 pair_tables=labels.select('pair_id','table_id').join(mapping,on='pair_id').join(table_players,on='table_id')
 placebo=[]; prng=np.random.default_rng(20260919)
 for rep in range(499):
  rotated=[]
  for pid,tid,actor,partner,direction,players in pair_tables.iter_rows():
   opts=[p for p in players if p!=actor and p!=partner]
   if not opts: opts=[p for p in players if p!=actor]
   rotated.append((pid,actor,opts[int(prng.integers(len(opts)))],direction))
  rm=pl.DataFrame(rotated,schema=['pair_id','actor','partner','direction'],orient='row')
  rr=contrasts(scored.join(rm,on='actor',how='inner').with_columns((pl.col('target')==pl.col('partner')).alias('treated'))
    .with_columns(*[sum([(pl.col(c)>v).cast(pl.Int8) for v in cc],start=pl.lit(0)).alias(c+'_bin') for c,cc in cuts.items()])
    .with_columns(((pl.col('price')>.20).cast(pl.Int8)+(pl.col('price')>.50).cast(pl.Int8)).alias('price_bin'),(pl.col('players_active')>=3).cast(pl.Int8).alias('multi'))).join(labels,on='pair_id')
  rs=rr.filter(pl.col('behavior_family')=='soft_play'); rn=rr.filter(pl.col('label')==0)
  if rs.height and rn.height: placebo.append(float(rs['contrast'].mean()-rn['contrast'].mean()))
 observed=float(soft['contrast'].mean()-normal['contrast'].mean())
 report={'protocol':'PROTOCOL.md','selected_grid':'medium','actual_target_rows':scored.height,'pair_coverage':result.height/labels.height,'all':summary(result),'soft_play':summary(soft),'confirmed_non_target':summary(normal),'primary_mean_difference':observed,'table_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'postflop':{'coverage':post.height/labels.height,'soft_n':ps.height,'normal_n':pn.height,'mean_difference':(ps['contrast'].mean()-pn['contrast'].mean()) if ps.height and pn.height else None},'rotation_placebo':{'replicates':len(placebo),'mean':float(np.mean(placebo)),'q95':float(np.quantile(placebo,.95)),'one_sided_p':float((1+sum(v>=observed for v in placebo))/(1+len(placebo)))},'elapsed_seconds':time.time()-t,'fit_performed':False}
 (OUT/'quality.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
