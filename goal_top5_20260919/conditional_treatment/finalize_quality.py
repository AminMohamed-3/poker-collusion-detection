from pathlib import Path
import os,json,time,numpy as np,polars as pl
os.environ.setdefault('POLARS_MAX_THREADS','2')
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
labels=pl.read_csv(ROOT/'data/development_labels.csv'); pairs=labels.select('pair_id','player_1','player_2'); scored=pl.read_parquet(OUT/'actual_target_residuals.parquet')
mapping=pl.concat([pairs.select('pair_id',pl.col('player_1').alias('actor'),pl.col('player_2').alias('partner')).with_columns(pl.lit(0).alias('direction')),pairs.select('pair_id',pl.col('player_2').alias('actor'),pl.col('player_1').alias('partner')).with_columns(pl.lit(1).alias('direction'))])
cuts={'own_strength':[.32499998807907104,.5375000238418579],'log_pot_bb':[1.7047480344772339,2.2512917518615723],'log_stack_bb':[4.574710845947266,4.832305908203125]}
def decorate(z):
 for c,cc in cuts.items(): z=z.with_columns(sum([(pl.col(c)>v).cast(pl.Int8) for v in cc],start=pl.lit(0)).alias(c+'_bin'))
 return z.with_columns(((pl.col('price')>.20).cast(pl.Int8)+(pl.col('price')>.50).cast(pl.Int8)).alias('price_bin'),(pl.col('players_active')>=3).cast(pl.Int8).alias('multi'))
keys=['pair_id','direction','actor','street_index','own_strength_bin','price_bin','log_pot_bb_bin','log_stack_bb_bin','multi']
def contrasts(z):
 c=z.group_by(keys).agg(pl.col('treated').sum().alias('nt'),(~pl.col('treated')).sum().alias('nc'),pl.col('score').filter(pl.col('treated')).mean().alias('st'),pl.col('score').filter(~pl.col('treated')).mean().alias('sc')).filter((pl.col('nt')>0)&(pl.col('nc')>0))
 d=c.group_by('pair_id','direction').agg(pl.col('nt').sum(),pl.col('nc').sum(),pl.len().alias('cells'),((pl.col('st')-pl.col('sc'))*pl.col('nt')).sum().alias('num')).with_columns((pl.col('num')/pl.col('nt')).alias('contrast')).filter((pl.col('nt')>=4)&(pl.col('nc')>=8)&(pl.col('cells')>=2))
 return d.group_by('pair_id').agg((pl.col('contrast')*pl.col('nt')).sum().alias('num'),pl.col('nt').sum()).with_columns((pl.col('num')/pl.col('nt')).alias('contrast')).select('pair_id','contrast')
x=decorate(scored.join(mapping,on='actor',how='inner').with_columns((pl.col('target')==pl.col('partner')).alias('treated')))
result=contrasts(x).join(labels,on='pair_id'); result.write_csv(OUT/'pair_contrasts.csv')
normal=result.filter(pl.col('label')==0); soft=result.filter(pl.col('behavior_family')=='soft_play'); observed=float(soft['contrast'].mean()-normal['contrast'].mean())
pair_tables=x.select('pair_id','table_id').unique(); assert pair_tables.group_by('pair_id').len()['len'].max()==1
tab=result.join(pair_tables,on='pair_id'); tids=tab['table_id'].unique().to_list(); rng=np.random.default_rng(20260919); boot=[]
for _ in range(2000):
 draw=rng.choice(tids,len(tids),replace=True); b=pl.concat([tab.filter(pl.col('table_id')==q) for q in draw]); s=b.filter(pl.col('behavior_family')=='soft_play')['contrast']; n=b.filter(pl.col('label')==0)['contrast']
 if len(s) and len(n): boot.append(float(s.mean()-n.mean()))
post=contrasts(x.filter(pl.col('street_index')>0)).join(labels,on='pair_id'); ps=post.filter(pl.col('behavior_family')=='soft_play'); pn=post.filter(pl.col('label')==0)
hands=pl.read_parquet(ROOT/'data/hands.parquet').select('hand_id','table_id'); seats=pl.read_parquet(ROOT/'data/seats.parquet').join(hands,on='hand_id').select('table_id','player_id').unique().group_by('table_id').agg(pl.col('player_id').sort().alias('players'))
pt=mapping.join(pair_tables,on='pair_id').join(seats,on='table_id'); prng=np.random.default_rng(20260919); placebo=[]
for rep in range(499):
 rot=[]
 for pid,actor,partner,direction,tid,players in pt.iter_rows():
  opts=[p for p in players if p!=actor and p!=partner] or [p for p in players if p!=actor]
  rot.append((pid,actor,opts[int(prng.integers(len(opts)))],direction))
 rm=pl.DataFrame(rot,schema=['pair_id','actor','partner','direction'],orient='row')
 rr=contrasts(decorate(scored.join(rm,on='actor',how='inner').with_columns((pl.col('target')==pl.col('partner')).alias('treated')))).join(labels,on='pair_id'); rs=rr.filter(pl.col('behavior_family')=='soft_play'); rn=rr.filter(pl.col('label')==0)
 if rs.height and rn.height: placebo.append(float(rs['contrast'].mean()-rn['contrast'].mean()))
def sm(r): return {'n':r.height,'mean':r['contrast'].mean(),'median':r['contrast'].median()}
report={'protocol':'PROTOCOL.md','selected_grid':'medium','actual_target_rows':scored.height,'pair_coverage':result.height/labels.height,'all':sm(result),'soft_play':sm(soft),'confirmed_non_target':sm(normal),'primary_mean_difference':observed,'table_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'postflop':{'coverage':post.height/labels.height,'soft_n':ps.height,'normal_n':pn.height,'mean_difference':float(ps['contrast'].mean()-pn['contrast'].mean()) if ps.height and pn.height else None},'rotation_placebo':{'replicates':len(placebo),'mean':float(np.mean(placebo)),'q95':float(np.quantile(placebo,.95)),'one_sided_p':float((1+sum(v>=observed for v in placebo))/(1+len(placebo)))},'fit_performed':False}
(OUT/'quality.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
