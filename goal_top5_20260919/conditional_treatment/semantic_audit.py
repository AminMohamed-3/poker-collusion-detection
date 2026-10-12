from pathlib import Path
import os,json,numpy as np,polars as pl
from scipy.stats import rankdata
os.environ.setdefault('POLARS_MAX_THREADS','2')
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
labels=pl.read_csv(ROOT/'data/development_labels.csv'); pairs=labels.select('pair_id','player_1','player_2'); scored=pl.read_parquet(OUT/'actual_target_residuals.parquet')
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
def auc(y,s):
 ranks=rankdata(s,method='average'); pos=y==1; return float((ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*(~pos).sum()))
def diagnostic(z):
 r=contrasts(z).join(labels,on='pair_id'); q=r.filter((pl.col('behavior_family')=='soft_play')|(pl.col('label')==0)); y=(q['behavior_family']=='soft_play').cast(pl.Int8).to_numpy(); s=q['contrast'].to_numpy(); diff=float(s[y==1].mean()-s[y==0].mean())
 pt=z.select('pair_id','table_id').unique(); assert pt.group_by('pair_id').len()['len'].max()==1
 q=q.join(pt,on='pair_id'); tids=q['table_id'].unique().to_list(); rng=np.random.default_rng(20260919); bs=[]
 for _ in range(2000):
  b=pl.concat([q.filter(pl.col('table_id')==t) for t in rng.choice(tids,len(tids),replace=True)]); yy=(b['behavior_family']=='soft_play').cast(pl.Int8).to_numpy(); ss=b['contrast'].to_numpy()
  if yy.any() and (~yy.astype(bool)).any(): bs.append(float(ss[yy==1].mean()-ss[yy==0].mean()))
 return {'pairs':r.height,'soft_n':int(y.sum()),'normal_n':int((y==0).sum()),'mean_difference':diff,'auc_signed':auc(y,s),'table_bootstrap_95':np.quantile(bs,[.025,.975]).tolist()}
report={'residual_rows':scored.height,'unique_physical_keys':scored.select(pl.struct('hand_id','action_no','actor').n_unique()).item(),'expanded_pair_direction_rows':x.height,'unique_expanded_keys':x.select(pl.struct('pair_id','direction','hand_id','action_no','actor').n_unique()).item(),'duplicate_expanded_keys':x.height-x.select(pl.struct('pair_id','direction','hand_id','action_no','actor').n_unique()).item(),'target_equals_actor':scored.filter(pl.col('target')==pl.col('actor')).height,'postflop':diagnostic(x.filter(pl.col('street_index')>0)),'preflop':diagnostic(x.filter(pl.col('street_index')==0))}
(OUT/'semantic_audit.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
