"""Fit only cash-balance/reset bookkeeping from original outer-training pools."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent
h=pl.read_parquet(H/'training-hands.parquet')
s=pl.read_parquet(H/'training-seats.parquet')
f=s.join(h.select('hand_id','table_id','started_at','big_blind'),on='hand_id').sort('player_id','started_at','hand_id')
f=f.with_columns((pl.col('starting_stack')/pl.col('big_blind')).alias('bb'),(pl.col('net_chips')/pl.col('big_blind')).alias('net_bb'))
f=f.with_columns(pl.col('bb').shift(1).over('player_id').alias('previous_bb'),pl.col('net_bb').shift(1).over('player_id').alias('previous_net_bb'))
first=f.filter(pl.col('previous_bb').is_null()).select('player_id','bb')
t=f.filter(pl.col('previous_bb').is_not_null()).with_columns((pl.col('previous_bb')+pl.col('previous_net_bb')).alias('balance'))
t=t.with_columns((pl.col('bb')-pl.col('balance')).abs().gt(1e-6).alias('changed'))
reset=t.filter(pl.col('changed')).group_by('bb').len().sort('len',descending=True)
assert reset.height==4, reset
levels=reset['bb'].to_list();counts=reset['len'].to_numpy();weights=(counts/counts.sum()).tolist()
# Equal post-balance and reset values hide some reset events. Estimate hazard
# only where every possible reset destination is observably different.
eligible=t.filter(pl.col('balance').is_between(40,250)&~pl.col('balance').is_in(levels))
p=float(eligible['changed'].mean())
regions=[]
for name,mask in [('below',pl.col('balance')<40),('inside',pl.col('balance').is_between(40,250)),('above',pl.col('balance')>250)]:
    part=t.filter(mask);res=part.filter(pl.col('changed')).group_by('bb').len().sort('bb')
    regions.append({'region':name,'n':part.height,'changed_fraction':float(part['changed'].mean()),'reset_counts':res.to_dicts()})
monthly=t.with_columns((pl.col('started_at').dt.epoch('s')//86400).alias('day')).group_by('day').agg(pl.len().alias('n'),pl.col('changed').mean().alias('changed_fraction')).sort('day')
r={'created_at':datetime.now(timezone.utc).isoformat(),'training_pools':sorted(h['table_id'].unique().to_list()),'min_balance_bb':40,'max_balance_bb':250,'spontaneous_reset_probability':p,'reset_stack_bb':levels,'reset_weights':weights,'eligible_transitions':eligible.height,'all_transitions':t.height,'regions':regions,'first_stack_by_player':dict(first.iter_rows()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'interpretation':'Empirical transition rule, not claimed original generator source. Carry forward exact game outcome; reset below40 or above250; independent fitted spontaneous reset within bounds.'}
out=H/'balance-process.json';assert not out.exists();out.write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k not in ['training_pools','first_stack_by_player']},indent=2))
