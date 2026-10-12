"""Recover observable stack transitions from eligible original training histories."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import json
import numpy as np
import polars as pl
from diagnose import quant,corr
H=Path(__file__).resolve().parent
h=pl.read_parquet(H/'training-hands.parquet')
s=pl.read_parquet(H/'training-seats.parquet')
f=s.join(h.select('hand_id','table_id','started_at','big_blind'),on='hand_id').sort('player_id','started_at','hand_id')
f=f.with_columns((pl.col('starting_stack')/pl.col('big_blind')).alias('bb'),(pl.col('net_chips')/pl.col('big_blind')).alias('net_bb'))
f=f.with_columns(pl.col('bb').shift(1).over('player_id').alias('previous_bb'),pl.col('net_bb').shift(1).over('player_id').alias('previous_net_bb'))
first=f.filter(pl.col('previous_bb').is_null());t=f.filter(pl.col('previous_bb').is_not_null()).with_columns((pl.col('previous_bb')+pl.col('previous_net_bb')).alias('balance'))
t=t.with_columns((pl.col('bb')-pl.col('balance')).alias('reset_delta'),(pl.col('bb')-pl.col('previous_bb')).alias('step'))
rows=[]
for lo,hi in [(-1000,40),(40,50),(50,80),(80,100),(100,150),(150,200),(200,250),(250,10000)]:
    part=t.filter((pl.col('balance')>=lo)&(pl.col('balance')<hi))
    rows.append({'previous_balance_interval':[lo,hi],'n':part.height,'unchanged_fraction':float((part['reset_delta'].abs()<1e-6).mean()),'next_stack':quant(part['bb']),'reset_delta':quant(part['reset_delta'])})
nonexact=t.filter(pl.col('reset_delta').abs()>1e-6)
r={'initial_stack':quant(first['bb']),'initial_stack_top_counts':first.group_by('bb').len().sort('len',descending=True).head(15).to_dicts(),'transitions':t.height,'exact_balance_continuation_fraction':float((t['reset_delta'].abs()<1e-6).mean()),'delta_step':quant(t['step']),'delta_after_previous_net':quant(t['reset_delta']),'step_previous_net_corr':corr(t['step'],t['previous_net_bb']),'by_balance':rows,'nonexact_start':nonexact.select('previous_bb','previous_net_bb','balance','bb','reset_delta').head(30).to_dicts(),'first_stack_constant_within_pool':first.group_by('table_id').agg(pl.col('bb').n_unique().alias('n')).group_by('n').len().to_dicts()}
(H/'stack-transitions.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
