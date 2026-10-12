"""Inspect session turnover, seating and button movement using training pools only."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import json
import numpy as np
import polars as pl
from diagnose import quant
H=Path(__file__).resolve().parent
h=pl.read_parquet(H/'training-hands.parquet').sort('table_id','started_at','hand_id').with_columns(pl.int_range(pl.len()).over('table_id').alias('index'))
s=pl.read_parquet(H/'training-seats.parquet')
f=s.join(h.select('hand_id','table_id','index','button_seat'),on='hand_id').sort('table_id','index','seat_no')
f=f.with_columns(pl.col('player_id').shift(1).over('table_id','seat_no').alias('previous_occupant'))
f=f.with_columns((pl.col('player_id')!=pl.col('previous_occupant')).fill_null(False).alias('changed'))
hands=f.group_by('table_id','hand_id','index').agg(pl.col('changed').sum().alias('n_changed')).sort('table_id','index')
totals=hands.filter(pl.col('index')>0).group_by('n_changed').len().sort('n_changed')
by_pool=hands.filter(pl.col('index')>0).group_by('table_id').agg((pl.col('n_changed')>0).mean().alias('turnover_event_fraction'),pl.col('n_changed').mean().alias('seats_changed_mean'))
fp=f.sort('player_id','index').with_columns(pl.col('index').diff().over('player_id').alias('gap'),pl.col('seat_no').shift(1).over('player_id').alias('previous_seat'))
fp=fp.with_columns(((pl.col('gap')!=1)|pl.col('gap').is_null()).cast(pl.Int32).cum_sum().over('player_id').alias('session'))
runs=fp.group_by('player_id','session').len()
trans=h.with_columns(((pl.col('button_seat')-pl.col('button_seat').shift(1).over('table_id'))%6).alias('button_step'))
gaps=fp.filter(pl.col('gap')>1)
example=h.filter(pl.col('table_id')==h['table_id'][0]).head(5)
observed=[]
for row in example.with_columns(pl.col('started_at').cast(pl.String)).to_dicts():
    observed.append({'index':row['index'],'button':row['button_seat'],'seats':f.filter(pl.col('hand_id')==row['hand_id']).select('seat_no','player_id').to_dicts()})
out={'turnover_counts':totals.to_dicts(),'pool_turnover_rate':quant(by_pool['turnover_event_fraction']),'pool_seats_changed_mean':quant(by_pool['seats_changed_mean']),'consecutive_appearance_fraction':float((fp['gap']==1).mean()),'same_physical_seat_given_consecutive':float((fp.filter(pl.col('gap')==1)['seat_no']==fp.filter(pl.col('gap')==1)['previous_seat']).mean()),'session_length':quant(runs['len']),'session_length_top_counts':runs.group_by('len').len(name='count').sort('count',descending=True).head(20).to_dicts(),'away_gap':quant(gaps['gap']),'button_step_counts':trans.group_by('button_step').len().sort('button_step').to_dicts(),'example_first_hands':observed}
(H/'attendance-diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
hands.write_parquet(H/'training-seat-transitions.parquet')
