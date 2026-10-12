"""Verify physical-seat events and measure complete-session schedules."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import hashlib,json,sys
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent;D=H/(sys.argv[1] if len(sys.argv)>1 else 'persistent_sessions')
original=json.loads((H/'attendance-diagnosis.json').read_text());counts=np.zeros(7,dtype=int);button_steps=np.zeros(6,dtype=int);event_counts={'refresh':0,'turnover':0};runs=[];total=0;consecutive=0;same_seat=0
for folder in sorted((D/'pilot').glob('pool-*')):
    h=pl.read_parquet(folder/'hands.parquet').sort('started_at','hand_id').with_row_index('index')
    s=pl.read_parquet(folder/'seats.parquet');f=s.join(h.select('hand_id','index'),on='hand_id').sort('index','seat_no')
    occupancy=f.group_by('index',maintain_order=True).agg(pl.col('player_id'))['player_id'].to_list()
    assert len(occupancy)==3000 and all(len(p)==len(set(p))==6 for p in occupancy)
    ledger={r['hand_number']:r for r in json.loads((folder/'seat-events.json').read_text())}
    for i in range(1,3000):
        old,new=occupancy[i-1],occupancy[i];changed=sum(a!=b for a,b in zip(old,new));counts[changed]+=1
        e=ledger.pop(i,None)
        if e:
            assert e['before']==old and e['after']==new
            event_counts[e['mode']]+=1
            if e['mode']=='turnover':
                assert changed>=1
                assert all(new[j] not in old for j in range(6) if old[j]!=new[j])
        else:assert changed==0
    assert not ledger
    buttons=h['button_seat'].to_numpy();button_steps+=np.bincount((buttons[1:]-buttons[:-1])%6,minlength=6)
    pf=f.sort('player_id','index').with_columns(pl.col('index').diff().over('player_id').alias('gap'),pl.col('seat_no').shift(1).over('player_id').alias('previous_seat'))
    pf=pf.with_columns(((pl.col('gap')!=1)|pl.col('gap').is_null()).cast(pl.Int32).cum_sum().over('player_id').alias('session'))
    runs.extend(pf.group_by('player_id','session').len()['len'].to_list());c=pf.filter(pl.col('gap')==1)
    total+=pf.height;consecutive+=c.height;same_seat+=int((c['seat_no']==c['previous_seat']).sum())
observed=np.array([r['len'] for r in original['turnover_counts']]);tv=float(.5*np.abs(counts/counts.sum()-observed/observed.sum()).sum())
fraction=consecutive/total;seat_fraction=same_seat/consecutive;median=float(np.median(runs))
r={'all_seat_event_ledger_entries_verified':True,'transitions':int(counts.sum()),'changed_seat_counts':counts.tolist(),'events':event_counts,'turnover_total_variation_from_original':tv,'consecutive_appearance_fraction':fraction,'original_consecutive_appearance_fraction':original['consecutive_appearance_fraction'],'same_seat_given_consecutive':seat_fraction,'session_length_median':median,'session_length_q90':float(np.quantile(runs,.9)),'button_step_counts':button_steps.tolist(),'schedule_gate_passed':tv<=.02 and abs(fraction-original['consecutive_appearance_fraction'])<=.01 and seat_fraction>=.98,'predeclared_thresholds':{'turnover_total_variation_max':.02,'consecutive_fraction_absolute_gap_max':.01,'same_seat_given_consecutive_min':.98},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(D/'seat-schedule-audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
