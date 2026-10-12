"""Public-data-only availability of prior full-player histories for current keys."""
import os
os.environ['POLARS_MAX_THREADS']='2'
import sys,json,hashlib,time
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl
from contestable_state_features import target_keys
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent;OUT=HERE/'research/recent_player_history_20260912'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'coverage.json').exists(),'write-once availability check'
    prev=HERE/'experiments/symmetry_censor_20260912'
    sr=json.loads((prev/'predictions-frozen.json').read_text(encoding='utf-8'));assert sha(prev/'scores.parquet')==sr['scores_sha256']
    keys=target_keys()
    pairs=(pl.read_csv(ROOT/'data/development_labels.csv',columns=['pair_id','player_1','player_2']).join(keys.select('pair_id').unique(),on='pair_id',how='semi'))
    pmap={pid:(a,b) for pid,a,b in pairs.iter_rows()};players=sorted(set(pairs['player_1'])|set(pairs['player_2']))
    hands=(pl.scan_parquet(ROOT/'data/hands.parquet').filter(pl.col('phase')=='development').select('hand_id','table_id','started_at','big_blind').with_columns(pl.col('started_at').cast(pl.Int64)))
    records=(pl.scan_parquet(ROOT/'data/seats.parquet').filter(pl.col('player_id').is_in(players))
        .select('hand_id','player_id','starting_stack','net_chips').join(hands,on='hand_id',how='inner').sort('player_id','started_at','hand_id').collect())
    assert records.select('player_id','hand_id').n_unique()==records.height
    table_counts=records.group_by('player_id').agg(pl.col('table_id').n_unique().alias('tables'))
    assert table_counts['tables'].max()==1
    tied=records.select('player_id','started_at').n_unique()!=records.height
    assert not tied,'Cannot assign causal order to same-player timestamp ties'
    histories={k[0]:g.to_dicts() for k,g in records.group_by('player_id')}
    index={player:{r['hand_id']:i for i,r in enumerate(rows)} for player,rows in histories.items()}
    members=defaultdict(set)
    for hand,player in records.select('hand_id','player_id').iter_rows():members[hand].add(player)
    rows=[];same_stack=0;comparisons=0
    for records_one in histories.values():
        for prior,current in zip(records_one,records_one[1:]):
            assert prior['started_at']<current['started_at']
            comparisons+=1;same_stack+=int(current['starting_stack']==prior['starting_stack']+prior['net_chips'])
    for pid,hid in keys.iter_rows():
        a,b=pmap[pid]
        for player,partner in ((a,b),(b,a)):
            seq=histories[player];i=index[player][hid];past=seq[max(0,i-5):i]
            assert all(r['started_at']<seq[i]['started_at'] for r in past)
            rows.append({'pair_id':pid,'hand_id':hid,'player_id':player,'past_count':i,'past5_count':len(past),'past5_unshared_count':sum(partner not in members[r['hand_id']] for r in past)})
    coverage=pl.DataFrame(rows).sort('pair_id','hand_id','player_id');kp=OUT/'availability.parquet';coverage.write_parquet(kp)
    sources=[Path(__file__),HERE/'src/contestable_state_features.py',prev/'scores.parquet',prev/'predictions-frozen.json',*[ROOT/'data'/n for n in ('development_labels.csv','hands.parquet','seats.parquet')]]
    result={'status':'complete_public_history_availability','players':len(players),'history_rows':records.height,'shortlist_member_rows':coverage.height,'same_player_timestamp_ties':False,'maximum_pools_per_player':int(table_counts['tables'].max()),'past_count_quantiles':{str(q):float(coverage['past_count'].quantile(q,interpolation='nearest')) for q in (0,.1,.5,.9,1)},'rows_with_no_prior_hand':coverage.filter(pl.col('past_count')==0).height,'rows_with_at_least5':coverage.filter(pl.col('past_count')>=5).height,'rows_with_at_least20':coverage.filter(pl.col('past_count')>=20).height,'previous5_hands_total':int(coverage['past5_count'].sum()),'previous5_hands_not_shared_with_partner':int(coverage['past5_unshared_count'].sum()),'stack_continuity_all_target_player_history':{'comparisons':comparisons,'exact':same_stack,'fraction':same_stack/comparisons},'availability_sha256':sha(kp),'source_hashes':{str(p):sha(p) for p in sources},'created_at':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start,'limits':['No relevance, behavior, action target or outer evidence read.','started_at supplies strict ordering, but no ended_at field exists; do not claim observed end-time verification.','Previous disclosed outcomes are forensic history, not a proof of a real-time information timestamp.','Availability and unshared-history coverage are not predictive validation.']}
    (OUT/'coverage.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))

if __name__=='__main__':main()
