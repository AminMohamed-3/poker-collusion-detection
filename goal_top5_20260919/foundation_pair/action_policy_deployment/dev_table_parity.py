"""Rebuild one deterministic development table and compare the historical cache."""
from collections import defaultdict
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'.venv/Lib/site-packages'));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
import polars as pl
from conditional_action_features import INPUTS, action_class, card_features, extract_hand

CACHE=ROOT/'reports/conditional-action-experiment/action-cache'
KEYS=['pair_id','direction','hand_id','action_no']
FLOATS=[*INPUTS,'partner_strength','observed_log_amount_bb']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    start=time.perf_counter();receipt=Path(os.environ.get('DEV_PARITY_RECEIPT',HERE/'DEV_TABLE_PARITY.json'));assert not receipt.exists()
    cache_paths=sorted(CACHE.glob('batch_*.parquet'));assert len(cache_paths)==20
    cached_all=pl.read_parquet(cache_paths)
    table_id=sorted(cached_all['table_id'].unique().to_list())[0]
    cached=cached_all.filter(pl.col('table_id')==table_id).sort(KEYS)
    labels=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','player_1','player_2')
    scans=[pl.scan_parquet(p).filter(pl.col('table_id')==table_id).select('pair_id','hand_id')
           for p in sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet'))]
    pairs=pl.concat(scans).collect().join(labels,on='pair_id',validate='m:1')
    ids=pairs.select('hand_id').unique()
    hands=(pl.scan_parquet(ROOT/'data/hands.parquet').filter(pl.col('phase')=='development')
           .filter(pl.col('table_id')==table_id).select('hand_id','table_id','board_cards','button_seat','big_blind').collect())
    seats=pl.scan_parquet(ROOT/'data/seats.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
    actions=(pl.scan_parquet(ROOT/'data/actions.parquet').join(ids.lazy(),on='hand_id',how='semi').collect().sort('hand_id','action_no'))
    phase_ids=pl.scan_parquet(ROOT/'data/hands.parquet').filter(pl.col('phase')=='development').select('hand_id')
    all_actions=pl.scan_parquet(ROOT/'data/actions.parquet').join(phase_ids,on='hand_id',how='semi')
    kind=(pl.when(pl.col('action')=='fold').then(0).when(pl.col('action')=='check').then(1)
          .when((pl.col('action')=='call')|((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call')))).then(2).otherwise(3))
    totals=all_actions.with_columns(kind.alias('kind')).group_by('player_id').agg(
        *[(pl.col('kind')==k).sum().alias('n'+str(k)) for k in range(4)]).collect()
    total_map={r[0]:np.asarray(r[1:],dtype=np.int64) for r in totals.iter_rows()}
    hp,hs,ha=defaultdict(list),defaultdict(list),defaultdict(list)
    for pair,hand,first,second in pairs.iter_rows():hp[hand].append((pair,first,second))
    for row in seats.iter_rows(named=True):hs[row['hand_id']].append(row)
    for row in actions.iter_rows(named=True):ha[row['hand_id']].append(row)
    rows=[]
    for hand in hands.iter_rows(named=True):
        hid=hand['hand_id'];observed=ha[hid];current=defaultdict(lambda:np.zeros(4,dtype=np.int64))
        for action in observed:current[action['player_id']][action_class(action)]+=1
        styles={}
        for seat in hs[hid]:
            p=seat['player_id'];counts=total_map.get(p,np.zeros(4,dtype=np.int64))-current[p];assert counts.min()>=0
            styles[p]=((counts+1)/(counts.sum()+4)).tolist()
        rows.extend(extract_hand(hand,hs[hid],observed,hp[hid],styles))
    rebuilt=pl.DataFrame(rows).with_columns(pl.col(FLOATS).cast(pl.Float32)).sort(KEYS)
    assert rebuilt.height==cached.height and rebuilt.select(KEYS).equals(cached.select(KEYS))
    assert rebuilt['action_class'].equals(cached['action_class'])
    diffs={c:float(np.max(np.abs(rebuilt[c].to_numpy().astype(np.float64)-cached[c].to_numpy().astype(np.float64)))) for c in FLOATS}
    exact={c:bool(np.array_equal(rebuilt[c].to_numpy(),cached[c].to_numpy())) for c in FLOATS}
    assert max(diffs.values())==0.0 and all(exact.values())
    out={'passed':True,'table_selection':'lexicographically first table in full historical development action cache',
         'table_id':table_id,'rows':rebuilt.height,'pairs':rebuilt['pair_id'].n_unique(),'hands':rebuilt['hand_id'].n_unique(),
         'key_columns':KEYS,'action_class_exact':True,'float_cast':'both historical cache and rebuild compared as float32',
         'float_tolerance':0.0,'all_float_columns_exact':True,'max_abs_diff_by_column':diffs,
         'elapsed_seconds':time.perf_counter()-start,
         'source_snapshot_sha256':sha(HERE/'conditional_action_features.py'),
         'historical_cache_files_sha256':{p.name:sha(p) for p in cache_paths}}
    receipt.write_text(json.dumps(out,indent=2)+'\n');card_features.cache_clear()
if __name__=='__main__':main()
