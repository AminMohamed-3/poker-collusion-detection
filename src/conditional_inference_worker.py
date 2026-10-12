"""Portable bounded-memory evaluation feature extraction; no labels or fitting."""
import os
os.environ['POLARS_MAX_THREADS']='2'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import argparse
from pathlib import Path
from collections import defaultdict
import gc
import json
import time
import joblib
import numpy as np
import polars as pl
pl.thread_pool_size()
from conditional_action_features import INPUTS, extract_hand, action_class, card_features
from run_conditional_policy_screen import hand_features, sha


def pair_aggregate(frame,features):
    return frame.group_by('pair_id','table_id').agg(
        *[pl.col(c).mean().alias(c+'_mean') for c in features],
        *[pl.col(c).std().fill_null(0).alias(c+'_std') for c in features],
        *[pl.col(c).max().alias(c+'_max') for c in features],
        *[pl.col(c).quantile(.95).alias(c+'_q95') for c in features])


def extract_chunk(keys,hands,seats,actions,counts):
    hand_pairs=defaultdict(list); hand_seats=defaultdict(list); hand_actions=defaultdict(list)
    for pair,hand,first,second in keys.select('pair_id','hand_id','player_1','player_2').iter_rows(): hand_pairs[hand].append((pair,first,second))
    for row in seats.iter_rows(named=True): hand_seats[row['hand_id']].append(row)
    for row in actions.sort('hand_id','action_no').iter_rows(named=True): hand_actions[row['hand_id']].append(row)
    rows=[]
    for hand in hands.iter_rows(named=True):
        h=hand['hand_id']; observed=hand_actions[h]
        current=defaultdict(lambda:np.zeros(4,dtype=np.int64))
        for action in observed: current[action['player_id']][action_class(action)]+=1
        styles={}
        for seat in hand_seats[h]:
            player=seat['player_id']; other=counts.get(player,np.zeros(4,dtype=np.int64))-current[player]
            assert other.min()>=0
            styles[player]=((other+1)/(other.sum()+4)).tolist()
        rows.extend(extract_hand(hand,hand_seats[h],observed,hand_pairs[h],styles))
    frame=pl.DataFrame(rows).with_columns(pl.col(INPUTS+['partner_strength','observed_log_amount_bb']).cast(pl.Float32))
    assert frame.select(pl.struct('pair_id','direction','hand_id','action_no').n_unique()).item()==frame.height
    return frame


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--batches',type=int,nargs='+',required=True)
    args=parser.parse_args(); root=args.root.resolve(); planpath=root/'config.json'; plan=json.loads(planpath.read_text())
    assert all(sha(root/name)==digest for name,digest in plan['files'].items())
    out=root/'results'; out.mkdir(exist_ok=True)
    policy=joblib.load(root/'final-models.joblib'); assert policy['inputs']==INPUTS
    allhands=pl.read_parquet(root/'data/hands.parquet')
    pairs=pl.read_csv(root/'data/evaluation_pairs.csv').select('pair_id','player_1','player_2')
    selected=pl.read_parquet(root/'selected-pairs.parquet')
    counts={r[0]:np.array(r[1:],dtype=np.int64) for r in pl.read_parquet(root/'style-counts.parquet').iter_rows()}
    for index in args.batches:
        name=f'batch_{index:03}'; receipt=out/(name+'.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['plan_sha256']==sha(planpath)
            assert all(sha(out/p)==h for p,h in saved['files'].items())
            print(json.dumps({'verified_existing_batch':index}),flush=True); continue
        started=time.perf_counter()
        keys=pl.read_parquet(root/'keys'/(name+'.parquet')).join(pairs,on='pair_id',validate='m:1')
        pools=sorted(keys['table_id'].unique()); aggregates=[]; evidence=[]; actions_count=0; missing_count=0
        for offset in range(0,len(pools),2):
            subset=pools[offset:offset+2]; k=keys.filter(pl.col('table_id').is_in(subset)); ids=k.select('hand_id').unique()
            h=allhands.join(ids,on='hand_id',how='semi')
            s=pl.scan_parquet(root/'data/seats.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
            a=pl.scan_parquet(root/'data/actions.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
            frame=extract_chunk(k,h,s,a,counts); actions_count+=frame.height
            X=frame.select(INPUTS).to_numpy().astype(np.float32)
            ca=hand_features(frame,policy['classifier'].predict_proba(X,num_threads=1),policy['regressor'].predict(X,num_threads=1))
            caf=[c for c in ca.columns if c.startswith('ca_')]; assert caf==plan['conditional_features']
            missing_count+=k.select('pair_id','hand_id').join(ca.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='anti').height
            ca=k.select('pair_id','hand_id','table_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0))
            assert ca.height==k.height
            aggregates.append(pair_aggregate(ca,caf)); evidence.append(ca.join(selected,on='pair_id',how='semi'))
            print(json.dumps({'batch':index,'pools':offset+len(subset),'total_pools':len(pools),'pair_hands':k.height,'seconds':time.perf_counter()-started}),flush=True)
            del k,h,s,a,frame,X,ca; card_features.cache_clear(); gc.collect()
        pairframe=pl.concat(aggregates).sort('pair_id'); evidenceframe=pl.concat(evidence).sort('pair_id','hand_id')
        assert pairframe['pair_id'].n_unique()==pairframe.height==keys['pair_id'].n_unique()
        expected=keys.join(selected,on='pair_id',how='semi').select('pair_id','hand_id').sort('pair_id','hand_id')
        assert expected.equals(evidenceframe.select('pair_id','hand_id'))
        pairpath=out/(name+'-pairs.parquet'); evidencepath=out/(name+'-evidence.parquet')
        pairframe.write_parquet(pairpath); evidenceframe.write_parquet(evidencepath)
        report={'plan_sha256':sha(planpath),'files':{p.name:sha(p) for p in [pairpath,evidencepath]},
            'pairs':pairframe.height,'pair_hands':keys.height,'evidence_hands':evidenceframe.height,'action_rows':actions_count,
            'pair_hands_without_eligible_action':missing_count,'seconds':time.perf_counter()-started}
        receipt.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'complete_batch':index,**report}),flush=True)


if __name__=='__main__': main()
