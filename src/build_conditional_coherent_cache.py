"""Rebuild held observation baselines after sampling complete pool hands."""
import os
os.environ['POLARS_MAX_THREADS']='2'
import sys
import json
import gc
import time
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha
from conditional_inference_worker import extract_chunk
from conditional_action_features import card_features
sys.path.insert(0,str(ROOT.parents[1]/'research/coherent-history-thinning'))
from extractor import sample_hands,replace_residuals

OUT=ROOT/'reports/conditional-coherent-history'


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    identity={'plan_sha256':sha(OUT/'plan.json'),'source_sha256':sha(__file__),
        'extractor_sha256':sha(ROOT.parents[1]/'research/coherent-history-thinning/extractor.py'),
        'conditional_sha256':sha(ROOT/'src/conditional_action_features.py'),
        'worker_sha256':sha(ROOT/'src/conditional_inference_worker.py')}
    idpath=OUT/'cache-identity.json'
    if idpath.exists(): assert json.loads(idpath.read_text())==identity
    else: idpath.write_text(json.dumps(identity,indent=2)+'\n')
    heldfiles=[ROOT/'reports/tabicl-screen/held-pairs.csv']+[ROOT/'reports/tabicl-confirmation'/f'fold{i}/held-pairs.csv' for i in [1,2]]
    pools=sorted(set().union(*[set(pl.read_csv(p)['table_id']) for p in heldfiles]))
    labels=pl.read_csv(ROOT/'data/development_labels.csv')
    raw=pl.read_parquet(sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet')))
    for fraction in plan['fractions']:
        folder=OUT/str(fraction); folder.mkdir(exist_ok=True)
        for offset in range(0,len(pools),20):
            name=f'batch_{offset//20:03}'; receipt=folder/(name+'.json')
            if receipt.exists():
                saved=json.loads(receipt.read_text()); assert saved['identity']==identity
                assert all(sha(folder/n)==h for n,h in saved['files'].items()); continue
            start=time.perf_counter(); subset=pools[offset:offset+20]
            hands=pl.scan_parquet(ROOT/'data/hands.parquet').filter((pl.col('phase')=='development')&pl.col('table_id').is_in(subset)).select('hand_id','table_id','phase','big_blind','board_cards','button_seat').collect()
            sampled=sample_hands(hands,fraction,plan['sampling_seed']); ids=sampled.select('hand_id')
            seats=pl.scan_parquet(ROOT/'data/seats.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
            actions=pl.scan_parquet(ROOT/'data/actions.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
            original=raw.filter(pl.col('table_id').is_in(subset))
            shortened,baselines=replace_residuals(original,labels,sampled,seats,actions)
            cls=pl.when(pl.col('action')=='fold').then(0).when(pl.col('action')=='check').then(1).when((pl.col('action')=='call')|((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call')))).then(2).otherwise(3)
            totals=actions.with_columns(cls.alias('kind')).group_by('player_id').agg(*[(pl.col('kind')==k).sum().alias('n'+str(k)) for k in range(4)])
            counts={r[0]:np.array(r[1:],dtype=np.int64) for r in totals.iter_rows()}
            keys=shortened.select('pair_id','hand_id','table_id').join(labels.select('pair_id','player_1','player_2'),on='pair_id',validate='m:1')
            needed=keys.select('hand_id').unique()
            frame=extract_chunk(keys,sampled.join(needed,on='hand_id',how='semi'),seats.join(needed,on='hand_id',how='semi'),actions.join(needed,on='hand_id',how='semi'),counts)
            files=[folder/(name+'-raw.parquet'),folder/(name+'-actions.parquet'),folder/(name+'-baselines.parquet')]
            for data,path in zip([shortened,frame,baselines],files): data.write_parquet(path)
            saved={'identity':identity,'fraction':fraction,'pools':subset,'full_pool_hands':hands.height,'sampled_pool_hands':sampled.height,
                'full_pair_hands':original.height,'retained_pair_hands':shortened.height,'action_rows':frame.height,
                'files':{p.name:sha(p) for p in files},'seconds':time.perf_counter()-start}
            receipt.write_text(json.dumps(saved,indent=2)+'\n'); print(json.dumps({k:v for k,v in saved.items() if k not in ['identity','pools','files']}),flush=True)
            del hands,sampled,seats,actions,original,shortened,frame; card_features.cache_clear(); gc.collect()
    print(json.dumps({'complete':True,'pools':len(pools),'fractions':plan['fractions']}),flush=True)


if __name__=='__main__': main()
