from pathlib import Path
import json, os, time
os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def main():
    started=time.time()
    cache=pl.read_parquet(sorted((ROOT/'reports/conditional-action-experiment/action-cache').glob('*.parquet')))
    pairs=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','player_1','player_2')
    mapping=pl.concat([
        pairs.select('pair_id',pl.col('player_1').alias('actor'),pl.col('player_2').alias('partner')).with_columns(pl.lit(0).alias('direction')),
        pairs.select('pair_id',pl.col('player_2').alias('actor'),pl.col('player_1').alias('partner')).with_columns(pl.lit(1).alias('direction'))])
    # Map cached direction to actor, then retain one copy of each physical decision.
    keyed=cache.join(mapping,on=['pair_id','direction'],validate='m:1')
    own=['hand_id','action_no','actor','table_id','street_index','own_strength','price','log_pot_bb','log_stack_bb','players_active']
    physical=keyed.select(own).unique(['hand_id','action_no','actor'],keep='first')
    consistency=keyed.group_by('hand_id','action_no','actor').agg(*[pl.col(c).n_unique().alias(c) for c in own[3:]])
    assert consistency.select(pl.max_horizontal(pl.exclude('hand_id','action_no','actor')).max()).item()==1

    actions=pl.scan_parquet(ROOT/'data/actions.parquet').select('hand_id','player_id','action_no','street','action','amount','to_call').collect().sort('hand_id','street','action_no')
    actions=actions.with_columns((pl.col('action').is_in(['bet','raise'])|((pl.col('action')=='all_in')&(pl.col('amount')>pl.col('to_call')))).alias('_agg'))
    actions=actions.with_columns(pl.when('_agg').then('player_id').otherwise(None).forward_fill().shift(1).over('hand_id','street').alias('actual_target'))
    targets=actions.select('hand_id','action_no',pl.col('player_id').alias('actor'),'actual_target')
    physical=physical.join(targets,on=['hand_id','action_no','actor'],validate='1:1').filter(pl.col('actual_target').is_not_null() & (pl.col('actual_target')!=pl.col('actor')))
    expanded=physical.join(mapping,on='actor',how='inner',validate='m:m').with_columns((pl.col('actual_target')==pl.col('partner')).alias('treated'))

    qs={}
    for c in ['own_strength','log_pot_bb','log_stack_bb']:
        qs[c]={str(n):physical[c].quantile(n/12,'nearest') for n in [3,4,6,8,9]}
    grids={
      'fine': {'n':4,'price':[0,.10,.25,.50,1]},
      'medium': {'n':3,'price':[0,.20,.50,1]},
      'coarse': {'n':2,'price':[0,.25,1]}}
    results=[]
    for name,g in grids.items():
        n=g['n']; cuts=[i/n for i in range(1,n)]
        def qcuts(c): return [physical[c].quantile(x,'nearest') for x in cuts]
        frame=expanded
        for c in ['own_strength','log_pot_bb','log_stack_bb']:
            expr=pl.lit(0)
            for cut in qcuts(c): expr=expr+(pl.col(c)>cut).cast(pl.Int8)
            frame=frame.with_columns(expr.alias(c+'_bin'))
        pexpr=pl.lit(0)
        for cut in g['price'][1:-1]: pexpr=pexpr+(pl.col('price')>cut).cast(pl.Int8)
        frame=frame.with_columns(pexpr.alias('price_bin'),(pl.col('players_active')>=3).cast(pl.Int8).alias('multi'))
        cells=['pair_id','direction','actor','street_index','own_strength_bin','price_bin','log_pot_bb_bin','log_stack_bb_bin','multi']
        counts=frame.group_by(cells).agg(pl.col('treated').sum().alias('nt'),(~pl.col('treated')).sum().alias('nc')).filter((pl.col('nt')>0)&(pl.col('nc')>0))
        dirs=counts.group_by('pair_id','direction').agg(pl.col('nt').sum(),pl.col('nc').sum(),pl.len().alias('cells')).with_columns(((pl.col('nt')>=4)&(pl.col('nc')>=8)&(pl.col('cells')>=2)).alias('supported'))
        pairsup=dirs.group_by('pair_id').agg(pl.col('supported').sum().alias('directions'))
        results.append({'grid':name,'matched_cells':counts.height,'supported_directions':dirs.filter('supported').height,
          'pairs_one_direction':pairsup.filter(pl.col('directions')>=1).height,'pairs_both_directions':pairsup.filter(pl.col('directions')==2).height,
          'fraction_one_direction':pairsup.filter(pl.col('directions')>=1).height/pairs.height,'fraction_both_directions':pairsup.filter(pl.col('directions')==2).height/pairs.height})
    selected=next((r['grid'] for r in results if r['fraction_both_directions']>=.25 or r['fraction_one_direction']>=.40),None)
    report={'protocol':'PROTOCOL.md','labels_read':False,'cache_rows':cache.height,'unique_physical_actions':physical.height,
      'expanded_actor_pair_rows':expanded.height,'pairs':pairs.height,'quantile_receipt':qs,'grids':results,'selected_grid':selected,
      'elapsed_seconds':time.time()-started,'dedup_key':['hand_id','action_no','actor'],'actual_target':'previous aggressive player on same street'}
    (OUT/'support.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
