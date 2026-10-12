"""Independent reconstruction of styles before each generated hand."""
import polars as pl
def attach_styles(f,h,s,a,lab):
    kind=pl.when(pl.col('action')=='fold').then(0).when(pl.col('action')=='check').then(1).when((pl.col('action')=='call')|((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call')))).then(2).otherwise(3)
    counts=a.with_columns(kind.alias('_kind')).group_by('hand_id','player_id').agg(*[(pl.col('_kind')==i).sum().alias(f'n{i}') for i in range(4)])
    frame=s.select('hand_id','player_id').join(h.select('hand_id','started_at'),on='hand_id').join(counts,on=['hand_id','player_id'],how='left').fill_null(0).sort('player_id','started_at','hand_id')
    frame=frame.with_columns(*[(pl.col(f'n{i}').cum_sum().over('player_id')-pl.col(f'n{i}')).alias(f'past{i}') for i in range(4)])
    denom=pl.sum_horizontal([pl.col(f'past{i}') for i in range(4)])+4;names=['causal_own_style_'+v for v in ['fold','check','call','raise']]
    frame=frame.with_columns(*[((pl.col(f'past{i}')+1)/denom).cast(pl.Float32).alias(names[i]) for i in range(4)])
    mapping={(r['hand_id'],r['player_id']):tuple(r[n] for n in names) for r in frame.select('hand_id','player_id',*names).iter_rows(named=True)}
    f=f.join(lab.select('pair_id','player_1','player_2'),on='pair_id',validate='m:1').with_columns(pl.when(pl.col('direction')==0).then(pl.col('player_1')).otherwise(pl.col('player_2')).alias('player_id'))
    f=f.join(frame.select('hand_id','player_id',*names),on=['hand_id','player_id'],how='left',validate='m:1').drop('player_1','player_2','player_id');assert not any(f[n].null_count() for n in names)
    return f,mapping
