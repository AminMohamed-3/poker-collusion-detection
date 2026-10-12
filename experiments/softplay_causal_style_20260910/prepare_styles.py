"""Public action tendencies computed strictly before the current hand."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time
import polars as pl
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (H/'training-actions-causal.parquet').exists();start=time.perf_counter()
    pools=json.loads((H.parent/'softplay_replication_20260910/b0_calibrated/training-calibration.json').read_text())['training_pools'];assert len(pools)==297
    hands=pl.scan_parquet(R/'data/hands.parquet').filter(pl.col('table_id').is_in(pools)).select('hand_id',pl.col('started_at').cast(pl.Int64).alias('_time')).collect()
    kind=pl.when(pl.col('action')=='fold').then(0).when(pl.col('action')=='check').then(1).when((pl.col('action')=='call')|((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call')))).then(2).otherwise(3)
    counts=pl.scan_parquet(R/'data/actions.parquet').join(hands.lazy().select('hand_id'),on='hand_id',how='semi').with_columns(kind.alias('_kind')).group_by('hand_id','player_id').agg(*[(pl.col('_kind')==i).sum().alias(f'n{i}') for i in range(4)]).collect()
    ledger=pl.scan_parquet(R/'data/seats.parquet').select('hand_id','player_id').join(hands.lazy(),on='hand_id',how='inner').collect().join(counts,on=['hand_id','player_id'],how='left').fill_null(0).sort('player_id','_time','hand_id')
    ledger=ledger.with_columns(*[(pl.col(f'n{i}').cum_sum().over('player_id')-pl.col(f'n{i}')).alias(f'past{i}') for i in range(4)])
    denominator=pl.sum_horizontal([pl.col(f'past{i}') for i in range(4)])+4
    ledger=ledger.with_columns(*[((pl.col(f'past{i}')+1)/denominator).cast(pl.Float32).alias('style_'+n) for i,n in enumerate(['fold','check','call','raise'])])
    # Independent prefix checks within sampled player streams, including first hand.
    checked=0
    for player in sorted(ledger['player_id'].unique().to_list())[::137]:
        frame=ledger.filter(pl.col('player_id')==player);past=[0]*4
        for row in frame.iter_rows(named=True):
            assert [row[f'past{i}'] for i in range(4)]==past
            past=[past[i]+row[f'n{i}'] for i in range(4)];checked+=1
    f=pl.read_parquet(H.parent/'softplay_fast_20260910/training-actions.parquet').with_row_index('_order');assert set(f['table_id'])<=set(pools)
    labels=pl.read_csv(R/'data/development_labels.csv').select('pair_id','player_1','player_2')
    f=f.join(labels,on='pair_id',validate='m:1').with_columns(pl.when(pl.col('direction')==0).then(pl.col('player_1')).otherwise(pl.col('player_2')).alias('_own'),pl.when(pl.col('direction')==0).then(pl.col('player_2')).otherwise(pl.col('player_1')).alias('_partner'))
    cols=['style_'+n for n in ['fold','check','call','raise']]
    for role in ['own','partner']:
        ref=ledger.select('hand_id',pl.col('player_id').alias('_'+role),*[pl.col(n).alias('causal_'+role+'_'+n) for n in cols])
        f=f.join(ref,on=['hand_id','_'+role],how='left',validate='m:1')
    f=f.sort('_order').drop('_order','player_1','player_2','_own','_partner');assert not any(f[c].null_count() for c in f.columns if c.startswith('causal_'));f.write_parquet(H/'training-actions-causal.parquet')
    receipt={'at':datetime.now(timezone.utc).isoformat(),'training_pools':pools,'hands':hands.height,'player_hand_rows':ledger.height,'action_context_rows':f.height,'independent_prefix_rows_checked':checked,'no_current_or_future_hand_counts_used':True,'training_only':True,'uniform_pseudocount_per_class':1,'source_sha256':sha(Path(__file__)),'data_sha256':sha(H/'training-actions-causal.parquet'),'seconds':time.perf_counter()-start};(H/'causal-style-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='training_pools'},indent=2))
if __name__=='__main__':main()
