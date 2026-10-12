"""Calibrate generation only from the current outer-training pools."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import hashlib
import json
from pathlib import Path
import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
meta_path = ROOT / 'reports/mil-pair-experiment/fold0/metadata.json'
meta = json.loads(meta_path.read_text())
pools = sorted({r['table_id'] for r in meta['train_full']['pairs']})
assert len(pools) == 297
assert not set(pools) & {r['table_id'] for r in meta['full']['pairs']}
hands = pl.read_parquet(ROOT / 'data/hands.parquet').filter((pl.col('phase') == 'development') & pl.col('table_id').is_in(pools))
seats = pl.scan_parquet(ROOT / 'data/seats.parquet').join(hands.lazy().select('hand_id','table_id','big_blind'), on='hand_id', how='inner').collect()
def quantiles(series): return {str(q): series.quantile(q) for q in [0,.1,.25,.5,.75,.9,1]}
counts = seats.group_by('table_id').agg(pl.col('player_id').n_unique().alias('players'))
summary = {'training_metadata_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
           'training_pools': pools, 'training_pairs': len(meta['train_full']['pairs']),
           'held_pools_excluded': True, 'hands': hands.height,
           'players_per_pool': quantiles(counts['players']),
           'hands_per_pool': quantiles(hands.group_by('table_id').len()['len']),
           'stack_bb': quantiles(seats['starting_stack']/seats['big_blind']),
           'players_dealt': hands.group_by('players_dealt').len().to_dicts(),
           'board_card_counts': hands.with_columns(pl.col('board_cards').fill_null('').str.split(' ').list.len().alias('cards')).group_by('cards').len().to_dicts(),
           'blinds': hands.group_by('big_blind').len().to_dicts()}
(HERE / 'training-calibration.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='training_pools'},indent=2))
