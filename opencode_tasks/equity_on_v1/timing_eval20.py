import os
for _k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='2'
os.environ['PYTHONUTF8']='1'
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode=True
import time, pathlib
import polars as pl
import numpy as np
sys.path.insert(0, r'C:\Data\Projects\Money Making\work\poker\src')
import equity_evidence_features as eq
ROOT=r'C:\Data\Projects\Money Making\work\poker'
# eval top20 keys from batch 000
batches=sorted(pathlib.Path(ROOT+r'\Claude Version\outputs_v1_view_averaged_exploratory_20260913\batches').glob('*-top20-scores.parquet'))
keys_eval=pl.read_parquet(str(batches[0])).select('pair_id','hand_id').head(20)
print('eval keys sample', keys_eval.height, flush=True)
players=pl.read_csv(ROOT+r'\data\evaluation_pairs.csv', columns=['pair_id','player_1','player_2'])
keys_eval=keys_eval.join(players, on='pair_id', validate='m:1').sort('pair_id','hand_id')
print(keys_eval.head(2).to_dicts(), flush=True)
hand_ids=keys_eval['hand_id'].unique().to_list()
hands=(pl.scan_parquet(ROOT+r'\data\hands.parquet').filter(pl.col('hand_id').is_in(hand_ids)).select('hand_id','big_blind','board_cards').collect())
seats=(pl.scan_parquet(ROOT+r'\data\seats.parquet').filter(pl.col('hand_id').is_in(hand_ids)).select('hand_id','player_id','hole_card_1','hole_card_2').collect())
actions=(pl.scan_parquet(ROOT+r'\data\actions.parquet').filter(pl.col('hand_id').is_in(hand_ids)).select('hand_id','action_no','street','player_id','action','amount','to_call','pot_before','stack_before').sort('hand_id','action_no').collect())
print(f'hands {hands.height} seats {seats.height} actions {actions.height}', flush=True)
hand_lookup={h:(b,[eq.CARDS[c] for c in (board or '').split()]) for h,b,board in hands.iter_rows()}
seat_lookup={}
for h,p,c1,c2 in seats.iter_rows():
    seat_lookup.setdefault(h,{})[p]=(eq.CARDS[c1],eq.CARDS[c2])
action_lookup={}
for r in actions.iter_rows(named=True):
    hh=r.pop('hand_id')
    action_lookup.setdefault(hh,[]).append(r)
def sample_indices(count):
    return np.argsort(np.random.default_rng(408331).random((count,48)), axis=1)[:,:5]
# verify default 256
assert eq.PREFLOP_SAMPLES==256
assert np.array_equal(eq._SAMPLE_INDICES, sample_indices(256))
print('256 generator verified', flush=True)
# time 256 20 rows
eq._canonical_equity.cache_clear()
t0=time.time()
for pid,hid,f,s in keys_eval.select('pair_id','hand_id','player_1','player_2').iter_rows():
    bl,bd=hand_lookup[hid]
    v=eq.extract_pair_hand(f,s,bl,seat_lookup[hid],bd,action_lookup.get(hid,[]))
print('256 20 eval rows', round(time.time()-t0,2), 's cache', eq._canonical_equity.cache_info(), flush=True)
# configure 4096
idx4096=sample_indices(4096)
assert np.array_equal(idx4096[:256], sample_indices(256))
print('4096 prefix verified', flush=True)
eq.PREFLOP_SAMPLES=4096
eq._SAMPLE_INDICES=idx4096
eq._canonical_equity.cache_clear()
t0=time.time()
for pid,hid,f,s in keys_eval.select('pair_id','hand_id','player_1','player_2').iter_rows():
    bl,bd=hand_lookup[hid]
    v=eq.extract_pair_hand(f,s,bl,seat_lookup[hid],bd,action_lookup.get(hid,[]))
print('4096 20 eval rows', round(time.time()-t0,2), 's cache', eq._canonical_equity.cache_info(), flush=True)
# postflop exact check: reset to 256 and compare one state? simplified
print('done', flush=True)
