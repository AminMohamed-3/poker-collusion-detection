import os
for _k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='2'
os.environ['PYTHONUTF8']='1'
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode=True
import time
import polars as pl
sys.path.insert(0, r'C:\Data\Projects\Money Making\work\poker\src')
import equity_evidence_features as eq
print('samples', eq.PREFLOP_SAMPLES, flush=True)
keys=pl.read_parquet(r'C:\Data\Projects\Money Making\work\poker\Codex Resume\research\preflop_equity_precision_20260912\inputs-id-only.parquet').head(3)
print('keys', keys.height, flush=True)
hand_ids=keys['hand_id'].unique().to_list()
hands=(pl.scan_parquet(r'C:\Data\Projects\Money Making\work\poker\data\hands.parquet').filter(pl.col('hand_id').is_in(hand_ids)).select('hand_id','big_blind','board_cards').collect())
print('hands', hands.height, flush=True)
seats=(pl.scan_parquet(r'C:\Data\Projects\Money Making\work\poker\data\seats.parquet').filter(pl.col('hand_id').is_in(hand_ids)).select('hand_id','player_id','hole_card_1','hole_card_2').collect())
print('seats', seats.height, flush=True)
actions=(pl.scan_parquet(r'C:\Data\Projects\Money Making\work\poker\data\actions.parquet').filter(pl.col('hand_id').is_in(hand_ids)).select('hand_id','action_no','street','player_id','action','amount','to_call','pot_before','stack_before').sort('hand_id','action_no').collect())
print('actions', actions.height, flush=True)
hand_lookup={h:(b,[eq.CARDS[c] for c in (board or '').split()]) for h,b,board in hands.iter_rows()}
seat_lookup={}
for h,p,c1,c2 in seats.iter_rows():
    seat_lookup.setdefault(h,{})[p]=(eq.CARDS[c1],eq.CARDS[c2])
action_lookup={}
for r in actions.iter_rows(named=True):
    hh=r.pop('hand_id')
    action_lookup.setdefault(hh,[]).append(r)
t0=time.time()
for pid,hid,f,s in keys.iter_rows():
    bl,bd=hand_lookup[hid]
    v=eq.extract_pair_hand(f,s,bl,seat_lookup[hid],bd,action_lookup.get(hid,[]))
    pass
print('256 3 rows', round(time.time()-t0,2), 's', flush=True)
print('cache', eq._canonical_equity.cache_info(), flush=True)
