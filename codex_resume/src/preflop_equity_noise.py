"""Label-free preflop equity sampling/equivariance measurement; no model fits."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import polars as pl
HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
import equity_evidence_features as eq
OUT=HERE/'research'/'preflop_equity_noise_20260912'
OUT.mkdir(parents=True,exist_ok=True)
keys=pl.read_parquet(HERE/'research/contestable_pot_feasibility_20260912/shortlist-state.parquet').select('pair_id','hand_id').sort('pair_id','hand_id')
ix=np.random.default_rng(409615).choice(keys.height,32,replace=False)
keys=keys[ix].sort('pair_id','hand_id')
pairs=pl.read_csv(ROOT/'data/development_labels.csv',columns=['pair_id','player_1','player_2'])
keys=keys.join(pairs,on='pair_id',validate='m:1').sort('pair_id','hand_id')
seat=pl.scan_parquet(ROOT/'data/seats.parquet').filter(pl.col('hand_id').is_in(keys['hand_id'].unique().implode())).select('hand_id','player_id','hole_card_1','hole_card_2').collect()
lookup={(r['hand_id'],r['player_id']):(r['hole_card_1'],r['hole_card_2']) for r in seat.iter_rows(named=True)}
rows=[]
perms=['cdhs','dchs','dhsc','shdc']
old_indices=eq._SAMPLE_INDICES.copy(); old_n=eq.PREFLOP_SAMPLES
start=time.perf_counter()
for pid,hid,a,b in keys.select('pair_id','hand_id','player_1','player_2').iter_rows():
    cardsA=lookup[(hid,a)]; cardsB=lookup[(hid,b)]
    vals=[]
    for perm in perms:
        smap=dict(zip('cdhs',perm)); conv=lambda cards:tuple(eq.CARDS[c[0]+smap[c[1]]] for c in cards)
        vals.append(eq.heads_up_equity(conv(cardsA),conv(cardsB),()))
    rows.append({'pair_id':pid,'hand_id':hid,'q256_identity':vals[0],'q256_suits':vals,'cardsA':cardsA,'cardsB':cardsB})
eq.PREFLOP_SAMPLES=4096
eq._SAMPLE_INDICES=np.argsort(np.random.default_rng(eq.SAMPLE_SEED).random((4096,48)),axis=1)[:,:5]
eq._canonical_equity.cache_clear()
for row in rows:
    a=tuple(eq.CARDS[c] for c in row['cardsA']); b=tuple(eq.CARDS[c] for c in row['cardsB'])
    row['q4096']=eq.heads_up_equity(a,b,())
    assert eq.heads_up_equity(b,a,())==1-row['q4096']
    del row['cardsA'];del row['cardsB']
error=np.array([r['q256_identity']-r['q4096'] for r in rows]); spreads=np.array([max(r['q256_suits'])-min(r['q256_suits']) for r in rows])
eq._SAMPLE_INDICES=old_indices;eq.PREFLOP_SAMPLES=old_n;eq._canonical_equity.cache_clear()
result={'label_free':True,'samples':len(rows),'seed':409615,'suit_permutations':perms,'q256_vs_q4096':{'mean_abs':float(abs(error).mean()),'rms':float(np.sqrt(np.mean(error**2))),'max_abs':float(abs(error).max())},'q256_suit_spread':{'mean':float(spreads.mean()),'max':float(spreads.max())},'seconds':time.perf_counter()-start,'limitations':['4096runouts is a lower-noise approximation, not exact preflop truth.','No relevance labels, outer truth, public outcome or model fit used.','This quantifies feature noise only, not predictive benefit.'],'source_sha256':hashlib.sha256((ROOT/'src/equity_evidence_features.py').read_bytes()).hexdigest(),'rows':rows}
(OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
