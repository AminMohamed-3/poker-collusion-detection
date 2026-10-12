"""Describe actual paid raise sizes after legal resolution; training-only evidence."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import json
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent
def describe(streets,fractions,log_amount):
    result={}
    for i in range(4):
        take=(streets==i)&np.isfinite(fractions);x=fractions[take];z=log_amount[take]
        result[str(i)]={'raises':len(x),'fraction_quantiles05_25_50_75_95':np.quantile(x,[.05,.25,.5,.75,.95]).tolist(),'fraction_mean':float(x.mean()),'fraction_std':float(x.std()),'log_amount_quantiles':np.quantile(z,[.05,.25,.5,.75,.95]).tolist(),'most_common_fraction_rounded_2dp':[(float(v),int(n)) for v,n in sorted(zip(*np.unique(x.round(2),return_counts=True)),key=lambda v:-v[1])[:8]]}
    return result
def main():
    assert (H/'sized/pilot-generation.json').exists();f=pl.read_parquet(H.parent/'softplay_fast_20260910/training-actions.parquet').filter((pl.col('behavior_family')=='none')&(pl.col('action_class')==3)).unique(subset=['hand_id','action_no'])
    pot=np.expm1(f['log_pot_bb'].to_numpy());price=f['price'].to_numpy();owed=pot*price/np.maximum(1e-8,1-price);amount=np.expm1(f['observed_log_amount_bb'].to_numpy());fraction=(amount-owed)/(pot+owed)
    result={'original_confirmed_none_actions':describe(f['street_index'].to_numpy(),fraction,f['observed_log_amount_bb'].to_numpy())}
    for name,d in [('control',H.parent/'softplay_capacity_20260910/b0_capacity'),('sized',H/'sized')]:
        frames=[]
        for p in sorted((d/'pilot').glob('pool-*')):
            exclude=set(json.loads((p/'provenance.json').read_text())['target']);hands=pl.read_parquet(p/'hands.parquet').select('hand_id','big_blind');a=pl.read_parquet(p/'actions.parquet').filter((pl.col('amount')>pl.col('to_call'))&~pl.col('player_id').is_in(sorted(exclude))).join(hands,on='hand_id');frames.append(a)
        a=pl.concat(frames);street=np.array([['preflop','flop','turn','river'].index(s) for s in a['street']]);fraction=(a['amount'].to_numpy()-a['to_call'].to_numpy())/(a['pot_before'].to_numpy()+a['to_call'].to_numpy());z=np.log1p(a['amount'].to_numpy()/a['big_blind'].to_numpy());result[name]=describe(street,fraction,z)
    out={'training_only':True,'purpose':'Descriptive realized raise sizes, no gate or model selection from outer validation.','limitation':'Original confirmed-none pair actions deduplicated; all generated declared target players excluded. Row populations are not exactly matched.','results':result};(H/'realized-size-diagnostic.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
