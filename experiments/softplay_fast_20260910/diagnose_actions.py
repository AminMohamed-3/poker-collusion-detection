"""Inspect training-only action contexts; no validation labels are read."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
from collections import defaultdict
from pathlib import Path
import json,sys,hashlib
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent; R=H.parents[1]; OLD=H.parent/'history_realism_20260910'
sys.path.insert(0,str(R/'src'))
from conditional_inference_worker import extract_chunk
from conditional_action_features import action_class

def summary(f):
    contexts={'all':pl.lit(True),'facing_partner':pl.col('facing_partner')==1,
      'facing_partner_cheap':(pl.col('facing_partner')==1)&(pl.col('price')<=.25),
      'heads_up':pl.col('players_active')==2,
      'multiway':pl.col('players_active')>2,
      'free':pl.col('price')==0,'owed':pl.col('price')>0,
      'preflop':pl.col('street_index')==0,'postflop':pl.col('street_index')>0}
    out={}
    for name,condition in contexts.items():
        g=f.filter(condition)
        if not g.height:continue
        counts=g.group_by('pair_id','action_class').len().pivot(on='action_class',index='pair_id',values='len').fill_null(0)
        for c in range(4):
            if str(c) not in counts.columns: counts=counts.with_columns(pl.lit(0).alias(str(c)))
        a=counts.select([str(i) for i in range(4)]).to_numpy();a=a/a.sum(1,keepdims=True)
        out[name]={'actions':g.height,'pairs':counts.height,'equal_pair_mean_fold_check_call_raise':a.mean(0).tolist(),
          'median_price':float(g['price'].median()),'median_strength':float(g['own_strength'].median())}
    return out

def main():
    pools=json.loads((OLD/'persistent_sessions/training-calibration.json').read_text())['training_pools']
    labels=pl.read_csv(R/'data/development_labels.csv').select('pair_id','behavior_family')
    real=pl.read_parquet(sorted((R/'reports/conditional-action-experiment/action-cache').glob('*.parquet'))).filter(pl.col('table_id').is_in(pools)).join(labels,on='pair_id',validate='m:1')
    syn=[]
    for batch in ['persistent_sessions','session_confirmation']:
        d=OLD/batch; frames=[]
        for p in sorted((d/'pilot').glob('pool-*')):
            lab=pl.read_csv(p/'development_labels.csv').filter(pl.col('behavior_family')=='soft_play')
            if not lab.height:continue
            h=pl.read_parquet(p/'hands.parquet').with_columns(pl.col('started_at').cast(pl.String));s=pl.read_parquet(p/'seats.parquet');a=pl.read_parquet(p/'actions.parquet')
            counts=defaultdict(lambda:np.zeros(4,dtype=np.int64))
            for row in a.iter_rows(named=True):counts[row['player_id']][action_class(row)]+=1
            keys=s.select('hand_id','player_id').join(lab,left_on='player_id',right_on='player_1').join(s.select('hand_id',pl.col('player_id').alias('player_2')),on=['hand_id','player_2']).select('pair_id','hand_id','player_id','player_2').rename({'player_id':'player_1'})
            frames.append(extract_chunk(keys,h,s,a,counts))
        syn.append(pl.concat(frames))
    synthetic=pl.concat(syn);synthetic.write_parquet(H/'old-soft-actions.parquet')
    real.write_parquet(H/'training-actions.parquet')
    result={name:summary(real.filter(pl.col('behavior_family')==name)) for name in ['none','soft_play']}
    result['old_synthetic_soft_play']=summary(synthetic)
    ev=pl.read_csv(R/'data/development_evidence.csv').select('pair_id','hand_id').unique()
    result['soft_play_evidence_hands']=summary(real.filter(pl.col('behavior_family')=='soft_play').join(ev,on=['pair_id','hand_id'],how='semi'))
    out={'training_pools':len(pools),'training_only':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':result}
    (H/'action-diagnosis.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
