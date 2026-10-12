"""Training-only interpretation of observed style distributions after generation."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import polars as pl
import json
H=Path(__file__).resolve().parent
def main():
    frames={'original_soft':pl.read_parquet(H/'training-actions-causal.parquet').filter(pl.col('behavior_family')=='soft_play')}
    for arm in ['b0_style','b1_style']:frames[arm]=pl.read_parquet(H/arm/'soft-actions.parquet')
    cols=['causal_own_style_'+s for s in ['fold','check','call','raise']];result={}
    for name,f in frames.items():
        p=f.group_by('pair_id').agg(*[pl.col(c).mean() for c in cols]);result[name]={'pairs':p.height,'equal_pair_mean_own_past_style':{c:float(p[c].mean()) for c in cols},'across_pair_std_own_past_style':{c:float(p[c].std()) for c in cols}}
    (H/'tendency-diagnostic.json').write_text(json.dumps({'training_only':True,'results':result,'not_a_fitted_or_tested_repair':True},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
