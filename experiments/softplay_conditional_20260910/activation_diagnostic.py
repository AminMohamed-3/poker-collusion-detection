"""Post-screen generated-data description; no fitting or held labels."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import json,hashlib
import polars as pl
H=Path(__file__).resolve().parent
def main():
    result={}
    for mode in ['baseline','strength','learned']:
        actions=pl.read_parquet(H/mode/'soft-actions.parquet');rows=[]
        for p in sorted((H/mode/'pilot').glob('pool-*')):
            pr=json.loads((p/'provenance.json').read_text());pid=f"SSP{pr['spec']['pool_id']:04d}PAIR0";f=actions.filter(pl.col('pair_id')==pid)
            if not f.height:continue
            f=f.with_columns((pl.col('hand_id').str.split('H').list.last().cast(pl.Int64)/3000).alias('_fraction'))
            f=f.with_columns(((pl.col('_fraction')>=pr['onset'])&(pl.col('_fraction')<pr['onset']+pr['duration'])).alias('active_episode'))
            for active in [False,True]:
                sub=f.filter(pl.col('active_episode')==active);pre=sub.filter(pl.col('street_index')==0)
                rows.append({'pair_id':pid,'active':active,'actions':sub.height,'preflop_actions':pre.height,'preflop_call_rate':float((pre['action_class']==2).mean()) if pre.height else None,'intensity':pr['intensity']})
        df=pl.DataFrame(rows);groups=[]
        for active in [False,True]:
            f=df.filter(pl.col('active')==active);groups.append({'active_episode':active,'pairs':f.height,'preflop_call_rate_equal_pair':float(f['preflop_call_rate'].mean()),'preflop_actions':int(f['preflop_actions'].sum()),'actions':int(f['actions'].sum())})
        result[mode]={'groups':groups,'mean_intervention_gate_probability':float(df['intensity'].mean())}
    out={'post_screen_descriptive_only':True,'no_fits_or_validation_labels':True,'results':result,'interpretation':'Learned soft action probabilities already average normal and coordinated play; an additional episode/probability/allowed-transition gate may attenuate behavior. Conditional state distributions also differ, so this does not identify gating as the sole cause. Do not tune gating on held results.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (H/'activation-diagnostic.json').write_text(json.dumps(out,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
