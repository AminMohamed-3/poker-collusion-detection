"""Count observable intervention hands in each synthetic training view."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import json,sys
import polars as pl
H=Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R/'src'))
from robust_pair_experiment import thin,TRAIN_VIEWS
def main():
    chosen=json.loads((H/'trial/freeze.json').read_text())['common_target_pairs'];ids=pl.DataFrame({'pair_id':chosen});out={}
    for mode in ['gated','calibrated','continuous']:
        d=H/mode;raw=pl.read_parquet(sorted((d/'features/advanced/hands_development').glob('batch_*.parquet'))).select('pair_id','hand_id','table_id').join(ids,on='pair_id',how='semi');ev=pl.concat([pl.read_csv(p/'development_evidence.csv') for p in sorted((d/'pilot').glob('pool-*'))]).select('pair_id','hand_id').unique();rows=[]
        for tag,key in [('full',None),('70',TRAIN_VIEWS[0]),('40',TRAIN_VIEWS[1])]:
            f=raw if key is None else thin(raw,*key);counts=f.join(ev,on=['pair_id','hand_id'],how='semi').group_by('pair_id').len();counts=ids.join(counts,on='pair_id',how='left').fill_null(0)
            rows.append({'view':tag,'pairs':counts.height,'zero_visible_intervention_pairs':counts.filter(pl.col('len')==0)['pair_id'].to_list(),'median_intervention_hands':float(counts['len'].median()),'minimum_intervention_hands':int(counts['len'].min())})
        out[mode]=rows
    (H/'history-coverage.json').write_text(json.dumps({'generated_training_views_only':True,'no_fit_or_selection_changes':True,'results':out,'limitation':'Original evidence lists are capped, so their missing annotations cannot be compared directly with exhaustive synthetic intervention traces.'},indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
