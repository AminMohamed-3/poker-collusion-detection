"""Verify selected-view trace coverage and distinct fresh deal payloads."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import json,sys,hashlib
import polars as pl
H=Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R/'src'))
from robust_pair_experiment import thin,TRAIN_VIEWS
def main():
    plan=json.loads((H/'study-plan.json').read_text());freeze=json.loads((H/'trial/freeze.json').read_text());out={};deals=set();duplicates=0;hands=0
    for batch in plan['batches']:
        chosen=freeze['selected_target_pairs'][str(batch['batch'])];ids=pl.DataFrame({'pair_id':chosen})
        for mode in batch['arms']:
            d=H/mode
            raw=pl.read_parquet(sorted((d/'features/advanced/hands_development').glob('batch_*.parquet'))).select('pair_id','hand_id','table_id').join(ids,on='pair_id',how='semi')
            ev=pl.concat([pl.read_csv(p/'development_evidence.csv') for p in sorted((d/'pilot').glob('pool-*'))]).select('pair_id','hand_id').unique();rows=[]
            for tag,key in [('full',None),('70',TRAIN_VIEWS[0]),('40',TRAIN_VIEWS[1])]:
                frame=raw if key is None else thin(raw,*key)
                counts=frame.join(ev,on=['pair_id','hand_id'],how='semi').group_by('pair_id').len();counts=ids.join(counts,on='pair_id',how='left').fill_null(0)
                rows.append({'view':tag,'pairs':counts.height,'zero_visible_intervention_pairs':counts.filter(pl.col('len')==0)['pair_id'].to_list(),'median_intervention_hands':float(counts['len'].median()),'minimum_intervention_hands':int(counts['len'].min())})
            out[mode]=rows
        # Count one counterfactual arm per scenario only; exclude IDs and learned outcomes.
        d=H/batch['arms'][0]
        for p in sorted((d/'pilot').glob('pool-*')):
            seats=pl.read_parquet(p/'seats.parquet').sort('hand_id','seat_no').group_by('hand_id',maintain_order=True).agg(pl.struct('seat_no','hole_card_1','hole_card_2').alias('cards'))
            for row in seats.iter_rows(named=True):
                key=hashlib.sha256(json.dumps(row['cards'],sort_keys=True).encode()).digest()
                duplicates+=key in deals;deals.add(key);hands+=1
    result={'created_at':datetime.now(timezone.utc).isoformat(),'generated_training_views_only':True,'no_fit_or_selection_changes':True,'history_coverage':out,'dealt_scenarios_checked':hands,'unique_deal_payloads_excluding_identifiers':len(deals),'duplicate_deal_payloads':duplicates,'expected_counterfactual_replays_per_scenario':2,'limitation':'Distinct deals do not imply independent hands within player histories or independent learned policy sources. Original evidence lists are capped and cannot be compared to exhaustive synthetic traces for missing annotations.'}
    assert hands==plan['independent_exogenous_scenarios'] and duplicates==0
    (H/'supplementary-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
