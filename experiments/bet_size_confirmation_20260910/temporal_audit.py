"""Training-only partner-facing temporal structure, without learned references."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent;E=H.parent;R=E.parent;N=E/'partner_temporal_audit_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(f,hands,name):
    eligible=f.filter((pl.col('facing_partner')==1)&(pl.col('price')>0));byhand=eligible.group_by('pair_id','table_id','hand_id').agg((pl.col('action_class')==0).mean().alias('fold_share'),((pl.col('action_class')==0)*pl.col('own_strength')).mean().alias('strength_fold'),(pl.col('action_class')!=3).mean().alias('passive_share'))
    time=pl.scan_parquet(hands).select('hand_id','started_at').filter(pl.col('hand_id').is_in(byhand['hand_id'].unique().to_list())).collect();byhand=byhand.join(time,on='hand_id').sort('pair_id','started_at','hand_id');rows=[]
    for key,part in byhand.group_by('pair_id',maintain_order=True):
        if part.height<16:continue
        row={'pair_id':key[0],'table_id':part['table_id'][0],'dataset':name,'eligible_hands':part.height}
        for col in ['fold_share','strength_fold','passive_share']:
            x=part[col].to_numpy();z=x-x.mean();den=float(z@z);row[col+'_mean']=float(x.mean());row[col+'_std']=float(x.std())
            for lag in [1,2,3]:
                # For a random permutation, E[sum adjacent z_i*z_j]/sum(z^2)
                # equals -(n-lag)/(n*(n-1)). Remove that finite-history bias.
                row[f'{col}_lag{lag}_excess']=float(z[:-lag]@z[lag:]/den+(len(z)-lag)/(len(z)*(len(z)-1))) if den>1e-12 else 0.
        rows.append(row)
    result=pl.DataFrame(rows);result.write_parquet(N/(name+'-pairs.parquet'));return result
def main():
    N.mkdir(exist_ok=False);plan={'at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'training_only':True,'eligible':'Original-training soft-play actions facing a partner with positive call price; average per pair/hand. Keep pairs with at least16 eligible hands. All supported synthetic targets included; no utility-based selection.','signals':['fold_share','strength_fold','passive_share'],'temporal':'Within-pair chronological lag1/2/3 autocovariance normalized by total centered energy, subtract analytic finite-history shuffled expectation. Reference-free behavioral metrics; no future-derived classifier residuals.','interpretation':'Exploratory mechanism diagnosis, not a new gate or proof of episodic coordination. Sparse opportunities, seating and game-state autocorrelation can affect these signals. No generator fit or outer score.'};(N/'plan.json').write_text(json.dumps(plan,indent=2))
    real=pl.read_parquet(E/'softplay_fast_20260910/training-actions.parquet').filter(pl.col('behavior_family')=='soft_play');allowed=set(json.loads((E/'data_expansion/training-calibration.json').read_text())['training_pools']);assert set(real['table_id'])<=allowed and len(allowed)==297
    frames={'original':summarize(real,R/'data/hands.parquet','original')};sources={'b0_sized':E/'bet_size_realism_20260910/sized','b1_sized':H/'b1_sized','fresh_sized':H/'fresh_sized'}
    for name,p in sources.items():frames[name]=summarize(pl.read_parquet(p/'soft-actions.parquet'),p/'features/data/hands.parquet',name)
    frames['pooled']=pl.concat([v for k,v in frames.items() if k!='original']);stats={}
    cols=[c for c in frames['original'].columns if c not in ['pair_id','table_id','dataset']]
    for name,frame in frames.items():stats[name]={'pairs':frame.height,'pools':frame['table_id'].n_unique(),'means':{c:float(frame[c].mean()) for c in cols},'medians':{c:float(frame[c].median()) for c in cols}}
    result={'at':datetime.now(timezone.utc).isoformat(),'training_only':True,'outer_validation_read':False,'results':stats,'plan_sha256':sha(N/'plan.json'),'not_a_new_gate':True};(N/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
