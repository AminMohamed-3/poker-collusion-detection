import site,json,hashlib
from pathlib import Path
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import polars as pl,numpy as np
from scipy.stats import rankdata,spearmanr
R=Path(r'C:/Data/Projects/Money Making/work/poker'); L=R/'goal_top5_20260919/foundation_pair/action_policy_llr'; I=R/'goal_top5_20260919/foundation_pair/private_card_integration'; O=R/'goal_top5_20260919/foundation_pair/action_policy_complementarity'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ps=[]
for f in (0,1):
 p=pl.read_parquet(I/f'pair_predictions_outer{f}.parquet'); q=pl.read_parquet(L/f'predictions_outer{f}.parquet'); a=pl.read_parquet(L/f'action_predictions_outer{f}.parquet')
 assert p['outer_fold'].unique().to_list()==[f] and p.height==q.height
 for x in (p,q,a): assert x['pair_id'].n_unique()==p.height
 pt=p.select('pair_id','table_id').unique(); qt=q.select('pair_id','table_id').unique(); at=a.select('pair_id','table_id').unique()
 assert pt.height==qt.height==at.height==p.height
 j=pt.join(qt,on='pair_id',suffix='_q',validate='1:1').join(at,on='pair_id',suffix='_a',validate='1:1')
 assert j.height==p.height and (j['table_id']==j['table_id_q']).all() and (j['table_id']==j['table_id_a']).all()
 y=a['action_class'].to_numpy(); ix=np.arange(a.height)
 rt=np.column_stack([a[f'real_target_p{k}'] for k in range(4)]); rn=np.column_stack([a[f'real_normal_p{k}'] for k in range(4)])
 g=np.log(np.maximum(rt[ix,y],1e-12))-np.log(np.maximum(rn[ix,y],1e-12))
 z=a.select('pair_id','direction').with_columns(pl.Series('g',g)).group_by('pair_id','direction').agg(pl.mean('g')).group_by('pair_id').agg(pl.mean('g').alias('derived'),pl.col('direction').n_unique().alias('dirs'))
 assert z['dirs'].min()==2
 jq=q.join(z,on='pair_id',validate='1:1');assert np.allclose(jq['real_llr'],jq['derived'],rtol=0,atol=1e-12)
 ps.append(p.join(q.select('pair_id','real_llr'),on='pair_id',validate='1:1'))
d=pl.concat(ps).join(pl.read_csv(R/'data/development_labels.csv').select('pair_id','label','behavior_family'),on='pair_id',validate='1:1').sort('pair_id'); assert d.height==1860 and d['pair_id'].n_unique()==1860
base=1-np.asarray(d['baseline_0'],float); llr=np.asarray(d['real_llr'],float); y=np.asarray(d['label'],bool); fam=np.asarray(d['behavior_family']); n=len(y)
rb=rankdata(base,method='average')/n; rl=rankdata(llr,method='average')/n; delta=rl-rb; des=np.where(y,delta,-delta); K=int(y.sum()); topb=np.argsort(-base,kind='stable')[:K]; topl=np.argsort(-llr,kind='stable')[:K]; fp=np.setdiff1d(topb,np.flatnonzero(y)); fn=np.setdiff1d(np.flatnonzero(y),topb)
def sm(x):return {'n':int(len(x)),'mean':float(np.mean(x)) if len(x) else None,'median':float(np.median(x)) if len(x) else None,'positive_fraction':float(np.mean(x>0)) if len(x) else None}
res={'scope':'action_policy_llr real target-vs-normal LLR; reused held-1860 descriptive complementarity; no blend/gate/fit','alignment':{'rows':n,'positive':K,'fold_rows':{str(f):int((d['outer_fold']==f).sum()) for f in (0,1)},'exact_action_derived_real_llr_matches_pair_predictions':True,'pair_fold_and_table_mappings_exact':True},'rank_dependence':{'spearman_all':float(spearmanr(base,llr).statistic),'spearman_normal':float(spearmanr(base[~y],llr[~y]).statistic),'spearman_positive':float(spearmanr(base[y],llr[y]).statistic)},'desirable_rank_movement':{'all':sm(des),'normal':sm(-delta[~y]),'positive':sm(delta[y])},'baseline_topK':{'K':K,'true_positives':int(y[topb].sum()),'false_positives':int((~y[topb]).sum()),'llr_topK_true_positives':int(y[topl].sum()),'llr_topK_false_positives':int((~y[topl]).sum()),'baseline_false_positives_llr_movement':sm(-delta[fp]),'baseline_false_negatives_llr_movement':sm(delta[fn]),'false_positives_removed_by_llr_topK':int(len(set(fp)-set(topl))),'false_negatives_recovered_by_llr_topK':int(len(set(fn)&set(topl)))},'families':{},'hashes':{}}
for family in ('directed_transfer','soft_play','coordinated_isolation'):
 m=fam==family; res['families'][family]={'n':int(m.sum()),'llr_minus_baseline_rank':sm(delta[m]),'within_family_spearman':float(spearmanr(base[m],llr[m]).statistic)}
for path in [I/'pair_predictions_outer0.parquet',I/'pair_predictions_outer1.parquet',L/'predictions_outer0.parquet',L/'predictions_outer1.parquet',L/'action_predictions_outer0.parquet',L/'action_predictions_outer1.parquet',R/'data/development_labels.csv',I/'RUN_RECEIPT.json',L/'RUN_RECEIPT.json',L/'RESULTS.json']:
 res['hashes'][str(path)]=sha(path)
(O/'audit.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res,indent=2))
