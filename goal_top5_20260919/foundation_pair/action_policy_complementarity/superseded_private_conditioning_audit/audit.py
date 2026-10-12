import site,json,hashlib
from pathlib import Path
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import polars as pl,numpy as np
from scipy.stats import rankdata,spearmanr
R=Path(r'C:/Data/Projects/Money Making/work/poker'); L=R/'goal_top5_20260919/foundation_pair/private_card_likelihood'; I=R/'goal_top5_20260919/foundation_pair/private_card_integration'; O=R/'goal_top5_20260919/foundation_pair/action_policy_complementarity'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ps=[];acts=[]
for f in (0,1):
 p=pl.read_parquet(I/f'pair_predictions_outer{f}.parquet');a=pl.read_parquet(L/f'predictions_outer{f}.parquet');assert p['outer_fold'].unique().to_list()==[f] and a['outer_fold'].unique().to_list()==[f]
 pt=p.select('pair_id','table_id').unique();at=a.select('pair_id','table_id').unique();assert pt.height==p.height and at.height==p.height
 tj=pt.join(at,on='pair_id',suffix='_action',validate='1:1');assert tj.height==p.height and (tj['table_id']==tj['table_id_action']).all()
 q0=np.column_stack([a[f'p0_{k}'] for k in range(4)]);q1=np.column_stack([a[f'p1_{k}'] for k in range(4)]);y=a['action_class'].to_numpy();g=np.log(np.maximum(q1[np.arange(len(y)),y],1e-12))-np.log(np.maximum(q0[np.arange(len(y)),y],1e-12));z=a.select('pair_id','direction').with_columns(pl.Series('g',g)).group_by('pair_id','direction').agg(pl.mean('g')).group_by('pair_id').agg(pl.mean('g').alias('derived'),pl.col('direction').n_unique().alias('dirs'));assert z['dirs'].min()==2
 j=p.join(z,on='pair_id',validate='1:1');assert np.allclose(j['priv_ll_mean'],j['derived'],rtol=0,atol=1e-12);assert set(p['pair_id'])==set(a['pair_id']);ps.append(p);acts.append(a)
d=pl.concat(ps).join(pl.read_csv(R/'data/development_labels.csv').select('pair_id','label','behavior_family'),on='pair_id',validate='1:1').sort('pair_id');assert d.height==1860 and d['pair_id'].n_unique()==1860
base=1-np.asarray(d['baseline_0'],float);llr=np.asarray(d['priv_ll_mean'],float);y=np.asarray(d['label'],bool);fam=np.asarray(d['behavior_family']);n=len(y);rb=rankdata(base,method='average')/n;rl=rankdata(llr,method='average')/n;delta=rl-rb;des=np.where(y,delta,-delta);K=int(y.sum());topb=np.argsort(-base,kind='stable')[:K];topl=np.argsort(-llr,kind='stable')[:K];fp=np.setdiff1d(topb,np.flatnonzero(y),assume_unique=False);fn=np.setdiff1d(np.flatnonzero(y),topb,assume_unique=False)
def sm(x):return {'n':int(len(x)),'mean':float(np.mean(x)) if len(x) else None,'median':float(np.median(x)) if len(x) else None,'positive_fraction':float(np.mean(x>0)) if len(x) else None}
res={'scope':'reused held-1860 descriptive complementarity; no blend/gate/fit','alignment':{'rows':n,'positive':K,'fold_rows':{str(f):int((d['outer_fold']==f).sum()) for f in (0,1)},'exact_action_derived_llr_matches_integration':True,'pair_fold_and_table_mappings_exact':True},'rank_dependence':{'spearman_all':float(spearmanr(base,llr).statistic),'spearman_normal':float(spearmanr(base[~y],llr[~y]).statistic),'spearman_positive':float(spearmanr(base[y],llr[y]).statistic)},'desirable_rank_movement':{'all':sm(des),'normal':sm(-delta[~y]),'positive':sm(delta[y])},'baseline_topK':{'K':K,'true_positives':int(y[topb].sum()),'false_positives':int((~y[topb]).sum()),'llr_topK_true_positives':int(y[topl].sum()),'llr_topK_false_positives':int((~y[topl]).sum()),'baseline_false_positives_llr_movement':sm(-delta[fp]),'baseline_false_negatives_llr_movement':sm(delta[fn]),'false_positives_removed_by_llr_topK':int(len(set(fp)-set(topl))),'false_negatives_recovered_by_llr_topK':int(len(set(fn)&set(topl)))},'families':{},'hashes':{}}
for f in ('directed_transfer','soft_play','coordinated_isolation'):
 m=fam==f;res['families'][f]={'n':int(m.sum()),'llr_minus_baseline_rank':sm(delta[m]),'within_family_spearman':float(spearmanr(base[m],llr[m]).statistic)}
for p in [I/'pair_predictions_outer0.parquet',I/'pair_predictions_outer1.parquet',L/'predictions_outer0.parquet',L/'predictions_outer1.parquet',R/'data/development_labels.csv',I/'RUN_RECEIPT.json',L/'FIT_RECEIPT.json']:res['hashes'][str(p)]=sha(p)
(O/'audit.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
