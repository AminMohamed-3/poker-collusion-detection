import json,hashlib
from pathlib import Path
import numpy as np,polars as pl
from sklearn.metrics import average_precision_score
R=Path(r'C:/Data/Projects/Money Making/work/poker'); E=R/'Claude Version/experiments/pu_pairs_20260911'; O=R/'goal_top5_20260919/foundation_pair/global_pu'
lab=pl.read_csv(R/'data/development_labels.csv').select('pair_id','player_1','player_2','label','behavior_family')
parts=[]; steps=[]
for f,fn in [(1,'step41_train2_test1_prefix60-scores.parquet'),(2,'step41_train1_test2_prefix60-scores.parquet')]:
 s=pl.read_parquet(E/f'fold{f}-prefix60/scores.parquet').filter(pl.col('shared_hands')>=38).with_columns(pl.lit(f).alias('source_fold'))
 q=pl.read_parquet(E/fn).with_columns(pl.lit(f).alias('source_fold'))
 assert s.height==q.height and s.select('pair_id','table_id').equals(q.select('pair_id','table_id'))
 parts.append(s);steps.append(q)
a=pl.concat(parts).sort('source_fold','pair_id');q=pl.concat(steps).sort('source_fold','pair_id')
# canonical union: fold1 wins for shared held tables, matching historical dev readers.
u=a.unique('pair_id',keep='first',maintain_order=True);uq=q.unique('pair_id',keep='first',maintain_order=True);assert u.select('pair_id','table_id').equals(uq.select('pair_id','table_id'))
assert u['pair_id'].n_unique()==u.height
j=u.join(lab,on='pair_id',how='left',suffix='_public',validate='1:1')
known=j['label_public'].is_not_null(); pos=(j['label_public'].fill_null(-1)==1).to_numpy(); neg=(j['label_public'].fill_null(-1)==0).to_numpy(); unl=~known.to_numpy()
# saved score comparison, treating only public positives as identified positives; every other row remains background/unknown.
scorecols={'fold_prefix_v11_risk':u['risk'].to_numpy()}
for c in ['risk_full','risk_first','labelled_only','pu_negonly','pu_both','pu_negonly_stacked']:scorecols['step41_'+c]=uq[c].to_numpy()
metrics={k:{'known_positive_retrieval_ap_all_unknown_background':float(average_precision_score(pos,v)),'known_positive_median_percentile':float(np.median(pl.Series(v).rank('average').to_numpy()[pos])/len(v)) if pos.any() else None} for k,v in scorecols.items()}
# duplicate/source facts
D=a.group_by('pair_id').len().filter(pl.col('len')>1); aj=parts[0].join(parts[1],on='pair_id',suffix='_2'); same_table=int((aj['table_id']==aj['table_id_2']).sum())
feat_schema=pl.read_parquet(R/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet',n_rows=1).schema
feat=[c for c,dt in feat_schema.items() if c not in {'pair_id','table_id'} and dt.is_numeric()]
# verify raw used features duplicate-exact
b1=pl.concat([pl.read_parquet(x,columns=['pair_id','table_id']+feat) for x in sorted((E/'fold1-prefix60').glob('batch_*-pairs.parquet'))]).filter(pl.col('pair_id').is_in(aj['pair_id']))
b2=pl.concat([pl.read_parquet(x,columns=['pair_id','table_id']+feat) for x in sorted((E/'fold2-prefix60').glob('batch_*-pairs.parquet'))]).filter(pl.col('pair_id').is_in(aj['pair_id']))
bj=b1.join(b2,on='pair_id',suffix='_2');diff=0
for c in feat:
 x,y=bj[c].to_numpy(),bj[c+'_2'].to_numpy();diff+=int((~np.isclose(x,y,equal_nan=True)).sum())
rec={'canonical_population':{'rows':u.height,'tables':u['table_id'].n_unique(),'source_fold1_rows':int((u['source_fold']==1).sum()),'source_fold2_rows':int((u['source_fold']==2).sum()),'known_labels':int(known.sum()),'known_positives':int(pos.sum()),'known_negatives':int(neg.sum()),'unlabelled':int(unl.sum()),'positive_families':j.filter(pl.col('label_public')==1)['behavior_family_public'].value_counts().to_dicts()},'source_rows_before_dedup':a.height,'duplicate_pair_ids':D.height,'duplicates_same_table':same_table,'raw_feature_count':len(feat),'duplicate_used_feature_cell_differences':diff,'metrics_are':'PU known-positive retrieval diagnostics only; unlisted pairs are unknown, never real negatives','saved_baselines':metrics,'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [E/'fold1-prefix60/scores.parquet',E/'fold2-prefix60/scores.parquet',E/'step41_train2_test1_prefix60-scores.parquet',E/'step41_train1_test2_prefix60-scores.parquet',R/'data/development_labels.csv']}}
(O/'availability_receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
