import os,sys,json
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[k]='2'
R=Path(r'C:/Data/Projects/Money Making/work/poker');CV=R/'Claude Version';sys.path.insert(0,str(CV/'src'))
import numpy as np,polars as pl
from sklearn.metrics import average_precision_score
from pu_pairs import step159c_C_combine_dev as S
E=CV/'experiments';S.DIRS['ARCH2']=(str(E/'arch2_views_20260915')+'/', 'arch2_dev_fold')
V=S.type_means('SXN3',S.VT).join(S.type_means('ARCH',S.NEW_VT),on=['pair_id','fold','surface'],how='inner',validate='1:1').join(S.type_means('ARCH2',['MLP2','HGB5']),on=['pair_id','fold','surface'],how='inner',validate='1:1')
cols=[f'SXN3_{v}' for v in S.VT]+[f'ARCH_{v}' for v in S.NEW_VT]+['ARCH2_MLP2','ARCH2_HGB5'];V=V.filter(pl.col('surface')=='prefix60').with_columns(pl.mean_horizontal(cols).alias('k_FINAL2')).select('pair_id','fold','k_FINAL2')
parts=[]
for f in (1,2):parts.append(pl.read_parquet(E/f'pu_pairs_20260911/fold{f}-prefix60/scores.parquet').filter(pl.col('shared_hands')>=38).with_columns(pl.lit(f).alias('fold')))
d=pl.concat(parts).join(V,on=['pair_id','fold'],how='left',validate='1:1').unique('pair_id',keep='first',maintain_order=True);assert d['k_FINAL2'].null_count()==0
risk=d['risk'].to_numpy();key=d['k_FINAL2'].to_numpy();out=S.permute(risk,key,risk<.5)
lab=pl.read_csv(R/'data/development_labels.csv',columns=['pair_id','label']);j=d.select('pair_id').join(lab,on='pair_id',how='left');y=(j['label'].fill_null(-1)==1).to_numpy();m={'rows':d.height,'key_columns':cols,'known_positives':int(y.sum()),'v11_prefix_risk_ap':float(average_precision_score(y,risk)),'v11_plus_FINAL2_tail_ap':float(average_precision_score(y,out)),'delta':float(average_precision_score(y,out)-average_precision_score(y,risk)),'tail_rows':int((risk<.5).sum()),'lineage':'STEP163 FINAL2_16 dev key; ten saved learner means; permutes only v11 prefix risk<.5'}
O=R/'goal_top5_20260919/foundation_pair/global_pu';np.savez_compressed(O/'final2_baseline.npz',pair_id=d['pair_id'].to_numpy().astype('U'),table_id=d['table_id'].to_numpy().astype('U'),risk_v11=risk,risk_v11_final2=out,k_FINAL2=key);(O/'final2_receipt.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
