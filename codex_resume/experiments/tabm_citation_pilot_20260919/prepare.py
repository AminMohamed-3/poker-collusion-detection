import os
os.environ['POLARS_MAX_THREADS']='2'
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import polars as pl,numpy as np
from sklearn.preprocessing import QuantileTransformer
import joblib
OUT=Path(__file__).resolve().parent;CR=OUT.parent.parent;R=CR.parent;CV=R/'Claude Version'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((OUT/'protocol.json').read_text())
for p,h in plan['sources'].items():assert sha(Path(p))==h,p
assert not (OUT/'context.npz').exists()
K=['pair_id','hand_id'];SP='soft_play'
labels=pl.read_csv(R/'data/development_labels.csv').select('pair_id','player_1','player_2','label','behavior_family')
parent=pl.read_parquet(CR/'experiments/step277_composition_check_20260919/predictions.parquet')
audit_pairs=set(parent['pair_id']);train_pairs=labels.filter((pl.col('label')==1)&(pl.col('behavior_family')==SP)&~pl.col('pair_id').is_in(sorted(audit_pairs))).select('pair_id','player_1','player_2')
assert train_pairs.height==85
keys=parent.filter(pl.col('is_top20')&(pl.col('behavior_family')==SP)).select(K+['table_id']).sort(K)
assert keys.height==940 and keys['pair_id'].n_unique()==47
source=CV/'experiments/step254_mechanical_tensor_20260917/step254_features.parquet'
cols=[c for c in pl.read_parquet_schema(source) if c.startswith(('t1_','t2_','t3_','t4_','t5_'))];assert len(cols)==662
tr=pl.scan_parquet(source).join(train_pairs.select('pair_id').lazy(),on='pair_id',how='semi').collect().sort(K)
ae=keys.join(pl.scan_parquet(source).select(K+cols).collect(),on=K,how='left',validate='1:1',maintain_order='left');assert ae.select(K).equals(keys.select(K))
pairpools=pl.read_parquet(R/'outputs_v2/pairs_development.parquet',columns=['pair_id','table_id']);tr=tr.join(pairpools,on='pair_id',validate='m:1').sort(K)
trainpools=set(tr['table_id']);heldpools=set(parent['table_id']);assert len(trainpools)==71 and not trainpools&heldpools
heldplayers=labels.filter(pl.col('pair_id').is_in(sorted(audit_pairs)));assert not (set(train_pairs['player_1'])|set(train_pairs['player_2']))&(set(heldplayers['player_1'])|set(heldplayers['player_2']))
assert not set(tr['hand_id'])&set(parent['hand_id'])
# Only actual training labels enter the training arrays. No audit relevance enters this process.
ev=pl.scan_csv(R/'data/development_evidence.csv').join(train_pairs.select('pair_id').lazy(),on='pair_id',how='semi').select(K).unique().collect().with_columns(pl.lit(1.).alias('target'))
tr=tr.join(ev,on=K,how='left',validate='1:1').with_columns(pl.col('target').fill_null(0.))
assert tr['target'].sum()==ev.height and tr.height==10438
ordered=sorted(trainpools,key=lambda s:hashlib.sha256(('tabm-inner-v1:'+s).encode()).hexdigest());valp=set(ordered[:14]);im=tr['table_id'].is_in(sorted(valp)).to_numpy();assert im.any() and (~im).any()
raw=lambda frame:np.nan_to_num(frame.select(cols).to_numpy().astype(np.float32),nan=0.,posinf=0.,neginf=0.)
X=raw(tr);A=raw(ae)
# Two separate transformations: inner fit does not see inner validation; final transform sees all training only.
qi=QuantileTransformer(n_quantiles=min(1000,int((~im).sum())),output_distribution='normal',random_state=19143,subsample=None).fit(X[~im])
qt=QuantileTransformer(n_quantiles=min(1000,len(X)),output_distribution='normal',random_state=19143,subsample=None).fit(X)
y=tr['target'].to_numpy().astype(np.float32)
def weights(mask):
 ids=tr['pair_id'].to_numpy()[mask];u,inv,count=np.unique(ids,return_inverse=True,return_counts=True);w=1./count[inv];return (w/w.mean()).astype(np.float32)
arrays={'X_fit':qi.transform(X[~im]).astype(np.float32),'y_fit':y[~im],'w_fit':weights(~im),'X_val':qi.transform(X[im]).astype(np.float32),'y_val':y[im],'w_val':weights(im),'X_train':qt.transform(X).astype(np.float32),'y_train':y,'w_train':weights(np.ones(len(y),bool)),'X_audit':qt.transform(A).astype(np.float32)}
assert all(np.isfinite(a).all() for a in arrays.values());np.savez_compressed(OUT/'context.npz',**arrays);keys.write_parquet(OUT/'audit-keys.parquet');tr.select(K+['table_id','target']).write_parquet(OUT/'training-keys.parquet');joblib.dump(qi,OUT/'inner-transform.joblib');joblib.dump(qt,OUT/'full-transform.joblib')
receipt={'utc':datetime.now(timezone.utc).isoformat(),'context_sha256':sha(OUT/'context.npz'),'audit_keys_sha256':sha(OUT/'audit-keys.parquet'),'features':cols,'train_pairs':85,'train_pools':71,'train_rows':len(X),'train_listed':int(y.sum()),'inner_fit_pairs':tr.filter(~pl.col('table_id').is_in(sorted(valp)))['pair_id'].n_unique(),'inner_val_pairs':tr.filter(pl.col('table_id').is_in(sorted(valp)))['pair_id'].n_unique(),'inner_val_pools':sorted(valp),'audit_pairs':47,'audit_rows':len(A),'pool_player_hand_overlaps':0,'audit_truth_read':False,'fit_count':0,'protocol_sha256':sha(OUT/'protocol.json')}
(OUT/'context-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['features','inner_val_pools']}))
