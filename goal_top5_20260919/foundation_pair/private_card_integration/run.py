"""Released-only nested reduced integration probe. Labels are used only by pair learners."""
import site;site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import hashlib,json,time,joblib,sys
from pathlib import Path
import numpy as np,polars as pl,lightgbm as lgb
H=Path(__file__).resolve().parent; P=H.parents[2]; L=H.parent/'private_card_likelihood'; sys.path.insert(0,str(L)); import fit as F
KEY=['pair_id','direction','hand_id','action_no']; SEED=35935
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def keyhash(z,cols):return hashlib.sha256(z.select(cols).sort(cols).write_csv().encode()).hexdigest()
def abit(t,o,j):return int(hashlib.sha256(f'private-integration-subinner|{o}|{j}|{t}'.encode()).hexdigest(),16)&1
def ll_summary(d,p0,p1):
 y=d['action_class'].to_numpy();i=np.arange(d.height);g=np.log(np.maximum(p1[i,y],1e-12))-np.log(np.maximum(p0[i,y],1e-12))
 z=d.select('pair_id','direction').with_columns(pl.Series('_g',g)).group_by('pair_id','direction').agg(pl.mean('_g')).group_by('pair_id').agg(pl.mean('_g').alias('priv_ll_mean'),pl.col('direction').n_unique().alias('_nd'));assert z['_nd'].min()==2;return z.drop('_nd')
def main():
 start=time.monotonic(); deadline=start+900; man=H/'MANIFEST.json'; rel=H/'RELEASE.json'; assert rel.exists() and json.loads(rel.read_text())['manifest_sha256']==sha(man);m=json.loads(man.read_text());assert all(sha(Path(k))==v for k,v in m['sources'].items()) and all(sha(H/k)==v for k,v in m['artifacts'].items());assert not (H/'RUN_RECEIPT.json').exists() and not any(H.glob('pair_predictions_outer*.parquet')) and not any(H.glob('*models_outer*.joblib')) and not any(H.glob('upstream_outer*')) and not any(H.glob('nested_train_summary_outer*.parquet'))
 lm=json.loads((L/'MANIFEST.json').read_text()); inputs=lm['inputs']; paths=[P/k for k in lm['sources'] if k.startswith('reports/conditional-action-experiment/action-cache/')]
 d=pl.read_parquet(paths).join(pl.read_parquet(L/'row_membership.parquet').select(*KEY,'actor_id'),on=KEY,validate='1:1').join(pl.read_parquet(L/'table_membership.parquet'),on='table_id',validate='m:1');cap=pl.read_parquet(L/'capped_training_keys.parquet').select(*KEY)
 raw=pl.read_parquet(P/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet'); labels=pl.read_csv(P/'data/development_labels.csv').select('pair_id','label','behavior_family'); raw=raw.join(labels,on='pair_id',validate='1:1').join(pl.read_parquet(L/'table_membership.parquet').select('table_id','outer_fold'),on='table_id',validate='m:1'); feats=[c for c,t in raw.schema.items() if c not in {'pair_id','table_id','label','behavior_family','outer_fold'} and t.is_numeric()];assert len(feats)==653
 receipts=[]
 for o in [0,1]:
  otr=d.filter(pl.col('outer_fold')!=o); summaries=[]
  # Deterministic inner fold is the already frozen inner assignment for this outer fold.
  inn=f'inner_fold_outer{o}'; pipeline_receipts=[]
  for j in [0,1]:
   train=F.weights(otr.filter(pl.col(inn)!=j).join(cap,on=KEY,how='semi')); held=otr.filter(pl.col(inn)==j); assert train.height and held.height and not(set(train['table_id'])&set(held['table_id'])) and not(set(train['actor_id'])&set(held['actor_id']))
   train=train.with_columns(pl.Series('_sub',[abit(t,o,j) for t in train['table_id']])).with_row_index('_i'); pred=np.empty(train.height); sub=[]; subreceipts=[]
   for s in [0,1]:
    ft=F.weights(train.filter(pl.col('_sub')!=s));sc=train.filter(pl.col('_sub')==s);assert ft.height and sc.height and not(set(ft['table_id'])&set(sc['table_id'])) and not(set(ft['actor_id'])&set(sc['actor_id'])) and time.monotonic()<deadline
    r=F.model(True).fit(ft.select(inputs).to_numpy(),ft['partner_strength'].to_numpy(),sample_weight=ft['_w'].to_numpy());pred[sc['_i'].to_numpy()]=r.predict(sc.select(inputs).to_numpy());sub.append(r)
    subreceipts.append({'subinner':s,'train_rows':ft.height,'score_rows':sc.height,'train_keys':keyhash(ft,KEY),'score_keys':keyhash(sc,KEY),'train_tables':sorted(ft['table_id'].unique().to_list()),'score_tables':sorted(sc['table_id'].unique().to_list()),'train_actors_sha256':keyhash(ft.unique('actor_id'),['actor_id']),'score_actors_sha256':keyhash(sc.unique('actor_id'),['actor_id']),'train_hands_sha256':keyhash(ft.unique('hand_id'),['hand_id']),'score_hands_sha256':keyhash(sc.unique('hand_id'),['hand_id'])})
   surprise=train['partner_strength'].to_numpy()-pred; rf=F.model(True).fit(train.select(inputs).to_numpy(),train['partner_strength'].to_numpy(),sample_weight=train['_w'].to_numpy()); hs=held['partner_strength'].to_numpy()-rf.predict(held.select(inputs).to_numpy());X=train.select(inputs).to_numpy();Z=held.select(inputs).to_numpy();y=train['action_class'].to_numpy();w=train['_w'].to_numpy();assert set(y)=={0,1,2,3} and time.monotonic()<deadline
   p0=F.model().fit(X,y,sample_weight=w);p1=F.model().fit(np.c_[X,surprise],y,sample_weight=w);q0=p0.predict_proba(Z);q1=p1.predict_proba(np.c_[Z,hs]);summaries.append(ll_summary(held,q0,q1));joblib.dump({'strength_subinner':sub,'strength_full':rf,'p0':p0,'p1':p1},H/f'upstream_outer{o}_inner{j}.joblib');pipeline_receipts.append({'inner':j,'train_rows':train.height,'held_rows':held.height,'train_keys':keyhash(train,KEY),'held_keys':keyhash(held,KEY),'subinner':subreceipts})
   (H/f'upstream_outer{o}_inner{j}_membership.json').write_text(json.dumps(pipeline_receipts[-1],indent=2)+'\n')
  sf=pl.concat(summaries);assert sf.height==raw.filter(pl.col('outer_fold')!=o).height and sf['pair_id'].n_unique()==sf.height;sf.with_columns(pl.lit('outer_train_inner_oof').alias('scope')).write_parquet(H/f'nested_train_summary_outer{o}.parquet')
  # Outer-held summary is reused only for scoring.
  ph=pl.read_parquet(L/f'predictions_outer{o}.parquet');q0=np.column_stack([ph[f'p0_{k}'] for k in range(4)]);q1=np.column_stack([ph[f'p1_{k}'] for k in range(4)]);sh=ll_summary(ph,q0,q1)
  tr=raw.filter(pl.col('outer_fold')!=o).join(sf,on='pair_id',validate='1:1');he=raw.filter(pl.col('outer_fold')==o).join(sh,on='pair_id',validate='1:1');assert not(set(tr['table_id'])&set(he['table_id'])) and time.monotonic()<deadline
  prm=dict(n_estimators=400,num_leaves=31,learning_rate=.05,min_child_samples=20,feature_fraction=1.,bagging_fraction=1.,random_state=SEED,n_jobs=2,verbosity=-1); cmap={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3};y=np.array([cmap[x] for x in tr['behavior_family']]);b=lgb.LGBMClassifier(**prm).fit(tr.select(feats).to_numpy(),y);a=lgb.LGBMClassifier(**prm).fit(tr.select(feats+['priv_ll_mean']).to_numpy(),y);assert np.array_equal(b.classes_,np.arange(4)) and np.array_equal(a.classes_,np.arange(4));pb=b.predict_proba(he.select(feats).to_numpy());pa=a.predict_proba(he.select(feats+['priv_ll_mean']).to_numpy())
  mp=H/f'pair_models_outer{o}.joblib';joblib.dump({'baseline':b,'augmented':a,'features':feats},mp);rr=joblib.load(mp);assert np.array_equal(rr['augmented'].predict_proba(he.select(feats+['priv_ll_mean']).to_numpy()),pa)
  out=he.select('pair_id','table_id').with_columns(pl.lit(o).alias('outer_fold'),he['priv_ll_mean'],*[pl.Series(f'baseline_{k}',pb[:,k]) for k in range(4)],*[pl.Series(f'augmented_{k}',pa[:,k]) for k in range(4)]);out.write_parquet(H/f'pair_predictions_outer{o}.parquet');receipts.append({'outer':o,'rows':he.height,'prediction_sha256':sha(H/f'pair_predictions_outer{o}.parquet'),'model_sha256':sha(mp),'nested_summary_sha256':sha(H/f'nested_train_summary_outer{o}.parquet'),'pipelines':pipeline_receipts,'replay_exact':True})
 assert time.monotonic()<=deadline;(H/'RUN_RECEIPT.json').write_text(json.dumps({'manifest_sha256':sha(man),'seconds':time.monotonic()-start,'folds':receipts},indent=2)+'\n')
if __name__=='__main__':main()
