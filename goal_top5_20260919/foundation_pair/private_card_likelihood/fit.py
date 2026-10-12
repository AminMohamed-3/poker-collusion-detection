"""Hash-gated fixed nested fit. Does not read coordination labels."""
import site; site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import hashlib,json,os,time,joblib
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
H=Path(__file__).resolve().parent; P=H.parents[2]; KEY=['pair_id','direction','hand_id','action_no']; SEED=35935
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weights(x):
 if '_w' in x.columns: x=x.drop('_w')
 c=x.group_by('hand_id','action_no','actor_id').len().rename({'len':'_n'}); z=x.join(c,on=['hand_id','action_no','actor_id'],validate='m:1').with_columns((1/pl.col('_n')).alias('_w'))
 assert z.group_by('hand_id','action_no','actor_id').agg(pl.col('_w').sum().alias('s')).select((pl.col('s')-1).abs().max()).item()<1e-12
 return z.drop('_n')
def model(reg=False):
 q=dict(n_estimators=200,num_leaves=15,min_child_samples=50,learning_rate=.1,random_state=SEED,n_jobs=2,verbosity=-1)
 return lgb.LGBMRegressor(**q) if reg else lgb.LGBMClassifier(**q)
def permute(z,v,outer,scope,edges):
 a=z.select(['street_index','own_strength','price','position_from_button','partner_last_action','facing_partner',*KEY]).to_dicts(); n=len(a)
 dec=np.searchsorted(edges,np.array([r['own_strength'] for r in a]),side='right')-1; price=np.digitize([r['price'] for r in a],[.1,.25,.5])
 levels=[(0,1,2,3,4,5),(0,1,2,4,5),(0,1,2,5),(0,1,2),(0,1),(0,)]
 base=[(int(r['street_index']),int(dec[i]),int(price[i]),int(r['position_from_button']),int(r['partner_last_action']),int(r['facing_partner'])) for i,r in enumerate(a)]
 chosen=[None]*n
 for lev in levels:
  groups={}
  for i,b in enumerate(base):
   if chosen[i] is None: groups.setdefault(tuple(b[j] for j in lev),[]).append(i)
  threshold=2 if len(lev)==1 else 20
  for k,ix in groups.items():
   if len(ix)>=threshold:
    for i in ix: chosen[i]=(lev,k)
 out=np.array(v,copy=True); bins={}
 for i,c in enumerate(chosen): bins.setdefault(c,[]).append(i)
 unchanged=0
 for c,ix in bins.items():
  if c is None or len(ix)<2: unchanged+=len(ix); continue
  ix=sorted(ix,key=lambda i:hashlib.sha256(f'{SEED}|placebo|{outer}|{scope}|{a[i]["pair_id"]}|{a[i]["direction"]}|{a[i]["hand_id"]}|{a[i]["action_no"]}'.encode()).hexdigest())
  out[ix]=np.asarray(v)[ix[-1:]+ix[:-1]]
 return out,unchanged
def main():
 man=H/'MANIFEST.json'; rel=H/'RELEASE.json'; assert rel.exists(),'root release absent'; rr=json.loads(rel.read_text()); assert rr['manifest_sha256']==sha(man)
 m=json.loads(man.read_text()); assert all(sha(P/k)==v for k,v in m['sources'].items()); assert all(sha(H/k)==v for k,v in m['artifacts'].items()); inputs=m['inputs']; paths=[P/k for k in m['sources'] if k.startswith('reports/conditional-action-experiment/action-cache/')]
 assert not (H/'FIT_RECEIPT.json').exists() and not any(H.glob('predictions_outer*.parquet')) and not any(H.glob('models_outer*.joblib')); started=time.monotonic(); deadline=started+900
 d=pl.read_parquet(paths).join(pl.read_parquet(H/'row_membership.parquet').select(*KEY,'actor_id'),on=KEY,validate='1:1').join(pl.read_parquet(H/'table_membership.parquet'),on='table_id',validate='m:1')
 cap=pl.read_parquet(H/'capped_training_keys.parquet').select(*KEY); outs=[]
 for o in [0,1]:
  tr=weights(d.filter(pl.col('outer_fold')!=o).join(cap,on=KEY,how='semi')); he=d.filter(pl.col('outer_fold')==o); inner=f'inner_fold_outer{o}'
  assert not (set(tr['table_id']) & set(he['table_id']))
  assert not (set(tr['actor_id']) & set(he['actor_id']))
  assert tr.height and he.height and set(tr['action_class'])=={0,1,2,3}
  membership={'outer':o,'train_key_sha256':hashlib.sha256(tr.select(KEY).write_csv().encode()).hexdigest(),'held_key_sha256':hashlib.sha256(he.select(KEY).write_csv().encode()).hexdigest(),'train_tables':sorted(tr['table_id'].unique().to_list()),'held_tables':sorted(he['table_id'].unique().to_list()),'inner':[]}
  surpr=np.empty(tr.height); tr=tr.with_row_index('_i'); inner_models=[]
  for j in [0,1]:
   fit=weights(tr.filter(pl.col(inner)!=j)); sc=tr.filter(pl.col(inner)==j)
   assert fit.height and sc.height and not (set(fit['table_id']) & set(sc['table_id'])) and not (set(fit['actor_id']) & set(sc['actor_id'])) and time.monotonic()<deadline
   r=model(True).fit(fit.select(inputs).to_numpy(),fit['partner_strength'].to_numpy(),sample_weight=fit['_w'].to_numpy()); surpr[sc['_i'].to_numpy()]=r.predict(sc.select(inputs).to_numpy())
   membership['inner'].append({'fold':j,'train_rows':fit.height,'score_rows':sc.height,'train_key_sha256':hashlib.sha256(fit.select(KEY).write_csv().encode()).hexdigest(),'score_key_sha256':hashlib.sha256(sc.select(KEY).write_csv().encode()).hexdigest()})
   inner_models.append(r)
  surpr=tr['partner_strength'].to_numpy()-surpr
  assert time.monotonic()<deadline; rf=model(True).fit(tr.select(inputs).to_numpy(),tr['partner_strength'].to_numpy(),sample_weight=tr['_w'].to_numpy()); hs=he['partner_strength'].to_numpy()-rf.predict(he.select(inputs).to_numpy())
  edges=np.unique(np.quantile(tr['own_strength'].to_numpy(),np.arange(11)/10)); tp,ut=permute(tr,surpr,o,'train',edges); hp,uh=permute(he,hs,o,'held',edges)
  X=tr.select(inputs).to_numpy(); Z=he.select(inputs).to_numpy(); y=tr['action_class'].to_numpy(); w=tr['_w'].to_numpy()
  assert set(y)=={0,1,2,3} and time.monotonic()<deadline
  p0=model().fit(X,y,sample_weight=w); assert time.monotonic()<deadline
  p1=model().fit(np.c_[X,surpr],y,sample_weight=w); assert time.monotonic()<deadline
  pp=model().fit(np.c_[X,tp],y,sample_weight=w); assert time.monotonic()<deadline
  assert all(np.array_equal(q.classes_,np.arange(4)) for q in [p0,p1,pp])
  probs=[q.predict_proba(xx) for q,xx in [(p0,Z),(p1,np.c_[Z,hs]),(pp,np.c_[Z,hp])]]
  out=he.select(*KEY,'table_id','action_class').with_columns(pl.lit(o).alias('outer_fold'),pl.Series('private_surprise',hs),pl.Series('placebo_surprise',hp))
  for name,pr in zip(['p0','p1','placebo'],probs): out=out.with_columns(*[pl.Series(f'{name}_{k}',pr[:,k]) for k in range(4)])
  modelpath=H/f'models_outer{o}.joblib'; joblib.dump({'p0':p0,'p1':p1,'placebo':pp,'strength_full':rf,'strength_inner':inner_models,'inputs':inputs,'membership':membership},modelpath)
  reload=joblib.load(modelpath); check=reload['p1'].predict_proba(np.c_[Z,hs]); assert np.array_equal(check,probs[1])
  out.write_parquet(H/f'predictions_outer{o}.parquet'); outs.append({'outer':o,'train':tr.height,'held':he.height,'unchanged_train':ut,'unchanged_held':uh,'membership':membership,'prediction_sha256':sha(H/f'predictions_outer{o}.parquet'),'model_sha256':sha(modelpath),'replay_exact':True})
 assert time.monotonic()<=deadline
 (H/'FIT_RECEIPT.json').write_text(json.dumps({'manifest_sha256':sha(man),'seconds':time.monotonic()-started,'folds':outs},indent=2)+'\n')
if __name__=='__main__': main()
