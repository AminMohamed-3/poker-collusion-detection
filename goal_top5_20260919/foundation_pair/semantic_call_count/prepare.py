import os,site,json,hashlib
from pathlib import Path
os.environ.update(POLARS_MAX_THREADS='2',OMP_NUM_THREADS='2')
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import polars as pl,numpy as np
R=Path(r'C:/Data/Projects/Money Making/work/poker');OUT=R/'goal_top5_20260919/foundation_pair/semantic_call_count';SCR=R/'goal_top5_20260919/foundation_pair/screen';FULL=R/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet';PC=R/'Claude Version/experiments/pu_pairs_20260911';LAB=R/'data/development_labels.csv';H=R/'data/hands.parquet';S=R/'data/seats.parquet';A=R/'data/actions.parquet';SALT='foundation-pf35-screen-v1:'; FAMS={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}
RAW=['n_call_sum','n_call_gap','n_call_min','n_call_max']; DESC=[x+'_'+a for a in ('mean','std','max','q95') for x in RAW]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fold(t):return int(hashlib.sha256((SALT+t).encode()).hexdigest()[:16],16)%2
def semantics_fixture():
 rows=[('call',5,5,1,1),('all_in',5,5,0,1),('all_in',3,5,0,1),('all_in',6,5,0,0),('raise',6,5,0,0)]
 got=[(int(a=='call'),int((a=='call') or (a=='all_in' and amt<=tc))) for a,amt,tc,_,_ in rows]
 assert got==[(x,y) for *_,x,y in rows]
 # Zero-call and asymmetric players must survive the symmetric pair transform.
 l1,l2=0.,2.; assert (l1+l2,abs(l1-l2),min(l1,l2),max(l1,l2))==(2.,2.,0.,2.)
 return {'action_cases':len(rows),'zero_call_asymmetric_pair':[2.,2.,0.,2.]}
def reconstruct(pairs,hands):
 ids=hands.select('hand_id'); seats=pl.scan_parquet(S).join(ids.lazy(),on='hand_id',how='semi').select('hand_id','player_id').collect()
 ac=(pl.scan_parquet(A).join(ids.lazy(),on='hand_id',how='semi').group_by('hand_id','player_id').agg((pl.col('action')=='call').sum().alias('literal'),((pl.col('action')=='call')|((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call')))).sum().alias('semantic')).collect())
 one=seats.join(ac,on=['hand_id','player_id'],how='left').with_columns(pl.col('literal','semantic').fill_null(0).cast(pl.Float64))
 q=(pairs.join(one.rename({'player_id':'player_1','literal':'l1','semantic':'s1'}),on='player_1').join(one.rename({'player_id':'player_2','literal':'l2','semantic':'s2'}),on=['hand_id','player_2']))
 def side(pre,a,b):return [(pl.col(a)+pl.col(b)).alias(pre+'_sum'),(pl.col(a)-pl.col(b)).abs().alias(pre+'_gap'),pl.min_horizontal(a,b).alias(pre+'_min'),pl.max_horizontal(a,b).alias(pre+'_max')]
 q=q.with_columns(*side('literal','l1','l2'),*side('semantic','s1','s2'))
 ag=[]
 for pre in ('literal','semantic'):
  for stat in ('mean','std','max','q95'):
   for x in RAW:
    c=x.replace('n_call',pre); e={'mean':pl.col(c).mean(),'std':pl.col(c).std().fill_null(0),'max':pl.col(c).max(),'q95':pl.col(c).quantile(.95)}[stat];ag.append(e.alias(x+'_'+stat+'_'+pre))
 return q.group_by('pair_id').agg(*ag).sort('pair_id')
def main():
 OUT.mkdir(parents=True,exist_ok=True); fixtures=semantics_fixture();d=pl.read_parquet(FULL).join(pl.read_csv(LAB).select('pair_id','player_1','player_2','label','behavior_family'),on='pair_id',validate='1:1').sort('pair_id');hands=pl.read_parquet(H).filter(pl.col('phase')=='development');pairs=d.select('pair_id','player_1','player_2','table_id')
 full=reconstruct(pairs,hands)
 pp=pl.concat([pl.concat([pl.read_parquet(x) for x in sorted((PC/f'fold{f}-prefix60').glob('batch_*-pairs.parquet'))],how='vertical_relaxed') for f in (1,2)],how='vertical_relaxed').unique('pair_id',keep='first').join(pairs,on='pair_id',how='inner',validate='1:1').sort('pair_id');assert pp.height==813
 hp=hands.sort('table_id','started_at').with_columns((pl.int_range(pl.len()).over('table_id')/pl.len().over('table_id')).alias('_pos')).filter(pl.col('_pos')<.6).drop('_pos');pref=reconstruct(pp.select('pair_id','player_1','player_2','table_id'),hp)
 # Literal parity is a hard precondition.
 def parity(re,cached):
  assert cached['pair_id'].n_unique()==cached.height and re['pair_id'].n_unique()==re.height
  assert set(cached['pair_id'].to_list())==set(re['pair_id'].to_list())
  j=cached.select(['pair_id']+DESC).join(re,on='pair_id',validate='1:1'); assert j.height==cached.height==re.height
  dif={x:float((j[x]-j[x+'_literal']).abs().max()) for x in DESC};return dif
 pf=parity(full,d); ppcache=pl.concat([pl.concat([pl.read_parquet(x) for x in sorted((PC/f'fold{f}-prefix60').glob('batch_*-pairs.parquet'))],how='vertical_relaxed') for f in (1,2)],how='vertical_relaxed').unique('pair_id',keep='first').join(d.select('pair_id'),on='pair_id',how='inner').sort('pair_id');ppar=parity(pref,ppcache)
 assert max(pf.values())<1e-6 and max(ppar.values())<1e-6,(max(pf.values()),max(ppar.values()))
 def corrected(base,re):
  j=base.join(re,on='pair_id',validate='1:1');return j.with_columns(*[pl.col(x+'_semantic').cast(pl.Float32).alias(x) for x in DESC]).select(base.columns)
 dc=corrected(d,full);pc=corrected(ppcache,pref); dc.write_parquet(OUT/'corrected_full.parquet');pc.write_parquet(OUT/'corrected_prefix.parquet')
 feats=[c for c,v in d.schema.items() if c not in {'pair_id','table_id','player_1','player_2','label','behavior_family'} and v.is_numeric()];X0=d.select(feats).to_numpy().astype('float32');X=dc.select(feats).to_numpy().astype('float32');PX=pc.select(feats).to_numpy().astype('float32');X0[~np.isfinite(X0)]=np.nan;X[~np.isfinite(X)]=np.nan;PX[~np.isfinite(PX)]=np.nan;ids=np.asarray(d['pair_id']);tids=np.asarray(d['table_id']);folds=np.asarray([fold(x) for x in tids]);y=np.asarray([FAMS[v] if z else 0 for z,v in zip(d['label'],d['behavior_family'])],dtype=np.int8);pmap={x:i for i,x in enumerate(pc['pair_id'])};model_input_parity={}
 for f in (0,1):
  old=np.load(SCR/f'context_fold{f}.npz');names=old['feature_name'].tolist();ix=np.asarray([feats.index(x) for x in names]);tr=folds!=f;va=~tr
  assert np.array_equal(old['pair_id'],ids[va]) and np.array_equal(old['y_train'],y[tr]) and np.array_equal(old['feature_name'],np.asarray(names))
  assert np.array_equal(old['X_train'],X0[tr][:,ix],equal_nan=True) and np.array_equal(old['X_held'],X0[va][:,ix],equal_nan=True)
  ep=np.load(SCR/f'evaluator_fold{f}_prefix60.npz');hpix=np.asarray([pmap[x] for x in ep['pair_id']]);model_input_parity[str(f)]='exact_float32_train_held_keys_labels_features';np.savez_compressed(OUT/f'context_fold{f}.npz',X_train=X[tr][:,ix],y_train=y[tr],X_full=X[va][:,ix],X_prefix=PX[hpix][:,ix],full_pair_id=ids[va],prefix_pair_id=ep['pair_id'],feature_name=np.asarray(names))
 changes={x:{'full_rows_changed':int((d[x]!=dc[x]).sum()),'prefix_rows_changed':int((ppcache[x]!=pc[x]).sum()),'full_max_abs':float((d[x]-dc[x]).abs().max()),'prefix_max_abs':float((ppcache[x]-pc[x]).abs().max())} for x in DESC}
 receipt={'status':'PREPARED_NO_FIT','classification':'semantic-call-count alternative; original literal call count remains valid and protected','fixtures':fixtures,'literal_parity_tolerance':1e-6,'literal_parity_full_max':max(pf.values()),'literal_parity_prefix_max':max(ppar.values()),'original_model_input_parity':model_input_parity,'rows':{'full':1860,'prefix':813,'eligible_prefix':int((pc['shared_hands']>=38).sum())},'descendant_changes':changes,'selected_touched':{str(f):[x for x in np.load(SCR/f'context_fold{f}.npz')['feature_name'].tolist() if x in DESC] for f in (0,1)},'hashes':{p.name:sha(p) for p in [OUT/'corrected_full.parquet',OUT/'corrected_prefix.parquet',OUT/'context_fold0.npz',OUT/'context_fold1.npz']},'sources':{str(p):sha(p) for p in [FULL,LAB,H,S,A]}}
 (OUT/'prep_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(OUT/'PREPARED_ONLY').write_text('No fit or quality read authorized.\n');print(json.dumps({k:receipt[k] for k in ('literal_parity_full_max','literal_parity_prefix_max','rows','selected_touched')},indent=2))
if __name__=='__main__':main()

