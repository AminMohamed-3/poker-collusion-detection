import os,site,json,hashlib
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import numpy as np
import polars as pl
ROOT=Path(r'C:/Data/Projects/Money Making'); P=ROOT/'work/poker'; OUT=P/'goal_top5_20260919/foundation_pair/history_context'; SCR=P/'goal_top5_20260919/foundation_pair/screen'
FULL=P/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet'; LAB=P/'data/development_labels.csv'; PC=P/'Claude Version/experiments/pu_pairs_20260911'; SALT='foundation-pf35-screen-v1:'
FAMS={'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fold(t): return int(hashlib.sha256((SALT+t).encode()).hexdigest()[:16],16)%2
def main():
 OUT.mkdir(parents=True,exist_ok=True); assert not (OUT/'GPU_RELEASE.json').exists()
 d=pl.read_parquet(FULL).join(pl.read_csv(LAB).select('pair_id','label','behavior_family','player_1','player_2'),on='pair_id',validate='1:1')
 feats=[c for c,v in d.schema.items() if c not in {'pair_id','table_id','label','behavior_family','player_1','player_2'} and v.is_numeric()]; assert len(feats)==653
 ids=np.asarray(d['pair_id'].to_list()); tids=np.asarray(d['table_id'].to_list()); folds=np.asarray([fold(x) for x in tids]); y=np.asarray([FAMS[v] if z else 0 for z,v in zip(d['label'],d['behavior_family'])],dtype=np.int8)
 X=d.select(feats).to_numpy().astype('float32'); X[~np.isfinite(X)]=np.nan
 frames=[]; prefix_files=[]
 for q in (1,2):
  ff=sorted((PC/f'fold{q}-prefix60').glob('batch_*-pairs.parquet')); prefix_files += ff; frames.append(pl.concat([pl.read_parquet(x) for x in ff],how='vertical_relaxed'))
 p=(pl.concat(frames,how='vertical_relaxed').unique('pair_id',keep='first')
    .join(d.select('pair_id'),on='pair_id',how='inner',validate='1:1'))
 assert p.height==813 and all(c in p.columns for c in feats)
 pmap={v:i for i,v in enumerate(p['pair_id'].to_list())}; PX=p.select(feats).to_numpy().astype('float32'); PX[~np.isfinite(PX)]=np.nan
 lab=pl.read_csv(LAB); players=dict(zip(lab['pair_id'],zip(lab['player_1'],lab['player_2']))); rec=[]
 for f in (0,1):
  base=np.load(SCR/f'context_fold{f}.npz'); names=base['feature_name'].tolist(); ix=np.asarray([feats.index(v) for v in names]); tr=folds!=f; va=~tr
  assert np.array_equal(base['X_train'],X[tr][:,ix],equal_nan=True) and np.array_equal(base['y_train'],y[tr])
  addidx=np.asarray([i for i in np.flatnonzero(tr) if ids[i] in pmap]); pix=np.asarray([pmap[ids[i]] for i in addidx])
  cand=np.concatenate([X[tr][:,ix],PX[pix][:,ix]]); ctrl=np.concatenate([X[tr][:,ix],X[addidx][:,ix]]); yy=np.concatenate([y[tr],y[addidx]])
  ep=np.load(SCR/f'evaluator_fold{f}_prefix60.npz'); hp=np.asarray([pmap[v] for v in ep['pair_id']]); assert np.array_equal(ep['pair_id'],ids[[i for i in np.flatnonzero(va) if ids[i] in pmap]])
  for arm,xt in [('prefix',cand),('duplicate',ctrl)]:
   np.savez_compressed(OUT/f'context_{arm}_fold{f}.npz',X_train=xt,y_train=yy,X_full=X[va][:,ix],X_prefix=PX[hp][:,ix],full_pair_id=ids[va],prefix_pair_id=ep['pair_id'],feature_name=np.asarray(names))
  np.savez_compressed(OUT/f'membership_fold{f}.npz',base_train_pair_id=ids[tr],appended_pair_id=ids[addidx],held_full_pair_id=ids[va],held_prefix_pair_id=ep['pair_id'])
  tr_tables=set(tids[tr]); va_tables=set(tids[va]); tr_players={z for pid in ids[tr] for z in players[pid]}; va_players={z for pid in ids[va] for z in players[pid]}
  assert not tr_tables&va_tables and not tr_players&va_players and not set(ids[tr])&set(ids[va])
  rec.append({'fold':f,'base_train':int(tr.sum()),'appended':len(addidx),'context_rows':len(yy),'class_counts':np.bincount(yy,minlength=4).tolist(),'held_full':int(va.sum()),'held_prefix':len(hp),'selected_features':names,'exclusions':{'table_overlap':0,'player_overlap':0,'pair_overlap':0,'physical_hand_overlap':0,'physical_basis':'prefix histories are derived only from training-pair tables; whole-table and player folds are disjoint'}})
 ev={'fold':folds,'pair_id':ids,'y':y,'table_id':tids}; np.savez_compressed(OUT/'evaluator.npz',**ev)
 epids=np.concatenate([np.load(SCR/f'evaluator_fold{f}_prefix60.npz')['pair_id'] for f in (0,1)]); shared=dict(zip(p['pair_id'].to_list(),p['shared_hands'].to_list())); full_shared=dict(zip(d['pair_id'].to_list(),d['shared_hands'].to_list()))
 np.savez_compressed(OUT/'evaluator_prefix.npz',pair_id=epids,y=np.concatenate([np.load(SCR/f'evaluator_fold{f}_prefix60.npz')['y'] for f in (0,1)]),table_id=np.concatenate([np.load(SCR/f'evaluator_fold{f}_prefix60.npz')['table_id'] for f in (0,1)]),fold=np.concatenate([np.full(len(np.load(SCR/f'evaluator_fold{f}_prefix60.npz')['y']),f) for f in (0,1)]),shared_hands=np.asarray([shared[x] for x in epids],dtype=np.int32))
 ratios=np.asarray([shared[x]/full_shared[x] for x in epids])
 context_files=[OUT/f'context_{a}_fold{f}.npz' for a in ('prefix','duplicate') for f in (0,1)]
 man={'status':'PREPARED_NO_GPU_AUTHORIZATION','seed':35935,'members':4,'batch':256,'memory_fraction':.70,'prefix_evaluation':{'all':len(epids),'eligible_shared_hands_ge_38':int(sum(shared[x]>=38 for x in epids)),'missing_prefix_coverage':int(len(ids)-len(epids)),'shared_hands_ratio_prefix_over_full':{'min':float(ratios.min()),'q10':float(np.quantile(ratios,.1)),'median':float(np.median(ratios)),'q90':float(np.quantile(ratios,.9)),'max':float(ratios.max()),'ratio_equal_1_count':int((ratios==1).sum())}},'source_hashes':{'full':sha(FULL),'labels':sha(LAB),'original_predictions':sha(SCR/'pf35_predictions.npz'),'original_prep':sha(SCR/'prep_receipt.json')},'prefix_file_hashes':{str(x):sha(x) for x in prefix_files},'context_hashes':{x.name:sha(x) for x in context_files},'membership_hashes':{f'membership_fold{f}.npz':sha(OUT/f'membership_fold{f}.npz') for f in (0,1)},'evaluator_hashes':{'evaluator.npz':sha(OUT/'evaluator.npz'),'evaluator_prefix.npz':sha(OUT/'evaluator_prefix.npz')},'folds':rec}
 (OUT/'prep_manifest.json').write_text(json.dumps(man,indent=2)+'\n'); (OUT/'PREPARED_ONLY').write_text('GPU inference requires explicit root release.\n')
if __name__=='__main__': main()
