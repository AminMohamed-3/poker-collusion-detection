"""Prepare frozen TabPFN-3 outer-confirmation contexts; no TFM inference or metrics."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
import sys,json,time
from datetime import datetime,timezone
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl
import lightgbm as lgb
from audit_common import *
from evidence_aug.frame import training_relevance,attach_relevance,family_columns,score
from evidence_aug.fold import fold_roles
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step10_chronology_fit import add_chronology,OUT as CHRONO_OUT
from evidence_aug.step14_audit_truncation import split_of,SPLITS

OUT=HERE/'experiments/tabpfn3_confirmation_20260912'; PRIMARY='tabpfn3_default'
STRUCT=CV/'experiments/family_stage2_20260911/dev-structural-features.parquet'; OLD=CV/'experiments/family_stage2_r382_20260911'
PROTOCOL=HERE/'research/tabpfn3_confirmation_protocol_20260912.md'

def ordinal(frame,v):return frame.select('pair_id').with_columns(pl.Series('s',v)).select(pl.col('s').rank('ordinal',descending=True).over('pair_id'))['s'].to_numpy()
def h(p):return sha(p)
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
 t=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);(OUT/'contexts').mkdir(exist_ok=True);(OUT/'scores').mkdir(exist_ok=True)
 assert not (OUT/'plan.json').exists(),'write-once confirmation'
 protected=check_protected();best=CV/'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv'
 import tabpfn3_evidence_pilot as pilot
 audit=json.loads((HERE/'experiments/tabpfn3_evidence_20260912/results.json').read_text());assert audit['primary_eligible'] and audit['negative_control_rejected']
 sources=[Path(__file__),HERE/'src/audit_common.py',PROTOCOL,STRUCT,HERE/'src/tabpfn3_evidence_pilot.py',HERE/'experiments/tabpfn3_evidence_20260912/freeze.json',best,HERE/'outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv',
          HERE/'src/tabpfn3_confirm_evaluate.py',HERE/'src/tabpfn3_confirm_infer.py',HERE/'experiments/tabpfn3_evidence_20260912/results.json',
          pilot.CHECKPOINT,pilot.RUNTIME/'runtime-receipt.json',ROOT/'data/hands.parquet',ROOT/'data/development_labels.csv',ROOT/'data/development_evidence.csv']
 sources += [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT.resolve())) and Path(m.__file__).suffix=='.py' and 'site-packages' not in str(m.__file__)]
 sources += [CHRONO_OUT/'scores'/f'fold{fold}-chrono1.parquet' for fold in (0,1,2)]
 sources += [OLD/'scores'/f'fold{fold}-{name}.parquet' for fold in (0,1,2) for name in ('allfam_top20_struct','allfam_top20_r382')]
 plan={'created_at':datetime.now(timezone.utc).isoformat(),'primary':PRIMARY,'confirmation_folds':[0,1,2],
  'sources':{str(p):h(p) for p in sources},'protected':protected,'best_sha256':h(best),'reference_params':dict(CLF,random_state=11),'candidate_params':pilot.PARAMS,'audit_result':audit['comparisons']['tabpfn3_default'],
  'contexts':'Nine whole-outer-pool-disjoint train/held float64 arrays: 453 DT/SP or 75 CI columns; held labels omitted.',
  'reference':'Exact chronology stage1 plus r382 DT/SP and structural CI hybrid; predicted none stays stage1.',
  'gate':{'mean_unique_pair_delta_min':.01,'ci_lower_positive':True,'all_fold_nonnegative':True,'family_floor':-.01,'halves_nonnegative':True},
  'held_evidence_evaluation_before_freeze':False,'no_rescue':True}
 write(OUT/'plan.json',plan)
 structural=pl.read_parquet(STRUCT);struct=[c for c in structural.columns if c not in ('pair_id','hand_id')]; manifests=[];receipts=[]
 for fold in (0,1,2):
  print('Preparing outerfold',fold,flush=True)
  data,features,pools,identity=load_fold(fold);data=add_chronology(data).join(structural,on=['pair_id','hand_id'],validate='1:1').sort('pair_id','hand_id')
  held=data.filter(pl.col('table_id').is_in(pools)).sort('pair_id','hand_id');train=data.filter(~pl.col('table_id').is_in(pools)).sort('pair_id','hand_id')
  evidence=training_relevance(set(train['table_id']));assert not set(evidence['pair_id'])&set(held['pair_id']);train=attach_relevance(train,evidence).sort('pair_id','hand_id')
  roles=fold_roles(fold);held=held.join(roles,on='pair_id',validate='m:1').sort('pair_id','hand_id')
  stage1=fit_ranker(train,features+['chrono_frac']);sh=score(stage1,family_columns(held,'predicted_behavior'),features+['chrono_frac']);old=pl.read_parquet(CHRONO_OUT/'scores'/f'fold{fold}-chrono1.parquet')
  assert held.select('pair_id','hand_id').equals(old.select('pair_id','hand_id')) and np.array_equal(sh,old['score'].to_numpy());stage1.booster_.save_model(str(OUT/f'fold{fold}-stage1.txt'))
  split=np.asarray([split_of(p) for p in train['table_id']]);st=np.full(train.height,np.nan)
  for s in range(SPLITS):
   te=np.flatnonzero(split==s);inner=fit_ranker(train.filter(pl.Series(split!=s)),features+['chrono_frac']);st[te]=score(inner,family_columns(train[te],'behavior_family'),features+['chrono_frac'])
  assert np.isfinite(st).all();train=train.with_columns(pl.Series('oof',st));held=held.with_columns(pl.Series('oof',sh))
  for z in (train,held):assert np.array_equal(z['pot_bb'].to_numpy().astype(np.float32),z['pot_bb_right'].to_numpy().astype(np.float32))
  train=train.with_columns(pl.col('pot_bb_right').alias('pot_bb'),pl.col('oof').rank('ordinal',descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
  held=held.with_columns(pl.col('pot_bb_right').alias('pot_bb'),pl.col('oof').rank('ordinal',descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
  shared=struct+['chrono_rank','chrono_frac','n_hands','oof','oof_rank'];extra=[x for x in features if not x.startswith('family_') and x not in struct]
  train=train.with_columns(pl.col(extra).fill_null(-999.));held=held.with_columns(pl.col(extra).fill_null(-999.))
  oldstruct=pl.read_parquet(OLD/'scores'/f'fold{fold}-allfam_top20_struct.parquet');oldr=pl.read_parquet(OLD/'scores'/f'fold{fold}-allfam_top20_r382.parquet');assert held.select('pair_id','hand_id').equals(oldstruct.select('pair_id','hand_id')) and held.select('pair_id','hand_id').equals(oldr.select('pair_id','hand_id'))
  expected=np.where((held['predicted_behavior']=='coordinated_isolation').to_numpy(),oldstruct['score'].to_numpy(),oldr['score'].to_numpy());control=sh.copy();modeldir=OUT/'models'/f'fold{fold}';modeldir.mkdir(parents=True,exist_ok=True)
  for fam in FAMILIES:
   cols=shared+(extra if fam!='coordinated_isolation' else []);tr=train.filter((pl.col('behavior_family')==fam)&(pl.col('oof_rank')<=20));ix=np.flatnonzero(((held['predicted_behavior']==fam)&(held['oof_rank']<=20)).to_numpy())
   ref=lgb.LGBMClassifier(**CLF,random_state=11).fit(tr.select(cols).to_numpy().astype(float),tr['relevance'].to_numpy());ref.booster_.save_model(str(modeldir/f'{fam}-incumbent.txt'))
   if len(ix):control[ix]=ref.predict_proba(held[ix].select(cols).to_numpy().astype(float))[:,1]+10
   control[((held['predicted_behavior']==fam)&(held['oof_rank']>20)).to_numpy()]=sh[((held['predicted_behavior']==fam)&(held['oof_rank']>20)).to_numpy()]-1e6
   xtr=tr.select(cols).to_numpy().astype(np.float64);xte=held[ix].select(cols).to_numpy().astype(np.float64);assert xtr.shape[1] in (75,453) and xte.shape[1]==xtr.shape[1] and np.isfinite(xtr).all() and np.isfinite(xte).all()
   assert cols==next(m['columns'] for m in json.loads((HERE/'experiments/tabpfn3_evidence_20260912/freeze.json').read_text())['folds'] if m['family']==fam),'feature route changed'
   assert tr.group_by('pair_id').len()['len'].max()<=20
   stem=f'fold{fold}-{fam}';p=OUT/'contexts'/(stem+'.npz');np.savez_compressed(p,xtrain=xtr,ytrain=tr['relevance'].to_numpy().astype(np.int64),xtest=xte,held_indices=ix.astype(np.int64),held_pair_ids=held['pair_id'].to_numpy()[ix].astype(str),groups_train=tr['table_id'].to_numpy().astype('U'),groups_held=held['table_id'].to_numpy()[ix].astype('U'))
   assert not set(tr['table_id'])&set(held[ix]['table_id']);manifests.append({'stem':stem,'fold':fold,'family':fam,'context_file':str(Path('contexts')/p.name),'context_sha256':h(p),'train_shape':list(xtr.shape),'query_shape':list(xte.shape),'width':xtr.shape[1],'train_pools':sorted(set(tr['table_id'])),'held_pools':sorted(set(held[ix]['table_id'])),'held_indices_sha256':__import__('hashlib').sha256(ix.astype(np.int64).tobytes()).hexdigest(),'columns':cols,'class_counts':np.bincount(tr['relevance'].to_numpy().astype(np.int64),minlength=2).tolist(),'control_exact':True})
  none=(~held['predicted_behavior'].is_in(FAMILIES)).to_numpy();np.testing.assert_array_equal(expected[none],sh[none])
  diff=float(np.max(np.abs(control-expected)));assert diff<=2e-14 and np.array_equal(ordinal(held,control),ordinal(held,expected));write(OUT/f'fold{fold}-reference-preflight.json',{'stage1_exact':True,'current_hybrid_max_abs_diff':diff,'ordinal_exact':True})
  held.select('pair_id','hand_id','table_id','predicted_behavior','oof','oof_rank').with_columns(pl.Series('currentcontrol',control)).write_parquet(OUT/f'fold{fold}-held.parquet');roles.write_csv(OUT/f'fold{fold}-roles.csv');receipts.append({'fold':fold,'identity':identity,'stage1_sha256':h(OUT/f'fold{fold}-stage1.txt'),'reference_max_abs_diff':diff})
  print('All current controls passed outerfold',fold,'seconds',round(time.perf_counter()-t,1),flush=True)
 assert len(manifests)==9 and all(m['train_shape'][0]>0 and m['query_shape'][0]>0 for m in manifests)
 assert check_protected()==protected and h(best)==plan['best_sha256']
 for p,digest in plan['sources'].items():assert h(p)==digest,p
 artifactfiles=list(OUT.glob('fold*-held.parquet'))+list(OUT.glob('fold*-roles.csv'))+list(OUT.glob('fold*-stage1.txt'))+list((OUT/'models').rglob('*.txt'))
 contextfiles=[OUT/m['context_file'] for m in manifests];write(OUT/'context-receipt.json',{'plan_sha256':h(OUT/'plan.json'),'contexts':manifests,'context_hashes':{str(p):h(p) for p in contextfiles},'artifact_hashes':{str(p):h(p) for p in artifactfiles},'seconds':time.perf_counter()-t,'fold_receipts':receipts,'max_context':max(m['train_shape'][0] for m in manifests),'max_query':max(m['query_shape'][0] for m in manifests),'max_width':max(m['width'] for m in manifests),'controls_exact':9,'held_relevance_or_metrics_read':False,'protected_unchanged':check_protected()==protected})
if __name__=='__main__':main()
