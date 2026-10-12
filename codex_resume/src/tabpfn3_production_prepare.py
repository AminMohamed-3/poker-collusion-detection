"""Production contexts for qualified pretrained T; no evaluation outcome reads or GPU calls."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time,gc,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import polars as pl
import joblib
import contestable_state_production as prod
ROOT=prod.ROOT;HERE=prod.HERE;CV=prod.CV
OUT=HERE/'outputs_tabpfn3_default_20260913'
CONF=HERE/'experiments/tabpfn3_confirmation_20260912'
QUAL=HERE/'experiments/tabpfn3_confirmation_vm_20260912'
BEST=HERE/'outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv'
M3=prod.BEST
KEYS=['pair_id','hand_id']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,j):
 assert not p.exists(),p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
def columns():
 ctx=read(CONF/'context-receipt.json');return {f:next(m['columns'] for m in ctx['contexts'] if m['family']==f) for f in prod.FAMILIES}
def plan():
 path=OUT/'plan.json'
 if path.exists():
  p=read(path)
  for n,h in p['sources'].items():assert sha(n)==h,n
  assert prod.check_protected()==p['protected'];return p
 OUT.mkdir(exist_ok=True)
 q=read(QUAL/'results.json');assert q['passed'] and q['negative_control_rejected']
 best=pl.read_csv(BEST,infer_schema=False);m3=pl.read_csv(M3,infer_schema=False);selected=pl.read_parquet(ROOT/'reports/replay-v13/selected-pairs.parquet').select('pair_id')
 assert selected.height==16060 and selected['pair_id'].n_unique()==16060
 assert best.select('pair_id','risk_score','predicted_behavior').equals(m3.select('pair_id','risk_score','predicted_behavior'))
 assert best.join(selected,on='pair_id',how='anti').equals(m3.join(selected,on='pair_id',how='anti'))
 fixed=[Path(__file__),Path(prod.__file__),CONF/'context-receipt.json',QUAL/'results.json',QUAL/'predictions-frozen.json',BEST,M3,prod.STAGE1,prod.R382_OUT/'stage2-models.joblib',prod.PREV_OUT/'stage2-models.joblib',ROOT/'reports/replay-v13/selected-pairs.parquet',ROOT/'reports/evidence-public-calibration/selected-pairs.parquet',ROOT/'data/development_labels.csv',ROOT/'data/development_evidence.csv',prod.CONDITIONAL/'evidence-model.joblib',prod.CONDITIONAL/'policy/development-hand-features.parquet',prod.FS2/'dev-structural-features.parquet',prod.FS2/'eval_structural/receipt.json',ROOT/'reports/sequence-evidence-experiment/development.parquet',ROOT/'reports/equity-evidence-experiment/development.parquet']
 fixed += [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT.resolve())) and Path(m.__file__).suffix=='.py' and 'site-packages' not in str(m.__file__)]
 p={'created_utc':datetime.now(timezone.utc).isoformat(),'candidate':'qualified pretrained TabPFN3 default, all3 families, exact original features','qualification_comparator':'M3_stage2_r382DTSP .89447; currentbest .89457 differs only in replaced evidence','source_public_best':.89457,'coverage':'exact existing frozen16060pairs; all112540submission rows; no new risk/behavior gate','columns':columns(),'sources':{str(x):sha(x) for x in sorted(set(fixed))},'protected':prod.check_protected(),'training_recipe':'Full372 development targets; fixed5 pool-hash innerOOF stage1, true-family top20 contexts. Evaluation predicted-family top20. Structural pot_bb exact float64 from qualified confirmation view.','inference_params':read(QUAL/'freeze.json')['params'],'no_finetuning':True,'no_unqualified_specialist_or_contestable_features':True,'no_submission_actions':True}
 write(path,p);return p

def tfm_view(frame):
 assert np.array_equal(frame['pot_bb'].to_numpy().astype(np.float32),frame['pot_bb_right'].to_numpy().astype(np.float32))
 return frame.with_columns(pl.col('pot_bb_right').alias('pot_bb'),pl.col('s1').alias('oof'),pl.col('s1_rank').cast(pl.Float64).alias('oof_rank'))

def train_contexts():
 p=plan();assert not (OUT/'training-receipt.json').exists();started=time.monotonic()
 data,chrono,struct,r382,overlap=prod.development_frame()
 assert data.height==45129 and data['pair_id'].n_unique()==372
 rb=joblib.load(prod.R382_OUT/'stage2-models.joblib');pb=joblib.load(prod.PREV_OUT/'stage2-models.joblib');controls={};items=[]
 for fam in prod.FAMILIES:
  cols=pb['cols']['chrono3'] if fam==prod.CI else rb['cols']['r382']
  models=pb['models'][f'chrono3/{fam}'] if fam==prod.CI else rb['models'][f'r382/{fam}']
  refit,_,_=prod.stage2_fit(data,cols,fam,prod.CLF)
  top=data.filter((pl.col('behavior_family')==fam)&(pl.col('s1_rank')<=20))
  np.testing.assert_array_equal(prod.stage2_predict(refit,top,cols),prod.stage2_predict(models,top,cols));controls[fam]=True
  view=tfm_view(top);cols=p['columns'][fam];matrix=view.select(cols).fill_null(-999.).to_numpy().astype(np.float64)
  assert np.isfinite(matrix).all() and matrix.shape[1]==(75 if fam==prod.CI else 453)
  path=OUT/'training'/f'{fam}.npz';path.parent.mkdir(exist_ok=True);assert not path.exists()
  np.savez_compressed(path,xtrain=matrix,ytrain=view['relevance'].to_numpy().astype(np.int64),groups_train=view['table_id'].to_numpy().astype('U'),pair_ids=view['pair_id'].to_numpy().astype('U'))
  items.append({'family':fam,'file':str(path.relative_to(OUT)).replace('\\','/'),'sha256':sha(path),'shape':list(matrix.shape),'pairs':view['pair_id'].n_unique(),'pools':view['table_id'].n_unique(),'class_counts':np.bincount(view['relevance'].to_numpy().astype(int),minlength=2).tolist()})
 write(OUT/'training-receipt.json',{'plan_sha256':sha(OUT/'plan.json'),'contexts':items,'production_stage1_exact':True,'production_hybrid_controls':controls,'struct_overlap':overlap,'all_labels_used_for_final_fit_only':True,'no_quality_metric':True,'seconds':time.monotonic()-started})
 print(json.dumps({'training':items,'seconds':time.monotonic()-started}),flush=True)

def prepare_batches():
 p=plan();tr=read(OUT/'training-receipt.json');assert tr['plan_sha256']==sha(OUT/'plan.json')
 source=pl.read_csv(M3,infer_schema=False);selected=pl.read_parquet(ROOT/'reports/replay-v13/selected-pairs.parquet').select('pair_id')
 added=selected.join(pl.read_parquet(ROOT/'reports/evidence-public-calibration/selected-pairs.parquet').select('pair_id'),on='pair_id',how='anti')
 rb=joblib.load(prod.R382_OUT/'stage2-models.joblib');pb=joblib.load(prod.PREV_OUT/'stage2-models.joblib');chrono=joblib.load(prod.STAGE1);caf=joblib.load(prod.CONDITIONAL/'evidence-model.joblib')['conditional_features']
 sr=read(prod.FS2/'eval_structural/receipt.json');eval_started=prod.started_at('evaluation');allmeta=[]
 for index,path in enumerate(sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet'))):
  name=path.stem;dest=OUT/'queries'/name;receipt=dest/'receipt.json'
  if receipt.exists():
   old=read(receipt);assert old['plan_sha256']==sha(OUT/'plan.json')
   for n,h in old['files'].items():assert sha(dest/n)==h
   allmeta.append(old);continue
  t=time.monotonic();dest.mkdir(parents=True,exist_ok=True);inputs={str(path):sha(path)}
  raw=pl.read_parquet(path);events=[c for c in raw.columns if c.startswith('v2_')]
  ep=ROOT/'reports/evidence-public-calibration'/(name+'-features.parquet');eq=pl.read_parquet(ep);inputs[str(ep)]=sha(ep)
  cp=prod.PACKAGE/'results'/(name+'-evidence.parquet');ca=pl.read_parquet(cp).drop('table_id');inputs[str(cp)]=sha(cp)
  for directory in [prod.PACKAGE/'results',ROOT/'outputs_v13_coverage/batches']:
   rp=directory/(name+'.json');inputs[str(rp)]=sha(rp)
   for n,h in read(rp)['files'].items():assert sha(directory/n)==h
  extra=pl.concat([eq.join(ca,on=KEYS,validate='1:1').select(KEYS+prod.SEQ+prod.EQ+caf),pl.read_parquet(ROOT/'outputs_v13_coverage/batches'/(name+'-features.parquet')).join(added,on='pair_id',how='semi').select(KEYS+prod.SEQ+prod.EQ+caf)])
  sp=prod.FS2/'eval_structural'/(name+'.parquet');assert sha(sp)==sr['files'][sp.name];sf=pl.read_parquet(sp);inputs[str(sp)]=sha(sp)
  pools=sorted(raw['table_id'].unique());frames=[];ranked=[]
  for offset in range(0,len(pools),5):
   part=raw.filter(pl.col('table_id').is_in(pools[offset:offset+5])).sort('pair_id','hand_id');part=part.with_columns(pl.Series('tie',np.random.default_rng(931207+index*100+offset).random(part.height))).join(selected,on='pair_id',how='semi')
   if not part.height:continue
   frame=part.join(extra,on=KEYS,validate='1:1').join(source.select('pair_id','predicted_behavior'),on='pair_id',validate='m:1').sort('pair_id','hand_id');assert frame.height==part.height
   frame=prod.add_chronology(prod.family_columns(prod.context(frame,events+prod.SEQ+prod.EQ+caf),'predicted_behavior'),eval_started).sort('pair_id','hand_id')
   s1=chrono['model'].predict(prod.matrix(frame,chrono['features']),num_threads=2)
   frame=prod.with_rank(frame.with_columns(pl.Series('s1',s1)),'s1','s1_rank').join(sf,on=KEYS,how='left',validate='1:1').sort('pair_id','hand_id')
   top=(frame['s1_rank']<=20).to_numpy();family=frame['predicted_behavior'].to_numpy();scores=s1-1e6
   for fam in prod.FAMILIES:
    mask=top&(family==fam)
    if mask.any():
     cols=pb['cols']['chrono3'] if fam==prod.CI else rb['cols']['r382'];models=pb['models'][f'chrono3/{fam}'] if fam==prod.CI else rb['models'][f'r382/{fam}']
     scores[mask]=prod.stage2_predict(models,frame.filter(pl.Series(mask)),cols)+10
   assert np.isfinite(scores).all();ranked.append(prod.rank(frame,scores));frames.append(tfm_view(frame.filter(pl.Series(top))))
  oldlists=pl.concat(ranked).sort('pair_id').select('pair_id',*[pl.col('hands').list.get(i,null_on_oob=True).fill_null('NO_EVIDENCE').alias(c) for i,c in enumerate(prod.EVIDENCE_COLUMNS)])
  assert oldlists.equals(source.join(oldlists.select('pair_id'),on='pair_id',how='semi').select(oldlists.columns).sort('pair_id'))
  oldcache=prod.R382_OUT/'batches'/(name+'-r382.parquet');assert oldlists.equals(pl.read_parquet(oldcache).select(oldlists.columns).sort('pair_id'));inputs[str(oldcache)]=sha(oldcache)
  topframe=pl.concat(frames).sort('pair_id','hand_id');assert topframe.select(KEYS).n_unique()==topframe.height
  fp=dest/'keys.parquet';topframe.select('pair_id','hand_id','table_id','predicted_behavior','tie','s1_rank').write_parquet(fp);files={fp.name:sha(fp)};items=[]
  for fam in prod.FAMILIES:
   ix=np.flatnonzero((topframe['predicted_behavior']==fam).to_numpy());x=topframe[ix].select(p['columns'][fam]).fill_null(-999.).to_numpy().astype(np.float64);assert np.isfinite(x).all()
   f=dest/(fam+'.npz');np.savez_compressed(f,xtest=x,indices=ix.astype(np.int64),pair_ids=topframe['pair_id'].to_numpy()[ix].astype('U'));files[f.name]=sha(f)
   items.append({'family':fam,'file':str(f.relative_to(OUT)).replace('\\','/'),'sha256':sha(f),'shape':list(x.shape)})
  meta={'batch':name,'plan_sha256':sha(OUT/'plan.json'),'files':files,'inputs':inputs,'contexts':items,'rows':topframe.height,'pairs':topframe['pair_id'].n_unique(),'m3_hybrid_lists_exact':True,'seconds':time.monotonic()-t};write(receipt,meta);allmeta.append(meta)
  print(json.dumps({k:meta[k] for k in ['batch','rows','pairs','m3_hybrid_lists_exact','seconds']}),flush=True)
  del raw,extra,sf,frames,topframe;gc.collect()
 assert len(allmeta)==20 and sum(m['pairs'] for m in allmeta)==16060
 assert prod.check_protected()==p['protected'];plan()
 write(OUT/'queries-complete.json',{'plan_sha256':sha(OUT/'plan.json'),'training_receipt_sha256':sha(OUT/'training-receipt.json'),'batch_receipts':{m['batch']:sha(OUT/'queries'/m['batch']/'receipt.json') for m in allmeta},'pairs':16060,'rows':sum(m['rows'] for m in allmeta),'all20_controls_exact':True,'no_quality_metrics_or_TFM_inference':True})

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['train','queries']);args=a.parse_args();{'train':train_contexts,'queries':prepare_batches}[args.mode]()
