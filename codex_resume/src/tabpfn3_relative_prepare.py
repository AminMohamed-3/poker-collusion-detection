"""Pool-isolated T versus T+RZ selection: no-fit preflight, then frozen context preparation."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
import sys,json,time,hashlib,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import polars as pl
import lightgbm as lgb
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent;CV=ROOT/'Claude Version'
sys.path.insert(0,str(CV/'src'))
from evidence_aug.frame import CACHES,assemble,training_relevance,attach_relevance,family_columns,score,FAMILIES,PARAMS
from evidence_aug.fold import fold_pools
from evidence_aug.paths import check_protected,sha
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step10_chronology_fit import add_chronology
from evidence_aug.step14_audit_truncation import CLF
sys.path.insert(0,str(ROOT/'src'))
from run_conditional_policy_screen import hand_features
OUT=HERE/'experiments/tabpfn3_relative_20260912'
PROTOCOL=HERE/'research/tabpfn3_relative_protocol_20260912.md'
STRUCT=CV/'experiments/family_stage2_20260911/dev-structural-features.parquet'
POLICY=ROOT/'reports/conditional-action-experiment'
SALTS=[f'codex-tfm-rz-20260912-{s}:' for s in 'abc']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
 p=Path(p);assert not p.exists(),f'write-once: {p}';p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def split(pool,salt):return int(hashlib.sha256((salt+str(pool)).encode()).hexdigest(),16)%5
def canonical(mapping):
 buckets=[tuple(sorted(k for k,v in mapping.items() if v==i)) for i in range(5)]
 return tuple(sorted(buckets))
def raw_keys():return pl.read_parquet(sorted(CACHES['raw'].glob('batch_*.parquet')),columns=['pair_id','hand_id','table_id'])
def labels():return pl.read_csv(ROOT/'data/development_labels.csv')
def policy_file(n):
 local=POLICY/n
 if local.exists():return local
 return {'development_labels.csv':ROOT/'data/development_labels.csv','held-pairs.csv':ROOT/'reports/tabicl-screen/held-pairs.csv','run_conditional_policy_screen.py':ROOT/'src/run_conditional_policy_screen.py'}[n]

def sample_policy(train):
 train=train.sort('pair_id','hand_id','direction','action_no');rng=np.random.default_rng(55193)
 train=train.with_columns(pl.Series('sample_order',rng.random(train.height))).sort('pair_id','sample_order').group_by('pair_id',maintain_order=True).head(128)
 return train.sample(n=100000,seed=55193) if train.height>100000 else train
def action_data():return pl.read_parquet(sorted((POLICY/'action-cache').glob('batch_*.parquet'))).sort('pair_id','hand_id','direction','action_no')
def protected():
 x=check_protected()
 for p in [CV/'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv',HERE/'outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv']:x[str(p)]=sha(p)
 return x

def preflight():
 OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'preflight.json').exists()
 policy=read(POLICY/'policy-plan.json')
 for n,h in policy['files'].items():assert sha(policy_file(n))==h,n
 keys=raw_keys();lab=labels();excluded=set().union(*(set(fold_pools(i)) for i in range(3)))
 target=keys.join(lab.filter(pl.col('label')==1).select('pair_id','behavior_family'),on='pair_id',how='inner').filter(~pl.col('table_id').is_in(sorted(excluded)))
 ap=set(target['table_id']);rp=set(keys['table_id'])-excluded-ap
 assert len(ap)==104 and target['pair_id'].n_unique()==162 and len(rp)==65
 ref=keys.filter(pl.col('table_id').is_in(sorted(rp))).select('pair_id','table_id').unique().join(lab,on='pair_id')
 assert ref.height==248 and set(ref['label'])=={0} and set(ref['label_status'])=={'confirmed_non_target'}
 action=action_data();ra=action.filter(pl.col('table_id').is_in(sorted(rp)));aa=action.filter(pl.col('table_id').is_in(sorted(ap)))
 assert not set(ra['pair_id'])&set(aa['pair_id']) and not set(ra['hand_id'])&set(aa['hand_id'])
 sampled=sample_policy(ra.join(lab.select('pair_id','label','label_status'),on='pair_id',validate='m:1'))
 assert set(sampled['action_class'])=={0,1,2,3}
 assert np.isfinite(sampled.select(policy['inputs']+['partner_strength']).to_numpy()).all()
 scored_actions=aa.join(target.select('pair_id').unique(),on='pair_id',how='semi')
 assert np.isfinite(scored_actions.select(policy['inputs']+['partner_strength']).to_numpy()).all()
 assert scored_actions.select('pair_id','hand_id').unique().sort('pair_id','hand_id').equals(target.select('pair_id','hand_id').unique().sort('pair_id','hand_id'))
 # Check all seated players, not just the labelled pair: this also validates public player-style isolation.
 seats=pl.read_parquet(ROOT/'data/seats.parquet',columns=['hand_id','player_id'])
 refplayers=set(seats.join(ra.select('hand_id').unique(),on='hand_id',how='semi')['player_id'])
 auditplayers=set(seats.join(aa.select('hand_id').unique(),on='hand_id',how='semi')['player_id'])
 assert not refplayers&auditplayers
 all_dev=pl.read_parquet(ROOT/'data/hands.parquet',columns=['hand_id','table_id','phase']).filter(pl.col('phase')=='development')
 actors=pl.scan_parquet(ROOT/'data/actions.parquet').select('hand_id','player_id').join(all_dev.select('hand_id').lazy(),on='hand_id',how='semi').unique().collect()
 assert actors.join(seats,on=['hand_id','player_id'],how='anti').height==0
 playerpool=seats.join(all_dev.select('hand_id','table_id'),on='hand_id',how='inner').select('player_id','table_id').unique()
 cross=playerpool.group_by('player_id').len().filter(pl.col('len')>1)
 assert cross.height==0,'style may mix pools; investigate before fitting'
 q=target.select('pair_id','table_id','behavior_family').unique();parts=[];canonical_maps=[]
 for s,salt in enumerate(SALTS):
  mapping={p:split(p,salt) for p in sorted(ap)};cm=canonical(mapping);assert cm not in canonical_maps;canonical_maps.append(cm)
  checks=[]
  for f in range(5):
   tr=q.filter(pl.col('table_id').is_in([p for p,v in mapping.items() if v!=f]));te=q.filter(pl.col('table_id').is_in([p for p,v in mapping.items() if v==f]))
   assert set(tr['behavior_family'])==set(te['behavior_family'])==set(FAMILIES)
   inner_salt=salt+f'outer{f}:inner:';im={p:split(p,inner_salt) for p in sorted(set(tr['table_id']))}
   for i in range(5):
    a=tr.filter(pl.col('table_id').is_in([p for p,v in im.items() if v!=i]));b=tr.filter(pl.col('table_id').is_in([p for p,v in im.items() if v==i]));assert a.height and b.height and set(a['behavior_family'])==set(FAMILIES)
   checks.append({'fold':f,'train_queries':tr.height,'held_queries':te.height,'held_family_counts':te.group_by('behavior_family').len().to_dicts(),'inner_mapping':im})
  parts.append({'salt_index':s,'salt':salt,'mapping':mapping,'folds':checks})
 # Hash all feature inputs and imported project code. No metrics or evidence truth have been loaded here.
 inputs=[policy_file(n) for n in policy['files']]+sorted(CACHES['raw'].glob('batch_*.parquet'))+[CACHES['sequence'],CACHES['equity'],STRUCT,POLICY/'policy-plan.json',ROOT/'data/development_labels.csv',ROOT/'data/development_evidence.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/actions.parquet']+sorted((POLICY/'action-cache').glob('batch_*.parquet'))
 sources={Path(__file__).resolve(),PROTOCOL,HERE/'src/tabpfn3_relative_vm.py',HERE/'src/tabpfn3_relative_evaluate.py',HERE/'src/tabpfn3_hardware_resume.py'}
 sources|={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT.resolve())) and Path(m.__file__).suffix=='.py' and 'site-packages' not in str(m.__file__)}
 receipt={'utc':datetime.now(timezone.utc).isoformat(),'excluded_old_outer_pools':sorted(excluded),'audit_pools':sorted(ap),'reference_pools':sorted(rp),'audit_queries':162,'reference_pairs':248,'reference_actions':ra.height,'sampled_reference_actions':sampled.height,'sampled_reference_class_counts':sampled.group_by('action_class').len().to_dicts(),'source_action_actor_missing_seat':0,'complete_audit_action_coverage':True,'reference_physical_hands':ra['hand_id'].n_unique(),'reference_audit_seated_player_overlap':0,'development_players_crossing_pools':0,'partitions':parts,'inputs':{str(p):sha(p) for p in inputs},'sources':{str(p):sha(p) for p in sources},'protected':protected(),'no_fit':True,'no_evidence_truth_read':True,'new_outer_metrics_read':False,'policy_file_locations':{n:str(policy_file(n)) for n in policy['files']},'policy_parameters':dict(read(POLICY/'policy-plan.json')['parameters'],n_jobs=2),'ranker_parameters':PARAMS,'lgbm_parameters':dict(CLF,random_state=11)}
 write(OUT/'preflight.json',receipt)
 print(json.dumps({k:receipt[k] for k in ['audit_queries','reference_pairs','reference_actions','reference_audit_seated_player_overlap','development_players_crossing_pools','no_fit']}),flush=True)

def check():
 p=read(OUT/'preflight.json')
 for path,h in {**p['inputs'],**p['sources']}.items():assert sha(path)==h,path
 assert protected()==p['protected'];return p

def relativise(d,cols):
 expr=[]
 for c in cols:
  v=pl.when(pl.col('oof_rank')<=20).then(pl.col(c)).otherwise(None)
  expr.extend([v.rank('average').over('pair_id').fill_null(0.).alias(c+'__wr'),((v-v.mean().over('pair_id'))/(v.std().over('pair_id')+1e-9)).fill_null(0.).alias(c+'__wz')])
 return d.with_columns(expr)

def make_frame(p):
 start=time.perf_counter();action=action_data();lab=labels();policy=read(POLICY/'policy-plan.json');inputs=policy['inputs']
 train=action.filter(pl.col('table_id').is_in(p['reference_pools'])).join(lab.select('pair_id','label','label_status'),on='pair_id',validate='m:1')
 assert set(train['label'])=={0} and set(train['label_status'])=={'confirmed_non_target'}
 train=sample_policy(train);assert train.height==p['sampled_reference_actions']
 target=lab.filter(pl.col('label')==1).select('pair_id','behavior_family')
 scored=action.filter(pl.col('table_id').is_in(p['audit_pools'])).join(target.select('pair_id'),on='pair_id',how='semi')
 x=train.select(inputs).to_numpy().astype(np.float32);z=scored.select(inputs).to_numpy().astype(np.float32)
 clf=lgb.LGBMClassifier(**p['policy_parameters']).fit(x,train['action_class'].to_numpy());reg=lgb.LGBMRegressor(**p['policy_parameters']).fit(x,train['partner_strength'].to_numpy())
 assert np.array_equal(clf.classes_,np.arange(4));ca=hand_features(scored,clf.predict_proba(z),reg.predict(z));ca.write_parquet(OUT/'normal-reference-features.parquet')
 clf.booster_.save_model(str(OUT/'normal-action-model.txt'));reg.booster_.save_model(str(OUT/'normal-partner-model.txt'))
 raw=pl.read_parquet(sorted(CACHES['raw'].glob('batch_*.parquet'))).filter(pl.col('table_id').is_in(p['audit_pools'])).join(target.select('pair_id'),on='pair_id',how='semi')
 d,features,caf=assemble(raw,pl.read_parquet(CACHES['sequence']),pl.read_parquet(CACHES['equity']),ca,target)
 struct=pl.read_parquet(STRUCT);sc=[c for c in struct.columns if c not in ['pair_id','hand_id']];assert len(sc)==70
 d=add_chronology(d).join(struct,on=['pair_id','hand_id'],validate='1:1').sort('pair_id','hand_id')
 assert d.height==19325 and d['pair_id'].n_unique()==162
 write(OUT/'normal-reference-receipt.json',{'preflight_sha256':sha(OUT/'preflight.json'),'training_actions':train.height,'training_pools':p['reference_pools'],'scored_pools':p['audit_pools'],'columns':inputs,'outputs':{n:sha(OUT/n) for n in ['normal-reference-features.parquet','normal-action-model.txt','normal-partner-model.txt']},'seconds':time.perf_counter()-start,'no_audit_labels_or_actions_used_to_fit':True})
 d.write_parquet(OUT/'audit-feature-frame.parquet');return d,features,sc

def prepare():
 p=check();assert not (OUT/'prepare-started.json').exists();write(OUT/'prepare-started.json',{'preflight_sha256':sha(OUT/'preflight.json'),'utc':datetime.now(timezone.utc).isoformat()})
 (OUT/'contexts').mkdir();(OUT/'frames').mkdir();(OUT/'models').mkdir();start=time.perf_counter();d,features,struct=make_frame(p)
 shared=struct+['chrono_rank','chrono_frac','n_hands','oof','oof_rank'];extra=[x for x in features if not x.startswith('family_') and x not in struct];relative=struct+['chrono_rank','chrono_frac','n_hands'];rcols=[c+s for c in relative for s in ['__wr','__wz']]
 assert len(shared)==75 and len(extra)==378 and len(rcols)==146
 manifests=[];frame_receipts=[];nested=[]
 for partition in p['partitions']:
  s=partition['salt_index'];mapping=partition['mapping']
  for spec in partition['folds']:
   f=spec['fold'];stem=f's{s}-f{f}';heldp=[pool for pool,v in mapping.items() if v==f]
   tr=d.filter(~pl.col('table_id').is_in(heldp));he=d.filter(pl.col('table_id').is_in(heldp));tr=attach_relevance(tr,training_relevance(set(tr['table_id']))).sort('pair_id','hand_id')
   assert not set(tr['table_id'])&set(he['table_id'])
   stage=fit_ranker(tr,features+['chrono_frac']);sh=score(stage,family_columns(he,'behavior_family'),features+['chrono_frac']);stage.booster_.save_model(str(OUT/'models'/(stem+'-stage1.txt')))
   st=np.full(tr.height,np.nan);spl=np.array([spec['inner_mapping'][pool] for pool in tr['table_id']]);inner_receipts=[]
   for inner in range(5):
    ix=np.flatnonzero(spl==inner);it=tr.filter(pl.Series(spl!=inner));ih=tr[ix];assert not set(it['table_id'])&set(ih['table_id'])
    model=fit_ranker(it,features+['chrono_frac']);st[ix]=score(model,family_columns(ih,'behavior_family'),features+['chrono_frac'])
    modelpath=OUT/'models'/(stem+f'-inner{inner}.txt');model.booster_.save_model(str(modelpath));inner_receipts.append({'inner':inner,'train_pools':sorted(set(it['table_id'])),'score_pools':sorted(set(ih['table_id'])),'model_sha256':sha(modelpath)})
   assert np.isfinite(st).all()
   frames=[]
   for z,pred in [(tr,st),(he,sh)]:
    assert np.array_equal(z['pot_bb'].to_numpy().astype(np.float32),z['pot_bb_right'].to_numpy().astype(np.float32))
    z=z.with_columns(pl.Series('oof',pred),pl.col('pot_bb_right').alias('pot_bb'),pl.col(extra).fill_null(-999.)).with_columns(pl.col('oof').rank('ordinal',descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
    frames.append(relativise(z,relative))
   tr,he=frames;control=sh.copy()-1e6
   for fam in FAMILIES:
    cols=shared+(extra if fam!='coordinated_isolation' else []);t=tr.filter((pl.col('behavior_family')==fam)&(pl.col('oof_rank')<=20));ix=np.flatnonzero(((he['behavior_family']==fam)&(he['oof_rank']<=20)).to_numpy());q=he[ix]
    assert len(ix)>0 and set(t['relevance'])=={0,1}
    x=t.select(cols).to_numpy().astype(np.float64);xt=q.select(cols).to_numpy().astype(np.float64);y=t['relevance'].to_numpy().astype(np.int64)
    ref=lgb.LGBMClassifier(**p['lgbm_parameters']).fit(x,y);control[ix]=ref.predict_proba(xt)[:,1]+10;ref.booster_.save_model(str(OUT/'models'/(stem+'-'+fam+'-lgbm.txt')))
    for arm in ['T','RZ']:
     used=cols+(rcols if arm=='RZ' else []);xx=t.select(used).to_numpy().astype(np.float64);xz=q.select(used).to_numpy().astype(np.float64)
     assert np.isfinite(xx).all() and np.isfinite(xz).all() and np.array_equal(xx[:,:len(cols)],x) and np.array_equal(xz[:,:len(cols)],xt)
     name=stem+'-'+fam+'-'+arm;path=OUT/'contexts'/(name+'.npz')
     np.savez_compressed(path,xtrain=xx,ytrain=y,xtest=xz,held_indices=ix.astype(np.int64),held_pair_ids=q['pair_id'].to_numpy().astype(str),groups_train=t['table_id'].to_numpy().astype(str),groups_held=q['table_id'].to_numpy().astype(str))
     manifests.append({'stem':name,'frame':stem,'salt':s,'fold':f,'family':fam,'arm':arm,'columns':used,'context_file':'contexts/'+path.name,'context_sha256':sha(path),'train_shape':list(xx.shape),'query_shape':list(xz.shape),'class_counts':np.bincount(y,minlength=2).tolist()})
   hp=OUT/'frames'/(stem+'.parquet');he.select('pair_id','hand_id','table_id','behavior_family','oof','oof_rank').with_columns(pl.Series('lgbm',control)).write_parquet(hp);frame_receipts.append({'stem':stem,'salt':s,'fold':f,'file':'frames/'+hp.name,'sha256':sha(hp)})
   nested.append({'frame':stem,'outer_train_pools':sorted(set(tr['table_id'])),'outer_held_pools':sorted(set(he['table_id'])),'inner':inner_receipts});print(json.dumps({'prepared':stem,'contexts':len(manifests),'seconds':round(time.perf_counter()-start,2)}),flush=True)
 assert len(manifests)==90;check()
 write(OUT/'context-receipt.json',{'preflight_sha256':sha(OUT/'preflight.json'),'normal_reference_receipt_sha256':sha(OUT/'normal-reference-receipt.json'),'contexts':manifests,'frames':frame_receipts,'nested':nested,'max_context':max(m['train_shape'][0] for m in manifests),'max_query':max(m['query_shape'][0] for m in manifests),'max_width':599,'artifact_hashes':{str(x.relative_to(OUT)).replace('\\','/'):sha(x) for x in list((OUT/'models').glob('*.txt'))+[OUT/'audit-feature-frame.parquet']},'seconds':time.perf_counter()-start,'held_labels_omitted':True,'old_outer_truth_or_metrics_read':False,'protected_unchanged':True})
 print(json.dumps({'context_preparation_complete':True,'seconds':time.perf_counter()-start}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['preflight','prepare']);args=a.parse_args();globals()[args.mode]()
