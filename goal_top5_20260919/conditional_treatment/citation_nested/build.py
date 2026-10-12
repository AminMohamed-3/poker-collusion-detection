from pathlib import Path
from collections import defaultdict
import os,sys,json,time,hashlib,joblib,numpy as np,polars as pl,lightgbm as lgb
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent; START=time.time(); CAP=900
sys.path.insert(0,str(ROOT/'src')); from conditional_action_features import extract_hand,action_class,INPUTS,card_features
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CTX=ROOT/'goal_top5_20260919/foundation_citation35/context.npz'; MP=ROOT/'Claude Version/experiments/step254_mechanical_tensor_20260917/step254_features.parquet'; REF=ROOT/'Codex Resume/experiments/tabm_citation_pilot_20260919/reference.parquet'
z=np.load(CTX,allow_pickle=False); train_ids=sorted(set(z['train_pair_ids'].tolist())); held_ids=sorted(set(z['pair_ids'].tolist())); wanted=sorted(set(train_ids+held_ids))
pairtable=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet',columns=['pair_id','table_id']); mpkeys=pl.read_parquet(MP,columns=['pair_id','hand_id']).join(pairtable,on='pair_id',validate='m:1'); train=mpkeys.filter(pl.col('pair_id').is_in(train_ids)).sort('pair_id','hand_id'); query=pl.DataFrame({'pair_id':z['pair_ids'],'hand_id':z['hand_ids']}).join(mpkeys,on=['pair_id','hand_id'],validate='1:1').sort('pair_id','hand_id')
assert train.height==10438 and train['pair_id'].n_unique()==85 and query.height==940 and query['pair_id'].n_unique()==47
audit_pools=sorted(pl.read_parquet(REF,columns=['table_id'])['table_id'].unique().to_list()); assert len(audit_pools)==104
train_pools=sorted(train['table_id'].unique().to_list()); held_pools=sorted(query['table_id'].unique().to_list()); assert len(train_pools)==71 and len(held_pools)==43 and not set(train_pools)&set(audit_pools)
labels=pl.read_csv(ROOT/'data/development_labels.csv'); assert set(labels.filter(pl.col('label')==0)['label_status'])=={'confirmed_non_target'}
pairs=labels.filter(pl.col('pair_id').is_in(wanted)).select('pair_id','player_1','player_2'); assert pairs.height==132
actors=sorted(set(pairs['player_1'])|set(pairs['player_2'])); target_pools=sorted(set(train_pools)|set(held_pools))

# Fit four clean normal-action classifiers from existing raw policy contexts.
cache=pl.read_parquet(sorted((ROOT/'reports/conditional-action-experiment/action-cache').glob('*.parquet'))).join(labels.select('pair_id','label'),on='pair_id',validate='m:1')
permitted=sorted(set(cache['table_id'].unique())-set(audit_pools)); folds={p:int(hashlib.sha256(f'ct-citation-nested-v1|{p}'.encode()).hexdigest()[:16],16)%3 for p in permitted}
params={'n_estimators':100,'learning_rate':.05,'num_leaves':15,'min_child_samples':50,'reg_lambda':10,'n_jobs':2,'verbosity':-1,'random_state':39117,'deterministic':True,'force_col_wise':True}
models={}; manifests={}
for name,scorefold in [('fold0',0),('fold1',1),('fold2',2),('final',None)]:
 pools=[p for p in permitted if scorefold is None or folds[p]!=scorefold]; d=cache.filter((pl.col('label')==0)&pl.col('table_id').is_in(pools)); d=d.with_columns(pl.Series('_sample',np.random.default_rng(55193).random(d.height))).sort('pair_id','_sample').group_by('pair_id',maintain_order=True).head(128)
 if d.height>100000:d=d.sample(n=100000,seed=55193)
 m=lgb.LGBMClassifier(**params).fit(d.select(INPUTS).to_numpy().astype(np.float32),d['action_class'].to_numpy()); path=HERE/f'{name}.joblib';joblib.dump({'classifier':m,'inputs':INPUTS},path);models[name]=m
 manifests[name]={'training_pools':pools,'training_rows':d.height,'training_pairs':d['pair_id'].n_unique(),'model_sha256':sha(path),'excluded_audit_pools':len(set(audit_pools)-set(pools)),'training_players':sorted(set(labels.filter(pl.col('pair_id').is_in(d['pair_id'].unique()))['player_1'])|set(labels.filter(pl.col('pair_id').is_in(d['pair_id'].unique()))['player_2'])),'training_hands_sha256':hashlib.sha256('\n'.join(sorted(d['hand_id'].unique())).encode()).hexdigest()}
 assert not set(pools)&set(audit_pools)

# Extract exact actual-target contexts within the required actors' complete local pools.
hands=pl.read_parquet(ROOT/'data/hands.parquet').filter((pl.col('phase')=='development')&pl.col('table_id').is_in(target_pools)).select('hand_id','table_id','board_cards','button_seat','big_blind'); ids=hands.select('hand_id')
acts=pl.scan_parquet(ROOT/'data/actions.parquet').join(ids.lazy(),on='hand_id',how='semi').collect().sort('hand_id','action_no'); seats=pl.scan_parquet(ROOT/'data/seats.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
cls=pl.when(pl.col('action')=='fold').then(0).when(pl.col('action')=='check').then(1).when((pl.col('action')=='call')|((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call')))).then(2).otherwise(3)
allphase=pl.read_parquet(ROOT/'data/hands.parquet').filter(pl.col('phase')=='development').select('hand_id'); allacts=pl.scan_parquet(ROOT/'data/actions.parquet').join(allphase.lazy(),on='hand_id',how='semi'); totals=allacts.with_columns(cls.alias('kind')).group_by('player_id').agg(*[(pl.col('kind')==k).sum().alias('n'+str(k)) for k in range(4)]).collect(); tl={r[0]:np.array(r[1:]) for r in totals.iter_rows()}
aa=acts.with_columns((pl.col('action').is_in(['bet','raise'])|((pl.col('action')=='all_in')&(pl.col('amount')>pl.col('to_call')))).alias('_agg')).with_columns(pl.when('_agg').then('player_id').otherwise(None).forward_fill().shift(1).over('hand_id','street').alias('target'))
eligible=aa.filter(pl.col('player_id').is_in(actors)&pl.col('target').is_not_null()&(pl.col('target')!=pl.col('player_id'))).select('hand_id','action_no',pl.col('player_id').alias('actor'),'target').unique(); rel=eligible.select('hand_id').unique();hands=hands.join(rel,on='hand_id',how='semi');acts=acts.join(rel,on='hand_id',how='semi');seats=seats.join(rel,on='hand_id',how='semi')
ep={(h,a):(actor,target) for h,a,actor,target in eligible.iter_rows()};hp=defaultdict(set);hs=defaultdict(list);ha=defaultdict(list)
for h,a,actor,target in eligible.iter_rows():hp[h].add((actor,target))
for r in seats.iter_rows(named=True):hs[r['hand_id']].append(r)
for r in acts.iter_rows(named=True):ha[r['hand_id']].append(r)
rows=[]
for hand in hands.iter_rows(named=True):
 h=hand['hand_id'];obs=ha[h];cur=defaultdict(lambda:np.zeros(4,dtype=int))
 for a in obs:cur[a['player_id']][action_class(a)]+=1
 styles={s['player_id']:((tl.get(s['player_id'],np.zeros(4,dtype=int))-cur[s['player_id']]+1)/((tl.get(s['player_id'],np.zeros(4,dtype=int))-cur[s['player_id']]).sum()+4)).tolist() for s in hs[h]}
 synth=[(f'{h}|{a}|{b}',a,b) for a,b in hp[h]]
 for r in extract_hand(hand,hs[h],obs,synth,styles):
  actor,target=ep.get((h,r['action_no']),(None,None))
  if actor is not None and r['pair_id']==f'{h}|{actor}|{target}' and r['facing_partner']==1:r['actor']=actor;r['target']=target;rows.append(r)
 if time.time()-START>CAP:
  (HERE/'checkpoint.json').write_text(json.dumps({'status':'time_cap_during_extraction','hands_completed':len(set(r['hand_id'] for r in rows)),'rows':len(rows),'elapsed_seconds':time.time()-START},indent=2)+'\n');raise SystemExit('15-minute cap reached; checkpoint preserved')
frame=pl.DataFrame(rows).with_columns(pl.col(INPUTS+['partner_strength','observed_log_amount_bb']).cast(pl.Float32));assert frame.select(pl.struct('hand_id','action_no','actor').n_unique()).item()==frame.height==eligible.height;frame.write_parquet(HERE/'raw_contexts.parquet');card_features.cache_clear()

# Score by clean nested lineage.
pieces=[]
for name in ['fold0','fold1','fold2','final']:
 if name=='final': part=frame.filter(pl.col('table_id').is_in(audit_pools))
 else: part=frame.filter(pl.col('table_id').is_in([p for p,f in folds.items() if f==int(name[-1])]))
 if not part.height:continue
 m=models[name];prob=m.predict_proba(part.select(INPUTS).to_numpy().astype(np.float32),num_threads=2);y=part['action_class'].to_numpy();res=(np.eye(4)[y]-prob)/np.sqrt(prob*(1-prob)+.05);score=np.clip(res[:,0],-4,4)*part['own_strength'].to_numpy()*(1-part['price'].to_numpy());pieces.append(part.select('hand_id','action_no','table_id','actor','target','street_index','own_strength','price','log_pot_bb','log_stack_bb','players_active').with_columns(pl.Series('score',score.astype(np.float32)),pl.lit(name).alias('policy')))
scored=pl.concat(pieces);assert scored.height==frame.height;scored.write_parquet(HERE/'nested_residuals.parquet')

# Training-only cutpoints and frozen per-hand features.
base=scored.filter(pl.col('table_id').is_in(train_pools));cuts={c:[float(base[c].quantile(1/3,'nearest')),float(base[c].quantile(2/3,'nearest'))] for c in ['own_strength','log_pot_bb','log_stack_bb']}
mapping=pl.concat([pairs.select('pair_id',pl.col('player_1').alias('actor'),pl.col('player_2').alias('partner')).with_columns(pl.lit(0).alias('direction')),pairs.select('pair_id',pl.col('player_2').alias('actor'),pl.col('player_1').alias('partner')).with_columns(pl.lit(1).alias('direction'))]);x=scored.join(mapping,on='actor',how='inner').with_columns((pl.col('target')==pl.col('partner')).alias('treated'))
for c,cc in cuts.items():x=x.with_columns(sum([(pl.col(c)>v).cast(pl.Int8) for v in cc],start=pl.lit(0)).alias(c+'_bin'))
x=x.with_columns(((pl.col('price')>.2).cast(pl.Int8)+(pl.col('price')>.5).cast(pl.Int8)).alias('price_bin'),(pl.col('players_active')>=3).cast(pl.Int8).alias('multi'));cell=['pair_id','direction','actor','street_index','own_strength_bin','price_bin','log_pot_bb_bin','log_stack_bb_bin','multi']
ctrl=x.filter(~pl.col('treated')).group_by(cell).agg(pl.col('score').sum().alias('cs'),pl.len().alias('cn'));ch=x.filter(~pl.col('treated')).group_by(cell+['hand_id']).agg(pl.col('score').sum().alias('hs'),pl.len().alias('hn'));t=x.filter('treated').join(ctrl,on=cell,how='left').join(ch,on=cell+['hand_id'],how='left').with_columns(pl.col('hs').fill_null(0),pl.col('hn').fill_null(0)).with_columns((pl.col('cn')-pl.col('hn')).alias('support'),(pl.col('cs')-pl.col('hs')).alias('ls')).filter(pl.col('support')>0).with_columns((pl.col('score')-pl.col('ls')/pl.col('support')).cast(pl.Float32).alias('delta'),pl.when(pl.col('street_index')==0).then(pl.lit('pre')).otherwise(pl.lit('post')).alias('phase'))
agg=t.group_by('pair_id','hand_id','phase').agg(pl.col('delta').sum().alias('sum'),pl.col('delta').max().alias('max'),pl.len().alias('count'),pl.col('support').sum().alias('support'));wide=agg.pivot(on='phase',index=['pair_id','hand_id'],values=['sum','max','count','support'],aggregate_function='first',separator='_').rename({'sum_pre':'ct_pre_sum','max_pre':'ct_pre_max','count_pre':'ct_pre_count','support_pre':'ct_pre_support','sum_post':'ct_post_sum','max_post':'ct_post_max','count_post':'ct_post_count','support_post':'ct_post_support'})
keys=pl.concat([train.select('pair_id','hand_id').with_columns(pl.lit('train').alias('split')),query.select('pair_id','hand_id').with_columns(pl.lit('held_query').alias('split'))]);nums=['ct_pre_sum','ct_pre_max','ct_pre_count','ct_pre_support','ct_post_sum','ct_post_max','ct_post_count','ct_post_support'];out=keys.join(wide,on=['pair_id','hand_id'],how='left',validate='m:1').with_columns(pl.col(nums).fill_null(0)).with_columns((pl.col('ct_pre_count')>0).cast(pl.Int8).alias('ct_pre_available'),(pl.col('ct_post_count')>0).cast(pl.Int8).alias('ct_post_available')).select('split','pair_id','hand_id','ct_pre_count','ct_pre_support','ct_pre_available','ct_post_count','ct_post_support','ct_post_available','ct_pre_sum','ct_pre_max','ct_post_sum','ct_post_max');path=HERE/'features.parquet';out.write_parquet(path)
man={'protocol_sha256':sha(HERE/'PROTOCOL.md'),'design_sha256':sha(ROOT/'Codex Resume/research/astra_collaboration_20260919/CONDITIONAL_CITATION_DESIGN.md'),'features_sha256':sha(path),'raw_contexts_sha256':sha(HERE/'raw_contexts.parquet'),'nested_residuals_sha256':sha(HERE/'nested_residuals.parquet'),'rows':out.height,'train_rows':10438,'held_rows':940,'train_pairs':85,'held_pairs':47,'train_pools':train_pools,'held_sp_pools':held_pools,'audit_excluded_pools':audit_pools,'audit_pool_count':len(audit_pools),'fold_salt':'ct-citation-nested-v1','fold_assignment':folds,'cutpoints_train_only':cuts,'models':manifests,'citation_actors':actors,'citation_actor_count':len(actors),'source_context_rows':frame.height,'scored_rows':scored.height,'elapsed_seconds':time.time()-START,'quality_read':False,'downstream_fit':False,'old_residual_fallback':False};(HERE/'manifest.json').write_text(json.dumps(man,indent=2)+'\n');print(json.dumps({k:v for k,v in man.items() if k not in ['fold_assignment','models','citation_actors','audit_excluded_pools','train_pools','held_sp_pools']},indent=2))
