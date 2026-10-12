from pathlib import Path
import os,json,hashlib,numpy as np,polars as pl
os.environ.setdefault('POLARS_MAX_THREADS','2')
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'actual_target_residuals.parquet'; CTX=ROOT/'goal_top5_20260919/foundation_citation35/context.npz'; MP=ROOT/'Claude Version/experiments/step254_mechanical_tensor_20260917/step254_features.parquet'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=np.load(CTX,allow_pickle=False); train_pairs=sorted(set(z['train_pair_ids'].tolist()))
train=pl.read_parquet(MP,columns=['pair_id','hand_id']).filter(pl.col('pair_id').is_in(train_pairs)).sort('pair_id','hand_id')
assert train.height==10438 and train['pair_id'].n_unique()==85 and train.select(pl.struct('pair_id','hand_id').n_unique()).item()==train.height
query=pl.DataFrame({'pair_id':z['pair_ids'],'hand_id':z['hand_ids']}); assert query.height==940 and query['pair_id'].n_unique()==47
keys=pl.concat([train.with_columns(pl.lit('train').alias('split')),query.with_columns(pl.lit('held_query').alias('split'))]); assert keys.select(pl.struct('split','pair_id','hand_id').n_unique()).item()==keys.height

labels=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','player_1','player_2')
mapping=pl.concat([labels.select('pair_id',pl.col('player_1').alias('actor'),pl.col('player_2').alias('partner')).with_columns(pl.lit(0).alias('direction')),labels.select('pair_id',pl.col('player_2').alias('actor'),pl.col('player_1').alias('partner')).with_columns(pl.lit(1).alias('direction'))])
s=pl.read_parquet(SRC); assert s.select(pl.struct('hand_id','action_no','actor').n_unique()).item()==s.height and s.filter(pl.col('actor')==pl.col('target')).is_empty()
x=s.join(mapping,on='actor',how='inner').with_columns((pl.col('target')==pl.col('partner')).alias('treated'))
cuts={'own_strength':[.32499998807907104,.5375000238418579],'log_pot_bb':[1.7047480344772339,2.2512917518615723],'log_stack_bb':[4.574710845947266,4.832305908203125]}
for c,cc in cuts.items(): x=x.with_columns(sum([(pl.col(c)>v).cast(pl.Int8) for v in cc],start=pl.lit(0)).alias(c+'_bin'))
x=x.with_columns(((pl.col('price')>.20).cast(pl.Int8)+(pl.col('price')>.50).cast(pl.Int8)).alias('price_bin'),(pl.col('players_active')>=3).cast(pl.Int8).alias('multi'))
cell=['pair_id','direction','actor','street_index','own_strength_bin','price_bin','log_pot_bb_bin','log_stack_bb_bin','multi']
ctrl=x.filter(~pl.col('treated')).group_by(cell).agg(pl.col('score').sum().alias('ctrl_sum'),pl.len().alias('ctrl_n'))
ctrl_hand=x.filter(~pl.col('treated')).group_by(cell+['hand_id']).agg(pl.col('score').sum().alias('hand_ctrl_sum'),pl.len().alias('hand_ctrl_n'))
t=x.filter('treated').join(ctrl,on=cell,how='left').join(ctrl_hand,on=cell+['hand_id'],how='left').with_columns(pl.col('hand_ctrl_sum').fill_null(0),pl.col('hand_ctrl_n').fill_null(0)).with_columns((pl.col('ctrl_n')-pl.col('hand_ctrl_n')).alias('support'),(pl.col('ctrl_sum')-pl.col('hand_ctrl_sum')).alias('loo_sum')).filter(pl.col('support')>0).with_columns((pl.col('score')-pl.col('loo_sum')/pl.col('support')).cast(pl.Float32).alias('delta'),pl.when(pl.col('street_index')==0).then(pl.lit('pre')).otherwise(pl.lit('post')).alias('phase'))
assert t.select(pl.struct('pair_id','direction','hand_id','action_no','actor').n_unique()).item()==t.height
agg=t.group_by('pair_id','hand_id','phase').agg(pl.col('delta').sum().alias('sum'),pl.col('delta').max().alias('max'),pl.len().alias('count'),pl.col('support').sum().alias('support'))
wide=agg.pivot(on='phase',index=['pair_id','hand_id'],values=['sum','max','count','support'],aggregate_function='first',separator='_').rename({'sum_pre':'ct_pre_sum','max_pre':'ct_pre_max','count_pre':'ct_pre_count','support_pre':'ct_pre_support','sum_post':'ct_post_sum','max_post':'ct_post_max','count_post':'ct_post_count','support_post':'ct_post_support'})
cols=['ct_pre_sum','ct_pre_max','ct_pre_count','ct_pre_support','ct_post_sum','ct_post_max','ct_post_count','ct_post_support']
out=keys.join(wide,on=['pair_id','hand_id'],how='left',validate='m:1').with_columns(pl.col(cols).fill_null(0)).with_columns((pl.col('ct_pre_count')>0).cast(pl.Int8).alias('ct_pre_available'),(pl.col('ct_post_count')>0).cast(pl.Int8).alias('ct_post_available')).select('split','pair_id','hand_id',*cols,'ct_pre_available','ct_post_available')
path=HERE/'features.parquet'; out.write_parquet(path)
def cov(part):
 p=out.filter(pl.col('split')==part); return {'rows':p.height,'pairs':p['pair_id'].n_unique(),'any_available':p.filter((pl.col('ct_pre_available')+pl.col('ct_post_available'))>0).height,'pre_available':p['ct_pre_available'].sum(),'post_available':p['ct_post_available'].sum(),'missing':p.filter((pl.col('ct_pre_available')+pl.col('ct_post_available'))==0).height,'matched_actions':p.select((pl.col('ct_pre_count')+pl.col('ct_post_count')).sum()).item(),'control_support':p.select((pl.col('ct_pre_support')+pl.col('ct_post_support')).sum()).item()}
rec={'protocol_sha256':sha(HERE/'PROTOCOL.md'),'source':str(SRC),'source_sha256':sha(SRC),'context_sha256':sha(CTX),'step254_keys_source_sha256':sha(MP),'features_sha256':sha(path),'rows':out.height,'unique_keys':out.select(pl.struct('split','pair_id','hand_id').n_unique()).item(),'feature_columns':cols+['ct_pre_available','ct_post_available'],'coverage':{'train':cov('train'),'held_query':cov('held_query')},'source_residual_rows':s.height,'treated_matched_action_rows':t.height,'formulas':{'delta':'saved score - leave-current-hand-out mean of same-actor, other-target controls in frozen medium cell','pre':'street_index == 0','post':'street_index > 0','unmatched':'numeric zero plus availability=0'},'quality_metrics_read':False,'fit_performed':False}
(HERE/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n'); print(json.dumps(rec,indent=2))
