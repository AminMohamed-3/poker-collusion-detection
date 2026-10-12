import os
os.environ['POLARS_MAX_THREADS']='2'
import sys,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
import polars as pl
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent; R=OUT.parent.parent; ROOT=R.parent
pre=R/'experiments/tabpfn3_relative_20260912/preflight.json'
j=json.loads(pre.read_text());allowed=sorted(set(j['audit_pools'])|set(j['reference_pools']));excluded=set(j['excluded_old_outer_pools']);assert not set(allowed)&excluded
start=time.perf_counter()
h=pl.scan_parquet(ROOT/'data/hands.parquet').filter((pl.col('phase')=='development')&pl.col('table_id').is_in(allowed)).select('hand_id','table_id').collect()
assert h['hand_id'].n_unique()==h.height
s=pl.scan_parquet(ROOT/'data/seats.parquet').select('hand_id','player_id').join(h.lazy(),on='hand_id',how='inner').collect()
assert s.select('hand_id','player_id').unique().height==s.height
roster=s.select('table_id','player_id').unique(); assert roster['player_id'].n_unique()==roster.height
pairs=s.lazy().rename({'player_id':'player_1'}).join(s.lazy().select('hand_id',pl.col('player_id').alias('player_2')),on='hand_id').filter(pl.col('player_1')<pl.col('player_2')).group_by('table_id','player_1','player_2').agg(pl.len().alias('shared_hands')).collect(engine='streaming')
labels=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','player_1','player_2','label','behavior_family')
pairs=pairs.join(labels,on=['player_1','player_2'],how='left',validate='1:1').with_columns(pl.col('label').fill_null(-1),pl.col('behavior_family').fill_null('unknown'))
synthetic=pl.concat_str([pl.lit('X'),pl.concat_str(['player_1','player_2']).map_elements(lambda s:hashlib.sha1(s.encode()).hexdigest()[:12].upper(),return_dtype=pl.String)])
pairs=pairs.with_columns(pl.coalesce('pair_id',synthetic).alias('pair_id')).sort('pair_id'); assert pairs['pair_id'].n_unique()==pairs.height
p=OUT/'development-pair-inventory.parquet'; assert not p.exists();pairs.write_parquet(p)
def counts(x):
 return {'pairs':x.height,'pools':x['table_id'].n_unique(),'pair_hand_observations':int(x['shared_hands'].sum()),'status_counts':x.group_by('label','behavior_family').len().sort('label','behavior_family').to_dicts()}
cache=[]
for k in [1,2]:
 x=pl.scan_parquet(str(ROOT/f'Claude Version/experiments/pu_pairs_20260911/fold{k}/*-pairs.parquet')).select('pair_id','table_id').collect()
 cache.append({'fold':k,'rows':x.height,'pools':x['table_id'].n_unique(),'allowed_pools':len(set(x['table_id'])&set(allowed)),'allowed_rows':x.filter(pl.col('table_id').is_in(allowed)).height})
report={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Inventory only, no model fits, pseudo-labels, quality metrics or new held evidence reads. Labels used solely to identify already-confirmed versus unknown pairs inside allowed pools.','allowed_pools':allowed,'allowed_pool_count':len(allowed),'old_outer_pool_overlap':len(set(allowed)&excluded),'physical_hands':h.height,'players':roster.height,'players_crossing_allowed_pools':0,'all_observed_pairs':counts(pairs),'shared_hands_ge38':counts(pairs.filter(pl.col('shared_hands')>=38)),'existing_pu_cache_overlap':cache,'warning':'The existing PU caches belong to old held-out pools and must not become training data. Their conditional features also depend on those old fold policies. Rebuild hand features within the allowed pools and the new inner fit boundary. Within each inner split exclude held pools, all their players, and physical hands from teachers, feature policy fitting, pseudo-corpus, student fitting and calibration.','sha256':{'inventory':hashlib.sha256(p.read_bytes()).hexdigest(),'preflight':hashlib.sha256(pre.read_bytes()).hexdigest(),'source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'seconds':time.perf_counter()-start}
(OUT/'inventory.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='allowed_pools'}))
