import sys,site,json
from pathlib import Path
site.addsitedir(r'work/poker/.venv/Lib/site-packages');sys.path[:0]=[r'work/poker/Claude Version/src',r'work/poker/Codex Resume/src']
import polars as pl
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.fold import fold_roles
D=S165.load(); d=D['d']; truth=d.select('pair_id','behavior_family').unique()
roles=pl.concat([fold_roles(i).with_columns(pl.lit(i).alias('fold')) for i in (0,1,2)]).unique('pair_id',keep='first')
x=truth.join(roles,on='pair_id',how='left',validate='1:1');sp=x.filter(pl.col('behavior_family')=='soft_play');r={'audit_pairs':x.height,'roles_found':x['predicted_behavior'].is_not_null().sum(),'true_sp_pairs':sp.height,'true_sp_roles_found':sp['predicted_behavior'].is_not_null().sum(),'true_sp_predicted_sp':sp.filter(pl.col('predicted_behavior')=='soft_play').height,'true_sp_mismatches':sp.filter(pl.col('predicted_behavior')!='soft_play').select('pair_id','predicted_behavior','fold').to_dicts(),'predicted_sp_total':x.filter(pl.col('predicted_behavior')=='soft_play').height,'predicted_sp_truth_counts':dict(x.filter(pl.col('predicted_behavior')=='soft_play')['behavior_family'].value_counts().iter_rows())}
Path(r'work/poker/goal_top5_20260919/foundation_citation35/routing_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
