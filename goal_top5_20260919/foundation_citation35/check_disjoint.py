import sys,site,json
from pathlib import Path
site.addsitedir(r'work/poker/.venv/Lib/site-packages')
sys.path[:0]=[r'work/poker/Claude Version/src',r'work/poker/Codex Resume/src']
import polars as pl
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug import step254_eval as S254
D=S165.load(); audit=set(D['d']['pair_id'].unique()); h,_=S254.load(); labels=pl.read_csv(r'work/poker/data/development_labels.csv')
sp=set(labels.filter((pl.col('behavior_family')=='soft_play')&(~pl.col('pair_id').is_in(list(audit))))['pair_id']);
train_tables=set(h.filter(pl.col('pair_id').is_in(list(sp)))['table_id']);audit_tables=set(h.filter(pl.col('pair_id').is_in(list(audit)))['table_id'])
train_players=set(labels.filter(pl.col('pair_id').is_in(list(sp))).select('player_1','player_2').to_numpy().ravel());audit_players=set(labels.filter(pl.col('pair_id').is_in(list(audit))).select('player_1','player_2').to_numpy().ravel())
r={'train_sp_pairs':len(sp),'audit_pairs':len(audit),'train_tables':len(train_tables),'audit_tables':len(audit_tables),'table_overlap_count':len(train_tables&audit_tables),'table_overlap':sorted(train_tables&audit_tables),'train_players':len(train_players),'audit_players':len(audit_players),'player_overlap_count':len(train_players&audit_players),'player_overlap':sorted(train_players&audit_players)}
Path(r'work/poker/goal_top5_20260919/foundation_citation35/disjointness.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
