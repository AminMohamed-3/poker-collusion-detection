import site,json,hashlib
from pathlib import Path
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import polars as pl, numpy as np
R=Path(r'C:/Data/Projects/Money Making/work/poker'); OUT=R/'goal_top5_20260919/foundation_pair/raw_semantics_audit'; OUT.mkdir(parents=True,exist_ok=True)
A=R/'data/actions.parquet';H=R/'data/hands.parquet';S=R/'data/seats.parquet';LAB=R/'data/development_labels.csv';EV=R/'data/evaluation_pairs.csv';SCR=R/'goal_top5_20260919/foundation_pair/screen'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selected():
 out={}
 for f in (0,1):out[str(f)]=np.load(SCR/f'context_fold{f}.npz')['feature_name'].tolist()
 return out
def phase_stats(ph,pairfile):
 h=pl.scan_parquet(H).filter(pl.col('phase')==ph).select('hand_id','table_id','big_blind')
 a=(pl.scan_parquet(A).join(h,on='hand_id').with_columns(((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call'))).alias('ai_call'),((pl.col('action')=='all_in')&(pl.col('amount')>pl.col('to_call'))).alias('ai_aggr')))
 stats=a.select(pl.len().alias('actions'),(pl.col('action')=='call').sum().alias('explicit_calls'),pl.col('ai_call').sum().alias('allin_calls'),pl.col('ai_aggr').sum().alias('allin_aggressive'),(pl.col('action')=='all_in').sum().alias('allin_total'),pl.col('hand_id').filter(pl.col('ai_call')).n_unique().alias('hands_with_allin_call'),pl.col('player_id').filter(pl.col('ai_call')).n_unique().alias('players_with_allin_call'),pl.col('big_blind').min().alias('min_big_blind'),pl.col('big_blind').max().alias('max_big_blind'),pl.col('players_active').min().alias('min_players_active'),pl.col('players_active').max().alias('max_players_active')).collect().row(0,named=True)
 affected=a.filter(pl.col('ai_call')).select('hand_id','table_id','player_id').unique()
 p=pl.scan_csv(pairfile).select('pair_id','player_1','player_2')
 seats=pl.scan_parquet(S).select('hand_id','player_id')
 # Exact shared-hand exposure: one pair member made a calling all-in and the other was seated.
 q1=(p.join(affected.rename({'player_id':'player_1'}),on='player_1',how='inner')
       .join(seats.rename({'player_id':'player_2'}),on=['hand_id','player_2'],how='semi').select('pair_id'))
 q2=(p.join(affected.rename({'player_id':'player_2'}),on='player_2',how='inner')
       .join(seats.rename({'player_id':'player_1'}),on=['hand_id','player_1'],how='semi').select('pair_id'))
 n=pl.concat([q1.collect(),q2.collect()]).unique().height
 stats['allin_call_share_of_call_semantic_actions']=stats['allin_calls']/(stats['explicit_calls']+stats['allin_calls'])
 stats['pairs_with_shared_hand_allin_call']=n;stats['pairs_total']=pl.read_csv(pairfile).height;stats['pair_share']=n/stats['pairs_total']
 return stats
sel=selected(); touched={f:[x for x in names if x.startswith('n_call_')] for f,names in sel.items()}
out={'verdict':'CONCRETE_BASE_FEATURE_DEFECT','defect':{'field':'n_call','current':'counts action == call only','correct_consistent_semantics':'counts explicit call plus nonaggressive all_in (amount <= to_call)','why':'official/raw convention makes all_in either call or aggression; call_bb, v2 call features, conditional code, and later pair_diagnostic already use the correct split'},'selected_top128_touched':touched,'selected_top128_touched_union':sorted(set(sum(touched.values(),[]))),'label_blind_support':{'development':phase_stats('development',LAB),'evaluation':phase_stats('evaluation',EV)},'other_semantics':{'all_in_aggression':'correct: amount > to_call in base and v2','amount':'correctly treated as incremental chips; amount_to is cumulative street contribution (independent replay receipt)','big_blind':'all chip magnitudes divide by per-hand big_blind; raw minimum reported and no zero fallback in builder','action_order':'sorted by hand_id, action_no; previous_aggressor is prior aggressive actor within hand and street','players_active':'raw pre-action count including actor and nonfolded all-in players; used directly/minimum, no off-by-one transform'},'limitations':['label-blind counts only; no quality read','does not establish corrected model gain','only selected n_call aggregate dependencies are identified as materially exposed'],'hashes':{str(p):sha(p) for p in [A,H,S,LAB,EV,R/'Claude Version/src/features.py',R/'Claude Version/src/advanced_features.py',R/'Claude Version/src/pu_pairs/build_allpairs_fold.py',R/'Claude Version/src/pu_pairs/step31_crossphase_features.py']}}
(OUT/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'touched':touched,'support':out['label_blind_support']},indent=2))
