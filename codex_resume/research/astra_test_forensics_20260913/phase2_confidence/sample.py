import polars as pl,json,hashlib
from pathlib import Path
p=Path('work/poker');out=p/'Codex Resume/research/astra_test_forensics_20260913/phase2_confidence';seed=20260915
post=pl.read_parquet(p/'Codex Resume/research/other_coordination_abstention_20260913/high-risk-family-posteriors.parquet');sub=pl.read_csv(p/'Codex Resume/outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv');d=post.join(pl.read_csv(p/'data/evaluation_pairs.csv'),on='pair_id').join(sub.drop('risk_score','predicted_behavior'),on='pair_id')
chosen=d.filter(pl.col('conditional_max')<=.5).sort('pair_id').to_dicts();prior=json.loads((out.parent/'manifest.json').read_text())['pairs'];used={r[k] for r in chosen+prior for k in ['player_1','player_2']}
for r in chosen:r['role']='low_confidence'
controls=d.filter(pl.col('conditional_max')>=.95).to_dicts()
for r in chosen.copy():
 m=min([x for x in controls if not {x['player_1'],x['player_2']} & used],key=lambda x:(abs(x['risk_score']-r['risk_score'])+abs(x['shared_hands']-r['shared_hands'])/r['shared_hands'],x['pair_id']));m['role']='high_confidence_comparison';m['matched_to']=r['pair_id'];chosen.append(m);used.update([m['player_1'],m['player_2']])
(out/'manifest.json').write_text(json.dumps({'seed':seed,'pairs':chosen},indent=2))
seats=pl.read_parquet(p/'data/seats.parquet');hands=pl.read_parquet(p/'data/hands.parquet').filter(pl.col('phase')=='evaluation').with_columns(pl.col('started_at').cast(pl.String));actions=pl.read_parquet(p/'data/actions.parquet');records=[];aggregates=[]
for r in chosen:
 a=seats.filter(pl.col('player_id')==r['player_1']).select('hand_id');b=seats.filter(pl.col('player_id')==r['player_2']).select('hand_id');shared=a.join(b,on='hand_id').join(hands.select('hand_id'),on='hand_id')['hand_id'].to_list();ids=[]
 for i in [1,2]:
  h=r[f'evidence_hand_{i}']
  if h in shared and h not in ids:ids.append(h)
 for h in sorted(shared,key=lambda h:hashlib.sha256(f'{seed}:{r["pair_id"]}:{h}'.encode()).hexdigest()):
  if h not in ids:ids.append(h)
  if len(ids)==4:break
 r['selected_hands']=ids;r['raw_shared_hands']=len(shared)
 aa=actions.filter(pl.col('hand_id').is_in(shared)&pl.col('player_id').is_in([r['player_1'],r['player_2']]));aggregates.append({'pair_id':r['pair_id'],'shared_hands':len(shared),'actions_by_player':aa.group_by('player_id','street','action').len().sort('player_id','street','action').to_dicts(),'seat_outcomes':seats.filter(pl.col('hand_id').is_in(shared)&pl.col('player_id').is_in([r['player_1'],r['player_2']])).group_by('player_id').agg(pl.col('net_chips').sum(),pl.col('total_contribution').sum(),pl.col('went_to_showdown').sum()).to_dicts()})
 for h in ids:records.append({'pair_id':r['pair_id'],'role':r['role'],'pair_players':[r['player_1'],r['player_2']],'hand':hands.filter(pl.col('hand_id')==h).to_dicts()[0],'seats':seats.filter(pl.col('hand_id')==h).sort('seat_no').to_dicts(),'actions':actions.filter(pl.col('hand_id')==h).sort('action_no').to_dicts()})
(out/'manifest.json').write_text(json.dumps({'seed':seed,'pairs':chosen},indent=2));(out/'raw_selected_hands.json').write_text(json.dumps(records,indent=2));(out/'shared_history_aggregates.json').write_text(json.dumps(aggregates,indent=2)); print([{k:r.get(k) for k in ['pair_id','role','risk_score','conditional_max','shared_hands','p1','p2','p3']} for r in chosen])
