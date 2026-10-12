import polars as pl, random, hashlib, json
from pathlib import Path
p=Path('work/poker');out=p/'Codex Resume/research/astra_test_forensics_20260913'; seed=20260913
sub=p/'Codex Resume/outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv'
d=pl.read_csv(p/'data/evaluation_pairs.csv').join(pl.read_csv(sub),on='pair_id').filter(pl.col('shared_hands')>=38)
rng=random.Random(seed); chosen=[]
for name,lo,hi in [('high',.8,1),('moderate',.02,.24)]:
 rows=d.filter(pl.col('risk_score').is_between(lo,hi)).sort('pair_id').to_dicts(); selected=rng.sample(rows,2)
 for r in selected:r['role']='candidate';r['stratum']=name; chosen.append(r)
 used={r['pair_id'] for r in chosen};players={r[k] for r in chosen for k in ['player_1','player_2']}
 for r in selected:
  avail=[x for x in rows if x['pair_id'] not in used and x['player_1'] not in players and x['player_2'] not in players]
  match=min(avail,key=lambda x:(abs(x['risk_score']-r['risk_score'])/(hi-lo)+abs(x['shared_hands']-r['shared_hands'])/r['shared_hands'],x['pair_id']))
  match['role']='comparison';match['stratum']=name;match['matched_to']=r['pair_id'];chosen.append(match);used.add(match['pair_id']);players.update([match['player_1'],match['player_2']])
manifest={'seed':seed,'submission_sha256':hashlib.sha256(sub.read_bytes()).hexdigest(),'pairs':chosen,'confidence_available':False}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
seats=pl.read_parquet(p/'data/seats.parquet');hands=pl.read_parquet(p/'data/hands.parquet').filter(pl.col('phase')=='evaluation').with_columns(pl.col('started_at').cast(pl.String));actions=pl.read_parquet(p/'data/actions.parquet'); records=[]
for r in chosen:
 a=seats.filter(pl.col('player_id')==r['player_1']).select('hand_id');b=seats.filter(pl.col('player_id')==r['player_2']).select('hand_id'); shared=a.join(b,on='hand_id').join(hands.select('hand_id'),on='hand_id')['hand_id'].to_list(); ids=[]
 for i in [1,2]:
  h=r[f'evidence_hand_{i}']
  if h in shared and h not in ids:ids.append(h)
 for h in sorted(shared,key=lambda h:hashlib.sha256(f'{seed}:{r["pair_id"]}:{h}'.encode()).hexdigest()):
  if h not in ids:ids.append(h)
  if len(ids)==4:break
 r['raw_shared_hands']=len(shared);r['selected_hands']=ids
 for h in ids:
  records.append({'pair_id':r['pair_id'],'role':r['role'],'stratum':r['stratum'],'pair_players':[r['player_1'],r['player_2']],'hand':hands.filter(pl.col('hand_id')==h).to_dicts()[0],'seats':seats.filter(pl.col('hand_id')==h).sort('seat_no').to_dicts(),'actions':actions.filter(pl.col('hand_id')==h).sort('action_no').to_dicts()})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'raw_selected_hands.json').write_text(json.dumps(records,indent=2))
print(json.dumps([{k:r.get(k) for k in ['pair_id','role','stratum','risk_score','shared_hands','raw_shared_hands','selected_hands']} for r in chosen],indent=2));print('review_count',len(records))
