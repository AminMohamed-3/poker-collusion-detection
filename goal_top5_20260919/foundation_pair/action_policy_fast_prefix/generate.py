"""Faithful prefix action extractor. Default is one-table semantic preflight; --full needs EXTRACT_RELEASE."""
import os;os.environ['POLARS_MAX_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import site;site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import argparse,hashlib,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np,polars as pl
H=Path(__file__).resolve().parent;P=H.parents[2];sys.path.insert(0,str(P/'src'))
from conditional_action_features import INPUTS,extract_hand,action_class,card_features
STYLE=['own_style_fold','own_style_check','own_style_call','own_style_raise','partner_style_fold','partner_style_check','partner_style_call','partner_style_raise']
NONSTYLE=[x for x in INPUTS if x not in STYLE];KEY=['pair_id','direction','hand_id','action_no']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prefix_hands():
 return pl.read_parquet(P/'data/hands.parquet').filter(pl.col('phase')=='development').sort('table_id','started_at').with_columns((pl.int_range(pl.len()).over('table_id')/pl.len().over('table_id')).alias('_pos')).filter(pl.col('_pos')<.6).drop('_pos')
def action_data(hands):
 ids=hands.select('hand_id');a=pl.scan_parquet(P/'data/actions.parquet').join(ids.lazy(),on='hand_id',how='semi').collect().sort('hand_id','action_no');return a
def style_lookup(actions):
 rows=[]
 for player,g in actions.group_by('player_id'):
  c=np.zeros(4,dtype=np.int64)
  for r in g.iter_rows(named=True):c[action_class(r)]+=1
  rows.append((player[0] if isinstance(player,tuple) else player,c))
 return dict(rows)
def pair_keys(pair_ids,hands):
 labels=pl.read_csv(P/'data/development_labels.csv').filter(pl.col('pair_id').is_in(pair_ids)).select('pair_id','player_1','player_2')
 parts=[]
 for p in sorted((P/'outputs_v2/hands_development').glob('batch_*.parquet')):
  z=pl.read_parquet(p,columns=['pair_id','hand_id']).filter(pl.col('pair_id').is_in(pair_ids));
  if z.height:parts.append(z)
 return pl.concat(parts).join(hands.select('hand_id'),on='hand_id',how='semi').join(labels,on='pair_id',validate='m:1')
def extract(hands,actions,keys,totals):
 ids=hands.select('hand_id');seats=pl.scan_parquet(P/'data/seats.parquet').join(ids.lazy(),on='hand_id',how='semi').collect();hp=defaultdict(list);hs=defaultdict(list);ha=defaultdict(list)
 for pair,hand,a,b in keys.iter_rows():hp[hand].append((pair,a,b))
 for r in seats.iter_rows(named=True):hs[r['hand_id']].append(r)
 for r in actions.iter_rows(named=True):ha[r['hand_id']].append(r)
 rows=[]
 for hand in hands.iter_rows(named=True):
  hid=hand['hand_id'];obs=ha[hid];cur=defaultdict(lambda:np.zeros(4,dtype=np.int64))
  for a in obs:cur[a['player_id']][action_class(a)]+=1
  styles={}
  for seat in hs[hid]:
   player=seat['player_id'];counts=totals.get(player,np.zeros(4,dtype=np.int64))-cur[player];assert counts.min()>=0;styles[player]=((counts+1)/(counts.sum()+4)).tolist();assert abs(sum(styles[player])-1)<1e-12
  rows.extend(extract_hand(hand,hs[hid],obs,hp[hid],styles))
 return pl.DataFrame(rows).with_columns(pl.col(INPUTS+['partner_strength','observed_log_amount_bb']).cast(pl.Float32))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');x=ap.parse_args();hands=prefix_hands();actions=action_data(hands);totals=style_lookup(actions)
 if x.full:
  man=H/'MANIFEST.json';rel=H/'EXTRACT_RELEASE.json';assert rel.exists() and json.loads(rel.read_text())['manifest_sha256']==sha(man);mm=json.loads(man.read_text());assert all(sha(Path(k))==v for k,v in mm['sources'].items());assert not(H/'EXTRACT_RECEIPT.json').exists();out=H/'prefix_action_cache';out.mkdir(exist_ok=False)
  ids=np.concatenate([np.load(H.parent/'screen'/f'context_fold{o}_prefix60.npz',allow_pickle=False)['pair_id'] for o in [0,1]]);keys=pair_keys(ids,hands);tables=sorted(keys.join(hands.select('hand_id','table_id'),on='hand_id')['table_id'].unique().to_list());receipt=[]
  for i in range(0,len(tables),20):
   ts=tables[i:i+20];hh=hands.filter(pl.col('table_id').is_in(ts));kk=keys.join(hh.select('hand_id'),on='hand_id',how='semi');aa=actions.join(hh.select('hand_id'),on='hand_id',how='semi');z=extract(hh,aa,kk,totals);dest=out/f'batch_{i//20:03}.parquet';z.write_parquet(dest);receipt.append({'file':dest.name,'rows':z.height,'sha256':sha(dest)})
  (H/'EXTRACT_RECEIPT.json').write_text(json.dumps({'manifest_sha256':sha(man),'rows':sum(r['rows'] for r in receipt),'batches':receipt},indent=2)+'\n');return
 # deterministic smallest eligible table; recompute rows and compare every non-style value to the historical cache.
 ids=np.concatenate([np.load(H.parent/'screen'/f'context_fold{o}_prefix60.npz',allow_pickle=False)['pair_id'] for o in [0,1]]);keys=pair_keys(ids,hands);tab=sorted(keys.join(hands.select('hand_id','table_id'),on='hand_id')['table_id'].unique().to_list())[0];hh=hands.filter(pl.col('table_id')==tab);kk=keys.join(hh.select('hand_id'),on='hand_id',how='semi');aa=actions.join(hh.select('hand_id'),on='hand_id',how='semi');z=extract(hh,aa,kk,totals)
 old=pl.read_parquet(sorted((P/'reports/conditional-action-experiment/action-cache').glob('*.parquet'))).join(z.select(KEY),on=KEY,how='semi').sort(KEY);z=z.sort(KEY);assert old.height==z.height and old.height>0
 for c in NONSTYLE+['action_class','partner_strength','observed_log_amount_bb']:assert np.array_equal(old[c].to_numpy(),z[c].to_numpy(),equal_nan=True),(c,np.nanmax(np.abs(old[c].to_numpy()-z[c].to_numpy())))
 assert any(not np.array_equal(old[c].to_numpy(),z[c].to_numpy()) for c in STYLE)
 dest=H/'TINY_PREFLIGHT.json';assert not dest.exists();dest.write_text(json.dumps({'table_id':tab,'hands':hh.height,'rows':z.height,'nonstyle_exact_columns':len(NONSTYLE)+3,'style_recomputed_columns':STYLE,'cutoff':'rank/len < .6 after table_id,started_at sort','passed':True},indent=2)+'\n')
if __name__=='__main__':main()
