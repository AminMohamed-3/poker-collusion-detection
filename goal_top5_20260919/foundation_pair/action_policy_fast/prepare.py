"""Bind original Fast folds/contexts and create membership receipts; fits nothing."""
import site;site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import hashlib,json,io
from pathlib import Path
import numpy as np,polars as pl
H=Path(__file__).resolve().parent;P=H.parents[2];S=H.parent/'screen';L=H.parent/'private_card_likelihood'; PC=H.parent/'private_card_fast';SALT='foundation-pf35-screen-v1:'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fold(t):return int(hashlib.sha256((SALT+t).encode()).hexdigest()[:16],16)%2
def bit(t,o,j,suffix):return int(hashlib.sha256(f'private-fast|{suffix}|{o}|{j}|{t}'.encode()).hexdigest(),16)&1
def main():
 b=io.BytesIO();pl.DataFrame(schema={'pair_id':pl.String,'direction':pl.Int64,'hand_id':pl.String,'action_no':pl.Int64,'table_id':pl.String,'actor_id':pl.String,'fit_weight':pl.Float64}).write_parquet(b);b.seek(0);assert pl.read_parquet(b).columns==['pair_id','direction','hand_id','action_no','table_id','actor_id','fit_weight']
 raw=pl.read_parquet(P/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet'); lab=pl.read_csv(P/'data/development_labels.csv'); raw=raw.join(lab.select('pair_id','player_1','player_2'),on='pair_id',validate='1:1').with_columns(pl.Series('fast_fold',[fold(t) for t in raw['table_id']]))
 rows=[]
 for o in [0,1]:
  c=np.load(S/f'context_fold{o}.npz',allow_pickle=False);e=np.load(S/f'evaluator_fold{o}.npz',allow_pickle=False); held=raw.filter(pl.col('fast_fold')==o);train=raw.filter(pl.col('fast_fold')!=o)
  assert np.array_equal(c['pair_id'],e['pair_id']) and np.array_equal(c['pair_id'],held['pair_id'].to_numpy()) and c['X_train'].shape==(train.height,128) and c['X_held'].shape==(held.height,128) and len(np.unique(c['feature_name']))==128
  full=pl.read_parquet(P/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet').select(c['feature_name'].tolist()).to_numpy().astype(np.float32); full[~np.isfinite(full)]=np.nan
  assert np.array_equal(full[raw['fast_fold'].to_numpy()!=o],c['X_train'],equal_nan=True) and np.array_equal(full[raw['fast_fold'].to_numpy()==o],c['X_held'],equal_nan=True)
  for j in [0,1]:
   train=train.with_columns(pl.Series(f'inner{o}',[bit(t,o,j,'inner') for t in train['table_id']])) if f'inner{o}' not in train.columns else train
  rows.extend(train.select('pair_id','table_id','player_1','player_2','fast_fold').with_columns(pl.Series('inner_fold',[bit(t,o,0,'inner') for t in train['table_id']]),pl.lit(o).alias('outer_context')).to_dicts());rows.extend(held.select('pair_id','table_id','player_1','player_2','fast_fold').with_columns(pl.lit(-1).alias('inner_fold'),pl.lit(o).alias('outer_context')).to_dicts())
 pl.DataFrame(rows).write_parquet(H/'membership.parquet')
 sources=[H/'PROTOCOL.md',S/'context_fold0.npz',S/'context_fold1.npz',S/'evaluator_fold0.npz',S/'evaluator_fold1.npz',S/'pf35_predictions.npz',S/'prep_receipt.json',S/'run_pf35.py',P/'goal_top5_20260919/tabpfn35/runtime35.py',P/'goal_top5_20260919/tabpfn35/tabpfn-v3.5-fast-20260909.safetensors',P/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet',P/'data/development_labels.csv',L/'fit.py',L/'capped_training_keys.parquet',PC/'MANIFEST.json',PC/'CPU_RECEIPT.json',PC/'cpu_generate.py',PC/'cpu_models_outer0.joblib',PC/'cpu_models_outer1.joblib',L/'row_membership.parquet',*sorted((P/'reports/conditional-action-experiment/action-cache').glob('*.parquet'))]
 arts=[H/'prepare.py',H/'cpu_generate.py',H/'gpu_run.py',H/'evaluate.py',H/'membership.parquet'];m={'status':'prepared_unreleased','cpu_release':'CPU_RELEASE.json','gpu_release':'GPU_RELEASE.json','sources':{str(x.resolve()):sha(x) for x in sources},'artifacts':{x.name:sha(x) for x in arts},'cpu_fits':12,'gpu_fits':2,'seed':35935,'feature_count_original':128,'feature_count_candidate':129,'new_nuisance_fits':0,'new_action_policy_fits':12}
 (H/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
