"""Apply frozen outer-held nuisance/policies to faithful prefix actions. No fitting."""
import site;site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import hashlib,json,joblib
from pathlib import Path
import numpy as np,polars as pl
H=Path(__file__).resolve().parent;P=H.parents[2];S=H.parent/'screen';AP=H.parent/'action_policy_fast';PC=H.parent/'private_card_fast';PL=H.parent/'private_card_likelihood';K=['pair_id','direction','hand_id','action_no']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 man=H/'SCORE_MANIFEST.json';rel=H/'SCORE_RELEASE.json';assert rel.exists() and json.loads(rel.read_text())['manifest_sha256']==sha(man);mm=json.loads(man.read_text());assert all(sha(Path(k))==v for k,v in mm['sources'].items());assert not(H/'SCORE_RECEIPT.json').exists() and not any(H.glob('context_scored_prefix_fold*.npz')) and not any(H.glob('policy_probabilities_prefix_fold*.parquet'))
 inputs=json.loads((PL/'MANIFEST.json').read_text())['inputs'];actions=pl.read_parquet(sorted((H/'prefix_action_cache').glob('*.parquet')));receipts=[]
 for o in [0,1]:
  base=np.load(S/f'context_fold{o}_prefix60.npz',allow_pickle=False);full=np.load(AP/f'fast129_context_fold{o}.npz',allow_pickle=False);ev=np.load(S/f'evaluator_fold{o}_prefix60.npz',allow_pickle=False);assert np.array_equal(base['pair_id'],ev['pair_id']) and np.array_equal(base['X_train'],full['X_train'][:,:128],equal_nan=True) and np.array_equal(base['feature_name'],full['feature_name'][:128])
  q=actions.filter(pl.col('pair_id').is_in(base['pair_id']));assert q['pair_id'].n_unique()==len(base['pair_id'])
  nuisance=joblib.load(PC/f'cpu_models_outer{o}.joblib')[2]['strength_full'];policies=joblib.load(AP/f'policy_models_outer{o}_pipeline2.joblib');surprise=q['partner_strength'].to_numpy()-nuisance.predict(q.select(inputs).to_numpy());X=np.c_[q.select(inputs).to_numpy(),surprise];probs={n:np.asarray(policies[n].predict_proba(X),dtype=np.float64) for n in ['target','normal']}
  for n,z in probs.items():assert np.array_equal(policies[n].classes_,np.arange(4)) and z.shape==(q.height,4) and np.isfinite(z).all() and (z>=0).all() and (z<=1).all() and np.allclose(z.sum(1),1,atol=1e-6)
  y=q['action_class'].to_numpy();ix=np.arange(q.height);g=np.log(np.maximum(probs['target'][ix,y],1e-12))-np.log(np.maximum(probs['normal'][ix,y],1e-12));score=q.select('pair_id','direction').with_columns(pl.Series('g',g)).group_by('pair_id','direction').agg(pl.mean('g')).group_by('pair_id').agg(pl.mean('g').alias('llr'),pl.col('direction').n_unique().alias('nd'));assert score.height==len(base['pair_id']) and score['nd'].min()==2
  sm=dict(score.select('pair_id','llr').iter_rows());v=np.array([sm[x] for x in base['pair_id']],np.float32);dest=H/f'context_scored_prefix_fold{o}.npz';np.savez_compressed(dest,X_train=full['X_train'],y_train=full['y_train'],X_held=np.c_[base['X_held'],v],pair_id=base['pair_id'],feature_name=full['feature_name']);chk=np.load(dest,allow_pickle=False);assert np.array_equal(chk['X_train'],full['X_train'],equal_nan=True) and np.array_equal(chk['X_held'][:,:128],base['X_held'],equal_nan=True)
  pp=H/f'policy_probabilities_prefix_fold{o}.parquet';out=q.select(*K,'table_id','action_class');out=out.with_columns(*[pl.Series(f'target_p{k}',probs['target'][:,k]) for k in range(4)],*[pl.Series(f'normal_p{k}',probs['normal'][:,k]) for k in range(4)]);out.write_parquet(pp);receipts.append({'fold':o,'pairs':len(v),'actions':q.height,'both_directions':True,'context_sha256':sha(dest),'probabilities_sha256':sha(pp),'finite_normalized':True,'first128_exact':True})
 (H/'SCORE_RECEIPT.json').write_text(json.dumps({'score_manifest_sha256':sha(man),'folds':receipts},indent=2)+'\n')
if __name__=='__main__':main()
