"""Reconstruct frozen model posteriors across all evaluation pairs; no fits, raw-hand judgments or labels."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,hashlib,datetime,joblib
import polars as pl
import numpy as np
ROOT=Path('C:/Data/Projects/Money Making/work/poker');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from robust_pair_experiment import matrix,LABELS
from tail_pair_experiment import tails

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
SUB=ROOT/'Codex Resume/outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv';OLD=ROOT/'outputs_v11_conditional';F=OLD/'inference-package/results'
plan={'written_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':'ALL112540 current submission pairs. User requests broader .6/.65 confidence rule and roughly1000 changes. Frozen classifier inference only; no target labels or public scores.','descriptive_condition':'Conditional known-family maximum <=.6 (and descriptive counts <=.5,<=.8); not a calibrated unknown detector.','no_raw_hand_forensic_selection_here':True,'no_fit_or_labels_or_quality_metrics':True,'quarantined_test_exploration':True,'source_submission_sha256':sha(SUB),'source_code_sha256':sha(Path(__file__))}
assert not (OUT/'confidence-plan.json').exists();(OUT/'confidence-plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
assert plan['source_submission_sha256']=='9d0f0d2f698a5976b6aeff818156a5e3373b52aba433d81d313eb99c4cb2c315'
fit=json.loads((OLD/'fit.json').read_text());assert sha(OLD/'pair-model.joblib')==fit['models_sha256']['pair-model.joblib']
bundle=joblib.load(OLD/'pair-model.joblib');sub=pl.read_csv(SUB,columns=['pair_id','risk_score','predicted_behavior']);assert sub.height==112540
ids=sub['pair_id'].to_list();pairs=pl.read_parquet(ROOT/'outputs_v2/pairs_evaluation.parquet').filter(pl.col('pair_id').is_in(ids));oldsub=pl.read_csv(OLD/'submission_conditional_combined.csv',columns=['pair_id','risk_score']).filter(pl.col('pair_id').is_in(ids));inverse={v:k for k,v in LABELS.items()}
outputs=[];source_hashes={}
for path in sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet')):
    name=path.stem;raw=pl.scan_parquet(path).filter(pl.col('pair_id').is_in(ids)).select(['pair_id','table_id']+bundle['events']).collect()
    if not raw.height:continue
    fr=json.loads((F/(name+'.json')).read_text());cpath=F/(name+'-pairs.parquet');assert sha(cpath)==fr['files'][cpath.name];source_hashes[cpath.name]=sha(cpath)
    frame=pairs.join(tails(raw,bundle['events']),on=['pair_id','table_id'],validate='1:1').join(pl.read_parquet(cpath),on=['pair_id','table_id'],validate='1:1').sort('pair_id')
    prob=bundle['model'].predict_proba(matrix(frame,bundle['features']),num_threads=2);assert list(bundle['model'].classes_)==[0,1,2,3]
    old=oldsub.join(frame.select('pair_id'),on='pair_id',how='semi').sort('pair_id');np.testing.assert_array_equal(1-prob[:,0],old['risk_score'].to_numpy())
    assigned=[inverse[int(x)] for x in (np.argmax(prob[:,1:],axis=1)+1)];expected=sub.join(frame.select('pair_id'),on='pair_id',how='semi').sort('pair_id');assert assigned==expected['predicted_behavior'].to_list()
    conditional=prob[:,1:]/prob[:,1:].sum(axis=1,keepdims=True)
    outputs.append(frame.select('pair_id','table_id').with_columns(*[pl.Series('p'+str(i),prob[:,i]) for i in range(4)],pl.Series('conditional_max',conditional.max(axis=1)),pl.Series('conditional_margin',np.sort(conditional,axis=1)[:,-1]-np.sort(conditional,axis=1)[:,-2])))
d=pl.concat(outputs).join(sub,on='pair_id',validate='1:1').sort('pair_id');assert d.height==d['pair_id'].n_unique()==112540
d.write_parquet(OUT/'all-family-posteriors.parquet')
report={'pairs':112540,'old_risk_exact':True,'current_family_exact':True,'model_sha256':sha(OLD/'pair-model.joblib'),'output_sha256':sha(OUT/'all-family-posteriors.parquet'),'conditional_max_minimum':float(d['conditional_max'].min()),'conditional_max_median':float(d['conditional_max'].median()),'conditional_max_counts':{str(x):d.filter(pl.col('conditional_max')<=x).height for x in [.5,.6,.65,.7,.8,.9,.95,.99]},'no_unknown_labels_or_quality_score':True,'source_feature_hashes':source_hashes}
(OUT/'confidence-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='source_feature_hashes'},indent=2))
