"""Package only disclosed evaluation gameplay and a frozen fitted reference."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import tarfile
import polars as pl
from robust_pair_experiment import ROOT,sha
from sequence_evidence_candidate import verify_source_cache

OUT=ROOT/'outputs_v11_conditional'
PACKAGE=OUT/'inference-package'


def main():
    policy=json.loads((OUT/'policy/summary.json').read_text())
    assert sha(OUT/'policy/final-models.joblib')==policy['final_reference_sha256']
    verify_source_cache('evaluation')
    PACKAGE.mkdir(exist_ok=False); (PACKAGE/'data').mkdir(); (PACKAGE/'keys').mkdir()
    hands=pl.scan_parquet(ROOT/'data/hands.parquet').filter(pl.col('phase')=='evaluation').select('hand_id','table_id','board_cards','button_seat','big_blind').collect()
    ids=hands.select('hand_id'); hands.write_parquet(PACKAGE/'data/hands.parquet')
    for filename in ['seats.parquet','actions.parquet']:
        frame=pl.scan_parquet(ROOT/'data'/filename).join(ids.lazy(),on='hand_id',how='semi').collect()
        frame.write_parquet(PACKAGE/'data'/filename)
        if filename=='actions.parquet':
            cls=pl.when(pl.col('action')=='fold').then(0).when(pl.col('action')=='check').then(1).when((pl.col('action')=='call')|((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call')))).then(2).otherwise(3)
            frame.with_columns(cls.alias('kind')).group_by('player_id').agg(*[(pl.col('kind')==k).sum().alias('n'+str(k)) for k in range(4)]).write_parquet(PACKAGE/'style-counts.parquet')
    shutil.copy2(ROOT/'data/evaluation_pairs.csv',PACKAGE/'data/evaluation_pairs.csv')
    shutil.copy2(ROOT/'reports/evidence-public-calibration/selected-pairs.parquet',PACKAGE/'selected-pairs.parquet')
    shutil.copy2(OUT/'policy/final-models.joblib',PACKAGE/'final-models.joblib')
    source=['conditional_action_features.py','advanced_features.py','run_conditional_policy_screen.py','conditional_inference_worker.py']
    for name in source: shutil.copy2(ROOT/'src'/name,PACKAGE/name)
    for path in sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet')):
        pl.read_parquet(path,columns=['pair_id','hand_id','table_id']).write_parquet(PACKAGE/'keys'/path.name)
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'conditional_features':policy['features'],
        'source_data_sha256':{p:sha(ROOT/'data'/p) for p in ['hands.parquet','seats.parquet','actions.parquet','evaluation_pairs.csv']},
        'files':{p.relative_to(PACKAGE).as_posix():sha(p) for p in files},
        'policy_summary_sha256':sha(OUT/'policy/summary.json'),
        'scope':'All 112540 evaluation pairs. No training or evaluation labels. Two-pool subchunks. Return pair aggregates for all pairs and complete hand features for the same 10000 pairs used by v8 evidence.',
        'evidence_selection':'Fixed original v6 top 10000 risk pairs; all remaining v10 evidence is preserved.',
        'model_versions':{'lightgbm':'4.7.0','polars':'1.44.1','phevaluator':'0.6.0'}}
    (PACKAGE/'config.json').write_text(json.dumps(plan,indent=2)+'\n')
    archive=OUT/'inference-package.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for p in [PACKAGE/'config.json',*files]: t.add(p,arcname=p.relative_to(PACKAGE).as_posix())
    receipt={'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files),'config_sha256':sha(PACKAGE/'config.json')}
    (OUT/'inference-package-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
