"""Explicit one-worker resume for the guarded partial extraction. No overwrite."""
import concurrent.futures as cf, hashlib, json, os, shutil, sys, time
from pathlib import Path
os.environ["POLARS_MAX_THREADS"]="1"
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'.venv/Lib/site-packages'));sys.path.insert(0,str(HERE))
import polars as pl
from stream_contexts import sha, table_ids
from run_extract import worker

def main():
    rm=json.loads((HERE/'RESUME_MANIFEST.json').read_text())
    release=json.loads((HERE/'RESUME_RELEASE.json').read_text()) if (HERE/'RESUME_RELEASE.json').exists() else {}
    assert release.get('resume_manifest_sha256')==sha(HERE/'RESUME_MANIFEST.json'),'root resume release absent'
    assert sha(HERE/'MANIFEST.json')==rm['original_manifest_sha256']
    original=json.loads((HERE/'MANIFEST.json').read_text())
    for rel,expected in original['inputs'].items(): assert sha(ROOT/rel)==expected,('original input drift',rel)
    for rel,expected in original['code'].items(): assert sha(HERE/rel)==expected,('original code drift',rel)
    assert sha(HERE/'run_extract_resume.py')==rm['resume_runner_sha256'],'resume runner drift'
    assert sha(HERE/'conditional_action_features.py')==rm['feature_source_sha256'],'feature source drift'
    assert sha(HERE/'FAILED_RUN_RECEIPT.json')==rm['failed_run_receipt_sha256'],'failed receipt drift'
    assert sha(HERE/'evaluation_style_counts.parquet')==rm['evaluation_style_counts_sha256']
    allowed={'parquet','json'}; files=[p for p in (HERE/'table_shards').iterdir() if p.is_file()]
    assert all(p.suffix.lstrip('.') in allowed for p in files),'stray partial output'
    parquets={p.stem:p for p in files if p.suffix=='.parquet'};receipts={p.stem:p for p in files if p.suffix=='.json'}
    assert set(parquets)==set(receipts)==set(rm['completed_tables']) and len(parquets)==14
    for table in sorted(parquets):
        bound=rm['completed_artifacts'][table]; rec=json.loads(receipts[table].read_text())
        assert sha(parquets[table])==bound['parquet_sha256']==rec['sha256']
        assert sha(receipts[table])==bound['receipt_sha256'] and rec['table_id']==table
    all_tables=table_ids();pending=[t for t in all_tables if t not in parquets]
    assert len(all_tables)==400 and len(pending)==386 and pending==rm['pending_tables']
    assert not (HERE/'pair_coverage.parquet').exists() and not (HERE/'COVERAGE.json').exists()
    assert shutil.disk_usage(HERE).free>=2*1024**3
    reports=[];start=time.perf_counter();pool=cf.ProcessPoolExecutor(max_workers=1)
    try:
        for table in pending:
            assert table not in parquets and not (HERE/'table_shards'/f'{table}.json').exists()
            future=pool.submit(worker,table);report=future.result();reports.append(report)
            artifact_bytes=sum(p.stat().st_size for p in (HERE/'table_shards').glob('*.parquet'))
            assert artifact_bytes<=2*1024**3 and shutil.disk_usage(HERE).free>=2*1024**3
            print(json.dumps({**report,'artifact_bytes':artifact_bytes,'resume_index':len(reports)}),flush=True)
    except BaseException:
        pool.shutdown(wait=True,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    assert len(list((HERE/'table_shards').glob('*.parquet')))==400
    directions=(pl.scan_parquet(HERE/'table_shards/*.parquet').group_by('pair_id','direction').agg(pl.len().alias('contexts')).collect())
    assert set(directions['direction'].unique().to_list())<={0,1}
    wide=(directions.with_columns((pl.lit('direction_')+pl.col('direction').cast(pl.String)).alias('d'))
          .pivot(on='d',index='pair_id',values='contexts').fill_null(0))
    keys=pl.read_parquet(HERE/'candidate_pair_keys.parquet').select('pair_id')
    assert keys.height==112540 and keys['pair_id'].n_unique()==112540
    assert directions.join(keys,on='pair_id',how='anti').height==0
    coverage=keys.join(wide,on='pair_id',how='left').fill_null(0)
    for name in ('direction_0','direction_1'):
        if name not in coverage.columns:coverage=coverage.with_columns(pl.lit(0).alias(name))
    assert coverage.height==112540 and coverage['pair_id'].n_unique()==112540
    coverage.write_parquet(HERE/'pair_coverage.parquet')
    missing=coverage.filter((pl.col('direction_0')==0)|(pl.col('direction_1')==0)).height
    result={'tables':400,'original_completed_tables':14,'resumed_tables':len(reports),
      'rows':sum(json.loads(p.read_text())['rows'] for p in (HERE/'table_shards').glob('*.json')),
      'candidate_pairs':112540,'pairs_missing_either_direction':missing,'extra_generated_pairs':0,
      'directions_valid_subset_0_1':True,'pair_coverage_sha256':sha(HERE/'pair_coverage.parquet'),
      'resume_seconds':time.perf_counter()-start,'style_sha256':sha(HERE/'evaluation_style_counts.parquet'),
      'neutral_fallback':'future score 0.0 if either direction has zero valid contexts; availability receipt-only',
      'malformed_cards':0,'unknown_actions':0,'nonfinite_features':0,'policy_or_TN_convention_defined':False}
    (HERE/'COVERAGE.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
