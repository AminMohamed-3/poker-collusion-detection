"""Adapt the existing audits only for fresh batch layout, before fitting."""
from pathlib import Path
import hashlib,json,py_compile
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;OLD=H.parent/'softplay_activation_20260910'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (H/'analysis-freeze.json').exists()
    source=(OLD/'diagnostics.py').read_text()
    a="    for i in range(32):\n        paths=[H/m/'pilot'/f'pool-{512+i:04d}' for m in plan['generator_arms']];prs=[json.loads((p/'provenance.json').read_text()) for p in paths]"
    b="    for batch,i in [(b,i) for b in plan['batches'] for i in b['pool_ids']]:\n        paths=[H/m/'pilot'/f'pool-{i:04d}' for m in batch['arms']];prs=[json.loads((p/'provenance.json').read_text()) for p in paths]"
    assert source.count(a)==1;source=source.replace(a,b)
    start=source.index('    common=set.intersection')
    end=source.index("    (H/'diagnostics.json')",start)
    source=source[:start]+'''    common_counts={}
    for batch in plan['batches']:
        common=set.intersection(*[{r['pair_id'] for r in json.loads((H/m/'features/metadata.json').read_text())['views'][0] if r['behavior_family']=='soft_play'} for m in batch['arms']])
        common_counts[str(batch['batch'])]=len(common)
        for mode in batch['arms']:output[mode]['common_paired_action_summary']=summary(pl.read_parquet(H/mode/'soft-actions.parquet').filter(pl.col('pair_id').is_in(sorted(common))))
    result={'training_only':True,'paired_scenarios_verified':paired,'paired_seats_hole_cards_profiles_episodes_exact':True,'controller_feature_parity_rows':parity,'common_supported_pairs_by_batch':common_counts,'results':output,'source_sha256':sha(__file__),'limitations':'Related policies and few positive pairs. Source AUC is not utility. Counterfactual arms share scenarios. Both batches use the same learned policy source and original training reference.'}
'''+source[end:]
    (H/'diagnostics.py').write_text(source,encoding='utf-8')
    source=(OLD/'test_randomness.py').read_text().replace("H/'calibrated'","H/'b0_calibrated'").replace("H/'calibrated/generator.py'","H/'b0_calibrated/generator.py'").replace('range(900031,900063)','range(1200031,1200063)')
    (H/'test_randomness.py').write_text(source,encoding='utf-8')
    names=['prepare_trial.py','train_trial.py','evaluate_trial.py','replication_statistics.py','diagnostics.py','test_randomness.py','test_statistics.py','run_analysis.py']
    for n in names:py_compile.compile(str(H/n),doraise=True)
    out={'created_at':datetime.now(timezone.utc).isoformat(),'study_plan_sha256':sha(H/'study-plan.json'),'sources':{n:sha(H/n) for n in names},'before_detector_fits_and_held_labels':True,'diagnostics_parent_sha256':sha(OLD/'diagnostics.py')}
    (H/'analysis-freeze.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
