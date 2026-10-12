"""Complete assembly using original phase-bearing data for full membership validation."""
from datetime import datetime,timezone
import json
import polars as pl
from build_diverse_evidence_candidates import OUT,BASE,RESEARCH,NAMES,ROOT,sha,write,EVIDENCE_COLUMNS,validate_submission

def main():
    assert not (OUT/'ready-audit.json').exists()
    source=pl.read_csv(BASE,infer_schema=False); selected=pl.read_parquet(ROOT/'reports/replay-v13/selected-pairs.parquet')
    plan=json.loads((OUT/'plan.json').read_text()); assert sha(BASE)==plan['source_csv_sha256']
    receipts=sorted((OUT/'batches').glob('batch_*.json')); assert len(receipts)==20
    for path in receipts:
        receipt=json.loads(path.read_text()); assert receipt['plan_sha256']==sha(OUT/'plan.json') and receipt['incumbent_evidence_exact']
        assert all(sha(OUT/'batches'/p)==h for p,h in receipt['files'].items())
    # Preserve the first failed validation, caused by trimmed hands lacking phase.
    previous=json.loads((OUT/'seed_evidence-validation.json').read_text()); assert previous['errors'][0]['code']=='schema'
    write(OUT/'assembly-plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'inference_plan_sha256':sha(OUT/'plan.json'),
        'reason':'Original builder completed all 20 exact incumbent controls, then supplied a trimmed hands table without phase to the validator. This assembly validates against original hands/seats. No model, score, ranking, gate or frozen inference plan changes.',
        'prior_validation_sha256':sha(OUT/'seed_evidence-validation.json'),'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/evaluation_pairs.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/sample_submission.csv']}})
    entries=[]
    for candidate in NAMES:
        replacement=pl.read_parquet(sorted((OUT/'batches').glob('batch_*-'+candidate+'.parquet'))).sort('pair_id'); assert replacement.height==16060
        result=pl.concat([source.join(selected,on='pair_id',how='anti'),source.join(selected,on='pair_id',how='semi').drop(EVIDENCE_COLUMNS).join(replacement,on='pair_id',validate='1:1')]).select(source.columns)
        result=source.select('pair_id').join(result,on='pair_id',validate='1:1',maintain_order='left')
        assert result.select('pair_id','risk_score','predicted_behavior').equals(source.select('pair_id','risk_score','predicted_behavior'))
        assert result.join(selected,on='pair_id',how='anti').equals(source.join(selected,on='pair_id',how='anti'))
        output=OUT/('submission_'+candidate+'.csv')
        if output.exists(): assert pl.read_csv(output,infer_schema=False).equals(result)
        else: result.write_csv(output)
        validation=validate_submission(output,ROOT/'data/evaluation_pairs.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/sample_submission.csv').to_dict()
        write(OUT/(candidate+'-full-validation.json'),validation); assert validation['valid']
        changed=int(result.select(pl.any_horizontal(*[pl.col(c)!=source[c] for c in EVIDENCE_COLUMNS]).sum()).item())
        entries.append({'candidate':candidate,'file':output.name,'sha256':sha(output),'validation':validation,'changed_evidence_lists':changed,'risk_behavior_exact':True,'outside_selection_exact':True,'eligible':True,'submitted':False})
        print(json.dumps({'candidate':candidate,'changed_evidence_lists':changed,'valid':True}),flush=True)
    assert len({r['sha256'] for r in entries})==5
    write(OUT/'ready-audit.json',{'completed_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(OUT/'plan.json'),'assembly_plan_sha256':sha(OUT/'assembly-plan.json'),'fit_sha256':sha(OUT/'fit.json'),'admission_sha256':sha(RESEARCH/'evidence-panel/admission.json'),
        'entries':entries,'twenty_incumbent_batch_controls_exact':True,'no_submissions_attempted':True})

if __name__=='__main__': main()
