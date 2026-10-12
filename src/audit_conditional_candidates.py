"""Full membership validation, unchanged-cell audit, and private draft checks."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from datetime import datetime,timezone
import json
import polars as pl
from robust_pair_experiment import ROOT,sha
from validation import validate_submission,EVIDENCE_COLUMNS

OUT=ROOT/'outputs_v11_conditional'


def main():
    assembly=json.loads((OUT/'assembly.json').read_text())
    source=ROOT/'outputs_v10_tabicl/submission_tabicl.csv'
    assert sha(source)=='736388d493d670c8f6862a379ac847add4b64374855e6a185837d2836e102b77'
    validations=[]
    for entry in assembly['entries']:
        path=OUT/entry['file']; assert sha(path)==entry['sha256']
        report=validate_submission(path,ROOT/'data/evaluation_pairs.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/sample_submission.csv','evaluation')
        payload=report.to_dict(); assert report.valid,payload
        reportpath=OUT/('validation-'+entry['candidate']+'.json'); reportpath.write_text(json.dumps(payload,indent=2)+'\n')
        validations.append({'candidate':entry['candidate'],'file':entry['file'],'sha256':entry['sha256'],'validation_sha256':sha(reportpath),'valid':True})
        print(json.dumps({'candidate':entry['candidate'],'validation':payload}),flush=True)
    risk=pl.read_csv(OUT/'submission_conditional_risk.csv',infer_schema=False)
    original=pl.read_csv(source,infer_schema=False)
    assert risk.select([c for c in original.columns if c!='risk_score']).equals(original.select([c for c in original.columns if c!='risk_score']))
    existing=json.loads((ROOT/'reports/methodology-v10-validation.json').read_text())['cases']; cases=[]
    for case in existing:
        row=risk.filter(pl.col('pair_id')==case['pair_id']); assert row.height==1
        evidence=row.select(EVIDENCE_COLUMNS).row(0); assert case['hand_id'] in evidence
        cases.append({'pair_id':case['pair_id'],'hand_id':case['hand_id'],'rank_in_risk_candidate':evidence.index(case['hand_id'])+1})
    draft=ROOT/'reports/methodology-v11-risk-draft.md'; words=len(draft.read_text().split()); assert words<=1500
    draftreport={'word_count_whitespace':words,'methodology_sha256':sha(draft),'source_submission_sha256':sha(OUT/'submission_conditional_risk.csv'),'five_reviewed_hands_retained':True,'cases':cases,'status':'Private draft for unsubmitted risk-only candidate.'}
    (ROOT/'reports/methodology-v11-risk-validation.json').write_text(json.dumps(draftreport,indent=2)+'\n')
    protected={'outputs_v5/submission_event_tails.csv':'e2ed0f5c25be000437ccb2ae753649b5643cf328e84c545ed9590b57e84d0e76',
        'outputs_v6/submission_sequence_evidence.csv':'bc309294d0e2feecc2cded72ed0330498485ccea55dfa433c8c6a5ebf05457f7',
        'outputs_v8_equity/submission_equity.csv':'be18a1590dd277fef2e376bb2936901b9ab42d937f19d55da4be46aa9e5b799d',
        'outputs_v10_tabicl/submission_tabicl.csv':'736388d493d670c8f6862a379ac847add4b64374855e6a185837d2836e102b77'}
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    ready={'created_at':datetime.now(timezone.utc).isoformat(),'entries':validations,'assembly_sha256':sha(OUT/'assembly.json'),
        'confirmation_sha256':sha(ROOT/'reports/conditional-action-confirmation/results/summary.json'),
        'all_controls_exact':assembly['v5_risk_and_v8_evidence_controls_exact_all_batches'],
        'protected_submissions_unchanged':True,'remote_import_sha256':sha(OUT/'remote-import.json'),
        'private_risk_draft_validated':True,'public_status':'Not yet submitted; daily quota was exhausted at last browser check.',
        'upload_order':'Risk-only and evidence-only first, then combined if both controls support it; preserve all immutable candidate files.'}
    (OUT/'ready-audit.json').write_text(json.dumps(ready,indent=2)+'\n'); print(json.dumps(ready),flush=True)


if __name__=='__main__': main()
