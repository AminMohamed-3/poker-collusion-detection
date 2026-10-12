"""Record the five controlled official scores and verify immutable artifacts."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import polars as pl

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    observation=ROOT/'reports/evidence-public-calibration/submissions-observed.json'
    official=json.loads(observation.read_text()); rows=official['submissions']
    assert len(rows)==15 and all(row['status']=='SubmissionStatus.COMPLETE' for row in rows)
    lookup={row['fileName']:row for row in rows}; assert len(lookup)==len(rows)
    manifest=json.loads((ROOT/'reports/conditional-candidate-manifest.json').read_text())
    entries=[]
    for item in manifest['entries']:
        path=ROOT/item['file']; assert sha(path)==item['sha256']
        receipt=lookup[path.name]
        entries.append({'path':path.relative_to(ROOT).as_posix(),'sha256':item['sha256'],**receipt})
    prior=float(lookup['submission_tabicl.csv']['publicScore'])
    risk,evidence,combined,behavior,coverage=[float(row['publicScore']) for row in entries]
    expected_combined=risk+evidence-prior
    assert abs(combined-expected_combined)<=.00002
    replay=json.loads((ROOT/'reports/replay-v13/result.json').read_text())
    assert replay['byte_identical'] and replay['sha256']==entries[-1]['sha256']
    prior_writeup=ROOT/'reports/methodology-v13-draft.md'
    prior_audit=json.loads((ROOT/'reports/methodology-v13-validation.json').read_text())
    assert sha(prior_writeup)==prior_audit['methodology_sha256']
    submission=pl.read_csv(ROOT/entries[-1]['path'],infer_schema=False)
    evidence_columns=[c for c in submission.columns if c not in ['pair_id','risk_score','predicted_behavior']]
    for case in prior_audit['cases']:
        row=submission.filter(pl.col('pair_id')==case['pair_id']).row(0,named=True)
        assert row[evidence_columns[case['rank']-1]]==case['hand_id']
        assert row['predicted_behavior']==case['predicted_behavior']
    text=prior_writeup.read_text(encoding='utf-8')
    text=text.replace('Unsubmitted and without a public score. Update to the final selected entry before publication.',
        f'Official entry {entries[-1]["ref"]} is COMPLETE at public score {coverage:.5f}. Update to the final selected entry before publication.')
    text=text.replace('The previous blend scored .84351 publicly; the new candidate has no public score yet.',
        f'The previous blend scored .84351 publicly. Controlled risk and evidence entries improved separately; the full v13 candidate scored {coverage:.5f}.')
    text=text.replace('Evaluation is processed in 20 bounded batches.',
        'Evaluation is processed in 20 bounded batches. A complete frozen-model inference replay reproduced the final CSV bytes without predecessor prediction inputs; it did not retrain or rebuild every feature.')
    target=ROOT/'reports/methodology-v13-scored-draft.md'; assert not target.exists()
    target.write_text(text,encoding='utf-8'); words=len(text.split()); assert words<=1500
    report={'created_at':datetime.now(timezone.utc).isoformat(),'official_observed_at':official['observed_at'],
        'source':'Official Kaggle CLI, with rank separately observed in authenticated browser',
        'official_snapshot_sha256':sha(observation),'entries':entries,'prior_best':prior,'best_score':max(float(r['publicScore']) for r in rows),
        'observed_rank':22,'observed_teams':129,'observed_leading_score':.93720,
        'net_gain_vs_v10':round(coverage-prior,5),'risk_only_gain':round(risk-prior,5),
        'evidence_only_gain':round(evidence-prior,5),'evidence_map_gain_approx':round((evidence-prior)/.20,5),
        'combined_expected_from_controls':round(expected_combined,5),'combined_rounding_difference':round(combined-expected_combined,5),
        'behavior_gain':round(behavior-combined,5),'coverage_gain_at_displayed_precision':round(coverage-behavior,5),
        'all_five_csv_hashes_unchanged':True,'inference_replay_byte_identical':True,'first_place_achieved':False,
        'manual_final_entries_selected':False,'new_kaggle_prize_income':0,
        'scored_private_writeup':{'path':target.relative_to(ROOT).as_posix(),'sha256':sha(target),'words':words,
            'cases':prior_audit['cases'],'five_memberships_and_families_verified':True},
        'limitations':['Public scores cover approximately 30 percent of test data.','Private performance is unknown.','Coverage tie at displayed precision does not prove private equivalence.','Validation pools were reused; their results are not independent estimates.']}
    path=ROOT/'reports/conditional-public-results-20260910.json'; assert not path.exists()
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report_sha256':sha(path),'best_score':report['best_score'],'net_gain':report['net_gain_vs_v10'],'writeup_words':words,'verified_entries':len(entries)}))


if __name__=='__main__': main()
