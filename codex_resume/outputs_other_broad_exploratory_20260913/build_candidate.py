import os
os.environ['POLARS_MAX_THREADS']='2'
import sys,csv,io,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import polars as pl
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent;R=OUT.parent;ROOT=R.parent
SOURCE=R/'outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv'
CONF=R/'research/other_coordination_broad_confidence_20260913/all-family-posteriors.parquet'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):
 assert not p.exists();p.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
assert sha(SOURCE)=='9d0f0d2f698a5976b6aeff818156a5e3373b52aba433d81d313eb99c4cb2c315'
assert sha(CONF)=='fbed361474cfaf092c3725b5d225e6d49e1525976a6adc909bae95ca62002b91'
protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'user_authorized_exploratory_exception':True,'user_request':'A more visible variant, possibly0.6 or0.65 confidence, roughly1000 label changes.','base_score':.89457,'base_sha256':sha(SOURCE),'confidence_sha256':sha(CONF),'rule':{'risk_at_least':.05,'conditional_known_family_max_strictly_below':.65,'replacement':'other_coordination'},'threshold_selection':'User suggested .65; selected a round .05 risk floor after label-free model-output counts yielded970 changes, close to requested1000. No quality score or manual hand judgement used. This is adaptive exploratory design, not a validated threshold.','frozen_model':'Existing v11 classifier, exact old-risk and current hard-family reproduction for all112540 rows','no_manual_test_labels_or_case_specific_overrides':True,'previous_test_exposure_disclosed':'14 pairs were descriptively inspected earlier; the uniform candidate rule does not import those opinions or IDs.','all_risks_and_evidence_preserved':True,'metrics_evaluated_here':False,'submitted':False,'feedback_scope':'Global candidate score only; no inference of individual test labels from public deltas.'}
write(OUT/'protocol.json',protocol)
x=pl.read_parquet(CONF);assert x.height==x['pair_id'].n_unique()==112540
selected=x.filter((pl.col('risk_score')>=.05)&(pl.col('conditional_max')<.65));ids=set(selected['pair_id'].to_list());assert len(ids)==970
raw=SOURCE.read_bytes();lines=raw.splitlines(keepends=True);new=[lines[0]];changes=[]
for line in lines[1:]:
 fields=line.split(b',');assert len(fields)==8
 pid=fields[0].decode()
 if pid in ids:
  assert float(fields[1])>=.05
  changes.append({'pair_id':pid,'previous_behavior':fields[2].decode(),'new_behavior':'other_coordination'})
  fields[2]=b'other_coordination';new.append(b','.join(fields))
 else:new.append(line)
assert len(changes)==970
p=OUT/'submission_other_conf065_risk005.csv';assert not p.exists();p.write_bytes(b''.join(new))
a=list(csv.reader(io.StringIO(raw.decode())));z=list(csv.reader(io.StringIO(p.read_text())))
assert len(a)==len(z)==112541
for before,after in zip(a,z):
 assert before[:2]+before[3:]==after[:2]+after[3:]
 assert before[2]==after[2] or (before[0] in ids and after[2]=='other_coordination')
assert sum(u!=v for u,v in zip(lines,p.read_bytes().splitlines(keepends=True)))==970
sys.path.insert(0,str(ROOT/'src'));from validation import validate_submission
validation=validate_submission(p,ROOT/'data/evaluation_pairs.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/sample_submission.csv').to_dict();write(OUT/'validation.json',validation);assert validation['valid']
assert sha(SOURCE)==protocol['base_sha256'] and sha(CONF)==protocol['confidence_sha256']
summary={'rows':112540,'changed_behavior_labels':970,'risk_ge_05':selected.filter(pl.col('risk_score')>=.5).height,'risk_ge_01':selected.filter(pl.col('risk_score')>=.1).height,'risk_min':selected['risk_score'].min(),'risk_max':selected['risk_score'].max(),'previous_family_counts':selected.group_by('predicted_behavior').len().to_dicts(),'candidate_sha256':sha(p),'protocol_sha256':sha(OUT/'protocol.json'),'builder_sha256':sha(Path(__file__)),'source_sha256':sha(SOURCE),'all_other_cells_identical':True,'unchanged_rows_byte_identical':True,'validation':validation,'submitted':False}
write(OUT/'build-receipt.json',summary);write(OUT/'changed-labels.json',changes)
(OUT/'README.md').write_text('# Broader exploratory fourth-type submission\n\nSubmit `submission_other_conf065_risk005.csv` after your quota resets. Base public score:0.89457. Exactly970 predicted_behavior labels change to other_coordination under the uniform rule: risk>=0.05 and conditional known-family maximum<0.65. All risks, evidence hands, pair IDs, row order and other rows are unchanged. Full submission validation passed.\n\nThis supersedes the earlier three-change file for the requested experiment. It is user-authorized exploration, not a proven improvement or identification of the fourth type. Thresholds were chosen from model-output counts to approach1000 changes, with no manual labels, case overrides, quality scores or raw-hand judgements. Of changed pairs,519 have risk>=0.10 and20 have risk>=0.50. It may improve, reduce or leave the public score unchanged; a delta does not reveal individual test labels.\n\nSuggested description: Exploratory behavior abstention: risk>=0.05 and conditional family max<0.65;970 behavior labels changed; all risks and evidence identical to best0.89457.\n\nThe file has not been submitted by Codex.\n',encoding='utf-8')
print(json.dumps({'path':str(p),**summary}))
