import os
os.environ['POLARS_MAX_THREADS']='2'
import sys,csv,io,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import polars as pl
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent; R=OUT.parent; ROOT=R.parent
SOURCE=R/'outputs_other_broad_exploratory_20260913/submission_other_conf065_risk005.csv'
CONF=R/'research/other_coordination_broad_confidence_20260913/all-family-posteriors.parquet'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):
 assert not p.exists(), str(p)
 p.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
assert sha(SOURCE)=='4a06d37d559ec4761f9aa2c29dff708a809e9b349d2b418a8e91901110290ec7'
assert sha(CONF)=='fbed361474cfaf092c3725b5d225e6d49e1525976a6adc909bae95ca62002b91'
x=pl.read_parquet(CONF); assert x.height==x['pair_id'].n_unique()==112540
eligible=x.filter(pl.col('conditional_max')<.65).sort('risk_score',descending=True)
inside=float(eligible['risk_score'][9999]); outside=float(eligible['risk_score'][10000]); assert inside>outside
cutoff=(inside+outside)/2
selected=eligible.filter(pl.col('risk_score')>=cutoff)
ids=set(selected['pair_id'].to_list()); assert len(ids)==10000
protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'user_authorized_exploratory_exception':True,'user_request':'Make the threshold result in 10k Other pairs, provide the file path, then return to ongoing work.','source_public_score_user_reported':.89463,'source_sha256':sha(SOURCE),'confidence_sha256':sha(CONF),'total_other_target':10000,'rule':{'risk_at_least':cutoff,'conditional_known_family_max_strictly_below':.65,'replacement':'other_coordination'},'boundary':{'last_selected_risk':inside,'first_unselected_risk':outside,'ties_at_boundary':False},'threshold_selection':'Keep the previous .65 confidence ceiling. Choose the risk midpoint between the 10000th and 10001st eligible rows to meet the user-requested total. Counts and stored model outputs only; no labels, hand judgements, case-specific overrides or local quality metrics.','adaptivity_disclosed':'The user requested this new exploratory candidate after reporting .89463 for the previous 970-label trial. This is adaptive public exploration and is not locally qualified.','prior_other_labels_expected':970,'new_changes_expected':9030,'all_risks_and_evidence_preserved':True,'previous_test_exposure_disclosed':'14 pairs were descriptively inspected earlier; their IDs and opinions are not used in this uniform rule.','submitted':False,'feedback_scope':'Aggregate public score only, with no inference of individual test labels.'}
write(OUT/'protocol.json',protocol)
raw=SOURCE.read_bytes(); lines=raw.splitlines(keepends=True); new=[lines[0]]; changes=[]; old_other=set()
for line in lines[1:]:
 fields=line.split(b','); assert len(fields)==8
 pid=fields[0].decode()
 if fields[2]==b'other_coordination': old_other.add(pid)
 if pid in ids:
  assert float(fields[1])>=cutoff
  if fields[2]!=b'other_coordination':
   changes.append({'pair_id':pid,'previous_behavior':fields[2].decode(),'new_behavior':'other_coordination'})
   fields[2]=b'other_coordination';new.append(b','.join(fields))
  else: new.append(line)
 else: new.append(line)
assert len(old_other)==970 and old_other<=ids
assert len(changes)==9030
p=OUT/'submission_other_conf065_total10000.csv';assert not p.exists();p.write_bytes(b''.join(new))
a=list(csv.reader(io.StringIO(raw.decode())));z=list(csv.reader(io.StringIO(p.read_text())))
assert len(a)==len(z)==112541
assert sum(v[2]=='other_coordination' for v in z[1:])==10000
for before,after in zip(a,z):
 assert before[:2]+before[3:]==after[:2]+after[3:]
 assert before[2]==after[2] or (before[0] in ids and after[2]=='other_coordination')
assert sum(u!=v for u,v in zip(lines,p.read_bytes().splitlines(keepends=True)))==9030
sys.path.insert(0,str(ROOT/'src')); from validation import validate_submission
validation=validate_submission(p,ROOT/'data/evaluation_pairs.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/sample_submission.csv').to_dict()
write(OUT/'validation.json',validation);assert validation['valid']
assert sha(SOURCE)==protocol['source_sha256'] and sha(CONF)==protocol['confidence_sha256']
receipt={'rows':112540,'total_other_labels':10000,'changed_behavior_labels_vs_89463':9030,'rule':protocol['rule'],'all_other_cells_identical':True,'unchanged_rows_byte_identical':True,'candidate_sha256':sha(p),'source_sha256':sha(SOURCE),'protocol_sha256':sha(OUT/'protocol.json'),'builder_sha256':sha(Path(__file__)),'validation':validation,'submitted':False}
write(OUT/'build-receipt.json',receipt);write(OUT/'changed-labels.json',changes)
(OUT/'README.md').write_text('# Exploratory 10000-Other submission\n\nFile: submission_other_conf065_total10000.csv\n\nExactly 10000 total other_coordination labels, including the existing 970 from the user-reported 0.89463 entry. This changes 9030 additional behavior labels. All risk scores, evidence hands, pair IDs, row order, and unaffected rows are preserved. Full submission validation passed.\n\nUniform rule: conditional known-family maximum < 0.65 and risk >= '+repr(cutoff)+'. The risk cutoff is chosen solely to meet the requested count, with no boundary ties. This is an exploratory trial following prior public feedback, not a locally established improvement. No new training or test-label inference was performed.\n\nNot submitted by Codex. Suggested description: Exploratory 10000 total Other labels; confidence < 0.65 with risk cutoff selected by count; 9030 changes versus 0.89463; all risks and evidence unchanged.\n',encoding='utf-8')
print(json.dumps({'path':str(p),**receipt}))
