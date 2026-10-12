"""Post-hoc field diagnostics only; never changes the frozen primary result."""
from pathlib import Path
import json,importlib.util,hashlib
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fixture',R/'src/teacher_mechanics_fixtures.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
cases={c['case_id']:c for c in json.loads((R/'fixtures/cases.json').read_text())['cases']}
rows=[]
for p in sorted((R/'runtime/returned/pilot').glob('[0-9][0-9][0-9].json')):
 x=json.loads(p.read_text());rid=x['request_id'];c=cases[rid.rsplit('_',1)[0]];expected=fixture.reference_response(c,rid.endswith('_variant'))['observed_facts']
 row={'request_id':rid}
 try:a=json.loads(x['raw_text'])
 except json.JSONDecodeError:
  row['valid_json']=False;rows.append(row);continue
 f=a.get('observed_facts',{});o=f.get('outcomes');outcomes=o if isinstance(o,list) else [o]
 row.update(valid_json=True,actions_exact=f.get('actions')==expected['actions'],net_chips_exact=f.get('net_chips')==expected['net_chips'],outcomes_content_exact=outcomes==expected['outcomes'])
 row['net_chips_observed_sum']=sum(f['net_chips'].values()) if isinstance(f.get('net_chips'),dict) and all(isinstance(v,(int,float)) for v in f['net_chips'].values()) else None
 rows.append(row)
result={'post_hoc':True,'not_an_admission_gate':True,'no_repairs_to_invalid_json':True,'outcome_object_wrapped_in_list_only_for_diagnostic_content_comparison':True,'primary_result_unchanged':True,'valid_json':sum(x['valid_json'] for x in rows),'total_responses':len(rows),'actions_exact':sum(x.get('actions_exact',False) for x in rows),'net_chips_exact':sum(x.get('net_chips_exact',False) for x in rows),'outcomes_content_exact':sum(x.get('outcomes_content_exact',False) for x in rows),'details':rows}
with (R/'posthoc-field-diagnostics.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='details'}))
