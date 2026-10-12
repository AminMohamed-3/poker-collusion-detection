"""Root factuality gate: strict JSON schema plus order-insensitive set fields."""
from pathlib import Path
import json,copy,importlib.util
from jsonschema import Draft202012Validator
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fixed_fixture',R/'src/teacher_mechanics_fixtures.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
schema=Draft202012Validator(json.loads((R/'fixtures/response_schema.json').read_text()))
def normalized(case,response):
 case=copy.deepcopy(case);response=copy.deepcopy(response)
 case['expected_facts']['unavailable_facts']=sorted(case['expected_facts']['unavailable_facts'])
 response['unavailable_facts']=sorted(response['unavailable_facts']);response['mechanism_assessment']['observable_tags']=sorted(response['mechanism_assessment']['observable_tags'])
 return case,response

def evaluate(case,text,variant=False):
 try: response=json.loads(text) if isinstance(text,str) else text
 except (ValueError,TypeError) as exc:return {'pass':False,'errors':['invalid JSON'],'response':None}
 errors=[f"schema {list(e.absolute_path)}: {e.message}" for e in schema.iter_errors(response)]
 if errors:return {'pass':False,'errors':errors,'response':response}
 c,response=normalized(case,response)
 errors=fixture.evaluate_response(c,response,variant)
 return {'pass':not errors,'errors':errors,'response':response}

def invariance(case,a,b):
 ra=evaluate(case,a,False);rb=evaluate(case,b,True)
 if not ra['pass'] or not rb['pass']:return ['one or both renderings fail factuality']
 c,a=normalized(case,ra['response']);_,b=normalized(case,rb['response'])
 return fixture.evaluate_invariance(c,a,b)

def verify_controls():
 cases=json.loads((R/'fixtures/cases.json').read_text())['cases'];checks=[]
 for c in cases:
  a=fixture.reference_response(c);b=fixture.reference_response(c,True)
  assert evaluate(c,a)['pass'] and evaluate(c,b,True)['pass'] and not invariance(c,a,b)
  x=copy.deepcopy(a);x['unavailable_facts'].reverse();assert evaluate(c,x)['pass']
  checks.append(c['case_id'])
 c=cases[0];a=fixture.reference_response(c)
 x=copy.deepcopy(a);x['observed_facts']['actions'][-1]['amount']=False;assert not evaluate(c,x)['pass']
 x=copy.deepcopy(a);x['cited_line_ids']=[{}];assert not evaluate(c,x)['pass']
 x=copy.deepcopy(a);x['unavailable_facts'].append(x['unavailable_facts'][0]);assert not evaluate(c,x)['pass']
 assert not evaluate(c,'truncated {')['pass']
 return {'reference_cases':len(checks),'unordered_facts_accepted':True,'boolean_as_number_rejected':True,'malformed_ID_types_rejected':True,'duplicate_fact_names_rejected':True,'invalid_JSON_rejected':True}
if __name__=='__main__':print(json.dumps(verify_controls()))
