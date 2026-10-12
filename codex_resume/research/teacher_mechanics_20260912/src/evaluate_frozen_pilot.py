"""Evaluate one complete, locally backed-up teacher pilot without refits or repairs."""
from pathlib import Path
import json,hashlib,importlib.util
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n')
def main():
 spec=importlib.util.spec_from_file_location('gate',R/'src/teacher_mechanics_evaluate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
 review=json.loads((R/'root-review-and-freeze.json').read_text())
 for name,digest in review['files_sha256'].items():assert sha(R/name)==digest,name
 assert gate.verify_controls()['reference_cases']==12
 out=R/'runtime/returned/pilot';complete=json.loads((out/'complete.json').read_text());assert complete['requests']==24
 for name,digest in complete['files_sha256'].items():assert sha(out/name)==digest,name
 requests=[json.loads(x) for x in (R/'runtime/pilot-requests.jsonl').read_text().splitlines()]
 assert complete['requests_sha256']==sha(R/'runtime/pilot-requests.jsonl')
 assert complete['source_sha256']==sha(R.parents[1]/'src/teacher_mechanics_infer_v3.py')
 files=sorted(out.glob('[0-9][0-9][0-9].json'));assert len(files)==24
 rows=[json.loads(p.read_text()) for p in files];assert [x['request_id'] for x in rows]==[x['request_id'] for x in requests]
 freeze={'complete_sha256':sha(out/'complete.json'),'root_review_sha256':sha(R/'root-review-and-freeze.json'),'evaluator_sha256':sha(Path(__file__)),'output_files':{p.name:sha(p) for p in files},'invocation':1}
 write(R/'factuality-started.json',freeze)
 cases={x['case_id']:x for x in json.loads((R/'fixtures/cases.json').read_text())['cases']};details=[];responses={}
 for request,row in zip(requests,rows):
  result=gate.evaluate(cases[request['case_id']],row['raw_text'],request['variant'])
  if not row['ended_with_eos']:result['errors'].append('output incomplete: non-EOS stopping')
  if not row['guard_pass']:result['errors'].append('memory guard failed')
  result['pass']=not result['errors']
  responses[request['request_id']]=result['response']
  details.append({'request_id':request['request_id'],'pass':result['pass'],'errors':result['errors'],'input_tokens':row['input_tokens'],'output_tokens':row['output_tokens'],'seconds':row['seconds']})
 invariance=[]
 for key,case in cases.items():
  a=responses[key+'_source'];b=responses[key+'_variant']
  try:errors=gate.invariance(case,a,b)
  except (KeyError,TypeError,ValueError):errors=['invalid response prevents invariance comparison']
  invariance.append({'case_id':key,'pass':not errors,'errors':errors})
 passed=sum(x['pass'] for x in details);iv=sum(x['pass'] for x in invariance)
 result={'factuality_passes':passed,'responses':24,'invariance_passes':iv,'invariance_cases':12,'mechanics_gate_pass':passed==24 and iv==12 and complete['all_responses_ended_with_eos'],'semantic_prose_review':'pending root review; valid citations alone do not prove prose truth','teacher_utility_proven':False,'leaderboard_gain_proven':False,'details':details,'invariance':invariance,'peak_reserved_bytes':max(x['peak_reserved'] for x in rows),'generation_seconds':sum(x['seconds'] for x in rows),'total_job_seconds':complete['seconds'],'all_outputs_backed_up':True}
 write(R/'factuality-results.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['details','invariance']}))
if __name__=='__main__':main()
