"""One-shot frozen evaluation; no new generation or scoring repairs."""
from pathlib import Path
import json,hashlib,copy,datetime
from collections import Counter
from jsonschema import Draft202012Validator
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):
 with (R/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2);f.write('\n')
def judge(case,raw):
 try:x=json.loads(raw)
 except (ValueError,TypeError):return {'pass':False,'errors':['invalid JSON'],'correct':0,'parsed':None}
 errs=[e.message for e in Draft202012Validator(json.loads((R/'schema.json').read_text())).iter_errors(x)]
 if errs:return {'pass':False,'errors':errs,'correct':0,'parsed':x}
 if x['case_id']!=case['case_id']:errs.append('wrong case ID')
 ids=[j['claim_id'] for j in x['judgments']]
 if len(set(ids))!=3 or set(ids)!=set(case['expected']):errs.append('claim IDs do not match exactly once')
 valid_ids={r['record_id'] for r in case['input']['records']};correct=0
 for j in x['judgments']:
  expected=case['expected'].get(j['claim_id']);error=[]
  if expected is None:error.append('unknown claim')
  else:
   if j['verdict']!=expected['verdict']:error.append('wrong verdict')
   if not set(expected['record_ids']).issubset(j['record_ids']):error.append('missing grounding records')
  if not set(j['record_ids']).issubset(valid_ids):error.append('invented record ID')
  if error:errs.append(j['claim_id']+': '+', '.join(error))
  else:correct+=1
 return {'pass':not errs,'errors':errs,'correct':correct,'parsed':x}
def ref(c):return {'case_id':c['case_id'],'judgments':[{'claim_id':k,**v} for k,v in c['expected'].items()]}
def controls():
 cases=json.loads((R/'cases.json').read_text());assert len(cases)==24
 assert all(judge(c,json.dumps(ref(c)))['pass'] for c in cases)
 c=cases[0];a=ref(c)
 b=copy.deepcopy(a);b['judgments'][0]['verdict']='undetermined';assert not judge(c,json.dumps(b))['pass']
 b=copy.deepcopy(a);b['judgments'][0]['record_ids']=['MADE_UP'];assert not judge(c,json.dumps(b))['pass']
 b=copy.deepcopy(a);b['judgments'][1]=b['judgments'][0];assert not judge(c,json.dumps(b))['pass']
 assert not judge(c,'broken {')['pass']
 abstain=0
 for c in cases:
  a=ref(c)
  for j in a['judgments']:j['verdict']='undetermined'
  abstain+=judge(c,json.dumps(a))['correct']
 assert abstain==40
 return {'reference_responses':24,'always_abstain_correct':40,'total_claims':72,'malformed_duplicate_wrong_citation_wrong_verdict_rejected':True}
def main():
 freeze=json.loads((R/'freeze.json').read_text())
 for n,d in freeze['files_sha256'].items():assert sha(R/n)==d,n
 ctrl=controls();out=R/'returned/pilot';comp=json.loads((out/'complete.json').read_text())
 assert comp['requests']==24 and comp['requests_sha256']==sha(R/'requests.jsonl')
 assert comp['source_sha256']==freeze['runner_sha256']
 for n,d in comp['files_sha256'].items():assert sha(out/n)==d,n
 cases=json.loads((R/'cases.json').read_text());files=sorted(out.glob('[0-9][0-9][0-9].json'));assert len(files)==24
 rows=[json.loads(p.read_text()) for p in files];assert [x['request_id'] for x in rows]==[x['request_id'] for x in cases]
 write('evaluation-started.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'invocation':1,'freeze_sha256':sha(R/'freeze.json'),'complete_sha256':sha(out/'complete.json')})
 detail=[];parsed={};cats=Counter();correct_cats=Counter()
 for c,x in zip(cases,rows):
  j=judge(c,x['raw_text']);parsed[c['request_id']]=j['parsed']
  if not x['ended_with_eos'] or not x['guard_pass']:j['errors'].append('incomplete response or memory guard failure');j['pass']=False
  detail.append({'request_id':x['request_id'],'pass':j['pass'],'correct_grounded_claims':j['correct'],'errors':j['errors']})
  for claim_id,e in c['expected'].items():
   cats[e['verdict']]+=1
   if j['parsed'] and isinstance(j['parsed'].get('judgments'),list):
    matches=[v for v in j['parsed']['judgments'] if isinstance(v,dict) and v.get('claim_id')==claim_id]
    if len(matches)==1 and matches[0].get('verdict')==e['verdict']:correct_cats[e['verdict']]+=1
 iv=[]
 for c in cases:
  if not c['request_id'].endswith('_source'):continue
  other=c['request_id'].replace('_source','_variant');ids=[c['request_id'],other]
  pair=[next(d for d in detail if d['request_id']==n) for n in ids]
  same=all(d['pass'] for d in pair)
  iv.append({'case_id':c['case_id'],'both_renderings_pass':same})
 result={'responses':24,'claims':72,'responses_pass':sum(x['pass'] for x in detail),'grounded_claims_correct':sum(x['correct_grounded_claims'] for x in detail),'verdict_only_correct_by_class':dict(correct_cats),'claims_by_class':dict(cats),'invariance_passes':sum(x['both_renderings_pass'] for x in iv),'primary_pass':all(x['pass'] for x in detail) and all(x['both_renderings_pass'] for x in iv),'controls':ctrl,'seconds':comp['seconds'],'peak_reserved_bytes':max(x['peak_reserved'] for x in rows),'student_utility_proven':False,'leaderboard_gain_proven':False,'details':detail,'invariance':iv}
 write('results.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['details','invariance']}))
if __name__=='__main__':main()
