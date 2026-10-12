"""Fixed ledger-assisted development pilot. Synthetic cases; no competition records."""
from pathlib import Path
import copy,json,hashlib,importlib.util,random,datetime
R=Path(__file__).resolve().parent
OLD=R.parent/'teacher_mechanics_20260912'
spec=importlib.util.spec_from_file_location('fixture',OLD/'src/teacher_mechanics_fixtures.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
SCHEMA={'type':'object','additionalProperties':False,'required':['case_id','judgments'],'properties':{'case_id':{'type':'string'},'judgments':{'type':'array','minItems':3,'maxItems':3,'items':{'type':'object','additionalProperties':False,'required':['claim_id','verdict','record_ids'],'properties':{'claim_id':{'type':'string'},'verdict':{'enum':['supported','contradicted','undetermined']},'record_ids':{'type':'array','minItems':1,'maxItems':2,'uniqueItems':True,'items':{'type':'string'}}}}}}}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):
 with (R/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')
def build():
 old=json.loads((OLD/'fixtures/cases.json').read_text())['cases'];cases=[];requests=[];checks=[]
 for case_index,i in enumerate([0,3,4,7,8,11,1,10]):
  c=old[i]
  h=copy.deepcopy(c['mechanics_ledger']['hands'][0]);scale=[3,5,7][i%3]
  h['hand_id']=f'lg_hand_{i+1:02d}'
  h['small_blind']*=scale;h['big_blind']*=scale
  h['starting_stacks']={k:v*scale for k,v in h['starting_stacks'].items()}
  for j,x in enumerate(h['chronology']):
   x['line_id']=f'E{j+1}'
   if 'amount' in x:x['amount']*=scale
   if x.get('to_amount') is not None:x['to_amount']*=scale
  h['outcome']['pot']*=scale;h['outcome']['line_id']='AWARD'
  for ret in h['outcome']['returned_uncalled']:
   ret['amount']*=scale
   ret['line_id']=next(x['line_id'] for x in h['chronology'] if x['type']=='return' and x['actor']==ret['actor'])
  errors=fixture.validate_hand(h);assert not errors,(i,errors)
  contribution={k:0 for k in h['players']}
  for x in h['chronology']:
   if x['type']=='action':contribution[x['actor']]+=x['amount']
   elif x['type']=='return':contribution[x['actor']]-=x['amount']
  assert sum(contribution.values())==h['outcome']['pot']
  net={k:-v for k,v in contribution.items()};net[h['outcome']['winner']]+=h['outcome']['pot'];assert sum(net.values())==0
  decision=[x for x in h['chronology'] if x['type']=='action' and not x['action'].startswith('post_')]
  earlier,later=decision[-2:]
  checks.append({'case':i+1,'scale':scale,'legal':True,'zero_sum':True,'pot':h['outcome']['pot']})
  for variant in [False,True]:
   names={p:f'Player_{j}' for p,j in zip(h['players'],random.Random(91871+i+100*variant).sample(range(101,999),len(h['players'])))}
   records=[]
   for j,x in enumerate(h['chronology']):
    rec=copy.deepcopy(x);rec['order']=j+1
    if rec.get('actor'):rec['actor']=names[rec['actor']]
    records.append({'record_id':rec.pop('line_id'),'fact':rec})
   for p in h['players']:
    records.append({'record_id':f'ACCOUNT_{p}','fact':{'player':names[p],'starting_stack':h['starting_stacks'][p],'called_contribution':contribution[p],'net_chips':net[p],'ending_stack':h['starting_stacks'][p]+net[p]}})
   records.append({'record_id':'AWARD','fact':{'winner':names[h['outcome']['winner']],'called_pot':h['outcome']['pot'],'showdown':h['outcome']['showdown']}})
   records.append({'record_id':'OMITTED','fact':{'unavailable':['private_intent','private_communications','complete_hole_cards','off_excerpt_history']}})
   winner=h['outcome']['winner'];truth=(case_index%2==0)
   account_player=winner if case_index%4<2 else min(net,key=net.get)
   assert net[account_player]!=0
   stated_net=net[account_player] if truth else -net[account_player]
   claims=[{'claim_id':'C1','text':f"{names[account_player]}'s net chip change over the complete hand was {stated_net}.", 'expected':'supported' if truth else 'contradicted','proof':[f'ACCOUNT_{account_player}']},
           {'claim_id':'C2','text':f"The {later['street']} {later['action']} by {names[later['actor']]} happened {'before' if truth else 'after'} the {earlier['street']} {earlier['action']} by {names[earlier['actor']] }.",'expected':'contradicted' if truth else 'supported','proof':[earlier['line_id'],later['line_id']]},
           {'claim_id':'C3','text':f"{names[c['review_roles'][0]]} and {names[c['review_roles'][1]]} privately agreed to manipulate this hand.",'expected':'undetermined','proof':['OMITTED']}]
   # Identifiers do not disclose their semantic category; variants also permute presentation order.
   ids=random.Random(7261+i).sample(['C1','C2','C3'],3)
   for claim,cid in zip(claims,ids):claim['claim_id']=cid
   random.Random(9631+i+100*variant).shuffle(claims)
   if variant:records=list(reversed(records)) # Explicit chronology order is the semantic time source.
   case_id=f'lg_case_{i+1:02d}';rid=case_id+('_variant' if variant else '_source')
   data={'case_id':case_id,'hand_id':h['hand_id'],'review_pair':[names[k] for k in c['review_roles']],'records':records,'claims':[{k:v for k,v in x.items() if k in ['claim_id','text']} for x in claims]}
   prompt="""Check each claim against this original synthetic poker record. Return only compact JSON matching the entire schema below. Do not repeat the record or write explanations.
The ledger has already been calculated and verified by code. Net chip change means ending stack minus starting stack, after returns and pot award. Chronology is given by each action's numeric order, not the order records appear in this message. Supported means the record establishes the claim; contradicted means it establishes the opposite; undetermined means the record establishes neither. Unknown private intent must not be treated as innocent or guilty. Cite the 1-2 record IDs that justify each judgment. Return each claim exactly once. The verdict concerns only the stated claim, not general guilt.
SCHEMA: """+json.dumps(SCHEMA,separators=(',',':'))+"\nINPUT: "+json.dumps(data,separators=(',',':'))
   requests.append({'request_id':rid,'messages':[{'role':'user','content':prompt}]})
   cases.append({'request_id':rid,'case_id':case_id,'input':data,'expected':{x['claim_id']:{'verdict':x['expected'],'record_ids':x['proof']} for x in claims}})
 # Eight missing-record controls preserve each source claim but remove all accounting/action facts.
 full_cases=list(cases)
 for c in full_cases:
  if not c['request_id'].endswith('_source'):continue
  data=copy.deepcopy(c['input']);data['records']=[{'record_id':'OMITTED','fact':{'unavailable':['private_intent','private_communications','complete_hole_cards','off_excerpt_history','action_timeline','chip_accounting']}}]
  rid=c['request_id'].replace('_source','_missing')
  template=next(x['messages'][0]['content'] for x in requests if x['request_id']==c['request_id']).split('\nINPUT: ')[0]
  requests.append({'request_id':rid,'messages':[{'role':'user','content':template+'\nINPUT: '+json.dumps(data,separators=(',',':'))}]})
  cases.append({'request_id':rid,'case_id':c['case_id'],'input':data,'expected':{x['claim_id']:{'verdict':'undetermined','record_ids':['OMITTED']} for x in data['claims']}})
 assert len(requests)==24 and len(cases)==24
 write('cases.json',cases);write('schema.json',SCHEMA);write('fixture-checks.json',checks)
 with (R/'requests.jsonl').open('x',encoding='utf-8') as f:
  for x in requests:f.write(json.dumps(x,separators=(',',':'))+'\n')
 write('build-receipt.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_case_family_reuse':True,'fresh_aliases_amounts_ids_and_representation':True,'not_independent_holdout':True,'competition_records':0,'requests':24,'claims':72,'files_sha256':{n:sha(R/n) for n in ['cases.json','schema.json','fixture-checks.json','requests.jsonl','build.py']},'source_cases_sha256':sha(OLD/'fixtures/cases.json'),'source_validator_sha256':sha(OLD/'src/teacher_mechanics_fixtures.py')})
 print(json.dumps({'requests':24,'claims':72,'all_8_histories_legal':True,'max_input_chars':max(len(x['messages'][0]['content']) for x in requests)}))
if __name__=='__main__':build()
