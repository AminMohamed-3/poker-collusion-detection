"""Label-free feasibility for all-in / contestable-pot state in current shortlist."""
from audit_common import *
from datetime import datetime, timezone
OUT=HERE/'research'/'contestable_pot_feasibility_20260912'
OUT.mkdir(parents=True,exist_ok=True)
saved=pl.read_parquet(PREV/'scores.parquet').sort('pair_id','hand_id')
a=saved.with_columns(pl.col('stage1_only').rank('ordinal',descending=True).over('pair_id').alias('rank'))
keys=a.filter(pl.col('rank')<=20).select('pair_id','hand_id')
pairs=pl.read_csv(ROOT/'data'/'development_labels.csv',columns=['pair_id','player_1','player_2']).join(keys.select('pair_id').unique(),on='pair_id',how='semi')
hids=keys['hand_id'].unique().implode()
acts=pl.scan_parquet(ROOT/'data'/'actions.parquet').filter(pl.col('hand_id').is_in(hids)).sort('hand_id','action_no').collect()
seats=pl.scan_parquet(ROOT/'data'/'seats.parquet').filter(pl.col('hand_id').is_in(hids)).collect()
pmap={r['pair_id']:(r['player_1'],r['player_2']) for r in pairs.iter_rows(named=True)}
amap={k[0]:g.to_dicts() for k,g in acts.group_by('hand_id')}
smap={k[0]:{r['player_id']:r for r in g.iter_rows(named=True)} for k,g in seats.group_by('hand_id')}
rows=[]
for pid,hid in keys.iter_rows():
    p1,p2=pmap[pid]; s=smap[hid]; live=set(s); allin=set(); first_allin=None
    stats={'any_allin':0,'pair_allin':0,'outsider_allin':0,'pair_decision_with_partner_allin':0,'pair_decision_with_any_allin':0,'call_price_capped':0,'pair_stack_ratio':min(s[p1]['starting_stack'],s[p2]['starting_stack'])/max(s[p1]['starting_stack'],s[p2]['starting_stack'])}
    for event in amap[hid]:
        actor=event['player_id']; partner=p2 if actor==p1 else p1
        if actor in (p1,p2) and partner in live:
            stats['pair_decision_with_partner_allin']+=int(partner in allin)
            stats['pair_decision_with_any_allin']+=int(bool(allin & live))
            stats['call_price_capped']+=int(event['to_call']>event['stack_before'])
        if event['action']=='fold': live.discard(actor)
        if event['amount']>=event['stack_before'] and event['stack_before']>0:
            allin.add(actor); stats['any_allin']=1
            stats['pair_allin' if actor in (p1,p2) else 'outsider_allin']=1
    rows.append({'pair_id':pid,'hand_id':hid,**stats})
f=pl.DataFrame(rows); f.write_parquet(OUT/'shortlist-state.parquet')
receipt={'created_at':datetime.now(timezone.utc).isoformat(),'label_free':True,'pairs':keys['pair_id'].n_unique(),'pair_hands':keys.height,'distinct_hands':keys['hand_id'].n_unique(),'action_rows':acts.height,'rates':{c:{'affected_pair_hands':f.filter(pl.col(c)>0).height,'fraction':f.filter(pl.col(c)>0).height/f.height} for c in stats if c!='pair_stack_ratio'},'stack_ratio_quantiles':{str(q):f['pair_stack_ratio'].quantile(q) for q in (.1,.25,.5,.75,.9)},'sources':{str(p):sha(p) for p in (Path(__file__),PREV/'scores.parquet')},'limitations':['No relevance or outer truth used; frequencies do not demonstrate learnability or utility.','All-in state tracks observed exhausted stacks; it does not assert every such hand has a side pot.','Partner all-in removes betting opportunities but does not remove showdown equity.']}
write(OUT/'receipt.json',receipt); print(json.dumps(receipt,indent=2))
