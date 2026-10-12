"""Audit every serialized stack transition against explicit simulated reset records."""
from pathlib import Path
import hashlib,json,sys
import polars as pl
H=Path(__file__).resolve().parent;D=H/'persistent_sessions'
sys.path.insert(0,str(H.parent/'data_expansion'))
import audit_pilot
audit_pilot.HERE=D
audit_pilot.main()
seen=0;continued=0;reset_count=0;rows=[];cells=set();pool_weights=[];family_weights={}
for d in sorted((D/'pilot').glob('pool-*')):
    h=pl.read_parquet(d/'hands.parquet').select('hand_id','started_at','big_blind');s=pl.read_parquet(d/'seats.parquet')
    f=s.join(h,on='hand_id').sort('player_id','started_at','hand_id').with_columns((pl.col('starting_stack')+pl.col('net_chips')).shift(1).over('player_id').alias('prior_balance'))
    reset=json.loads((d/'balance-resets.json').read_text())
    reset_index={(r['player_id'],r['hand_number']):r for r in reset}
    n=0;nc=0
    for row in f.select('player_id','hand_id','starting_stack','prior_balance','big_blind').iter_rows(named=True):
        if row['prior_balance'] is None:continue
        key=(row['player_id'],int(row['hand_id'].split('H')[1]));r=reset_index.pop(key,None);n+=1
        if r:
            assert r['before']==row['prior_balance'] and r['after']==row['starting_stack']
            assert row['starting_stack']/row['big_blind'] in [80,100,120,150]
            boundary=not 40*row['big_blind']<=r['before']<=250*row['big_blind']
            assert boundary==(r['reason']=='boundary')
        else:assert row['starting_stack']==row['prior_balance']
        nc+=row['starting_stack']==row['prior_balance']
    assert not reset_index
    labels=pl.read_csv(d/'development_labels.csv');e=pl.read_csv(d/'development_evidence.csv');p=json.loads((d/'provenance.json').read_text())
    assert all(p['shared_hands'][pid]>=50 for pid in labels['pair_id'])
    assert set(e['pair_id'])<=set(labels.filter(pl.col('label')==1)['pair_id'])
    for pid in labels.filter(pl.col('label')==1)['pair_id']:assert e.filter(pl.col('pair_id')==pid).height>0
    assert not e.join(h.select('hand_id'),on='hand_id',how='anti').height
    family=p['spec']['policy_family'];behavior=p['spec']['behavior']
    if behavior=='none' or labels['label'].sum()>0:cells.add((family,behavior))
    pool_weights.append(labels.height);family_weights[family]=family_weights.get(family,0)+labels.height
    seen+=n;continued+=nc;reset_count+=len(reset);rows.append({'pool':d.name,'transitions':n,'exact_continuation':nc,'recorded_resets':len(reset)})
ratio=continued/seen
r={'all_stack_transitions_verified':True,'transitions':seen,'recorded_resets':reset_count,'exact_balance_continuation_fraction':ratio,'original_fraction':.967974495464757,'continuation_gap':abs(ratio-.967974495464757),'continuation_gate_passed':abs(ratio-.967974495464757)<=.015,'emitted_cells':len(cells),'maximum_pool_weight':max(pool_weights)/sum(pool_weights),'maximum_policy_family_weight':max(family_weights.values())/sum(pool_weights),'coverage_passed':len(cells)==16 and max(pool_weights)/sum(pool_weights)<=.04 and max(family_weights.values())/sum(pool_weights)<=.3,'pools':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(D/'balance-audit.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='pools'},indent=2))
