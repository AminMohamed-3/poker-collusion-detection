"""All-row accounting, card, transition and causal-label checks."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
from collections import Counter
import json,hashlib
import polars as pl
H=Path(__file__).resolve().parent;D=H/'repaired'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((D/'pilot-plan.json').read_text());gen=json.loads((D/'pilot-generation.json').read_text())
    assert gen['plan_sha256']==sha(D/'pilot-plan.json')
    assert all(sha(D/n)==v for n,v in plan['source_hashes'].items())
    hands=pairs=positive=transitions=continuous=0;reasons=Counter();cells=[];fingerprints=set();source_players=Counter()
    allowed={'partner_check','partner_raise_suppressed','partner_low_pressure_preflop','partner_low_pressure_call','partner_multiway_check'}
    for r in gen['pools']:
        p=D/'pilot'/f"pool-{r['pool_id']:04d}"
        assert all(sha(p/n)==v for n,v in r['files'].items())
        h=pl.read_parquet(p/'hands.parquet');s=pl.read_parquet(p/'seats.parquet');a=pl.read_parquet(p/'actions.parquet');lab=pl.read_csv(p/'development_labels.csv');ev=pl.read_csv(p/'development_evidence.csv')
        pr=json.loads((p/'provenance.json').read_text());trace=json.loads((p/'interventions.json').read_text());source_players.update(pr['prototype_sources'].values())
        assert h.height==3000 and h['hand_id'].n_unique()==3000 and s.height==18000
        assert a.select(pl.struct('hand_id','action_no').n_unique()).item()==a.height
        accounting=s.group_by('hand_id').agg(pl.col('net_chips').sum().alias('net'),pl.col('total_contribution').sum().alias('contribution'),pl.col('won_share').sum().alias('shares')).join(h.select('hand_id','final_pot'),on='hand_id')
        assert accounting.filter((pl.col('net')!=0)|(pl.col('contribution')!=pl.col('final_pot'))|((pl.col('shares')-1).abs()>1e-8)).height==0
        assert not a.filter((pl.col('amount')<0)|(pl.col('amount')>pl.col('stack_before'))|(pl.col('to_call')<0)).height
        assert not s.filter((pl.col('starting_stack')+pl.col('net_chips')<0)|(pl.col('total_contribution')>pl.col('starting_stack'))).height
        payload=s.sort('hand_id','seat_no').group_by('hand_id',maintain_order=True).agg(pl.struct(pl.exclude('hand_id','player_id')).alias('seats')).join(h.select('hand_id','board_cards','big_blind'),on='hand_id')
        for row in payload.iter_rows(named=True):
            cards=row['board_cards'].split()+[c for seat in row['seats'] for c in [seat['hole_card_1'],seat['hole_card_2']]]
            assert len(cards)==len(set(cards))
            fp=hashlib.sha256(json.dumps([row['seats'],row['board_cards'],row['big_blind']],sort_keys=True).encode()).hexdigest();assert fp not in fingerprints;fingerprints.add(fp)
        resets={(r['player_id'],r['hand_number']):r for r in json.loads((p/'balance-resets.json').read_text())}
        ordered=s.join(h.select('hand_id','started_at'),on='hand_id').sort('player_id','started_at','hand_id').with_columns((pl.col('starting_stack')+pl.col('net_chips')).shift(1).over('player_id').alias('prior'))
        for row in ordered.select('hand_id','player_id','starting_stack','prior').iter_rows(named=True):
            if row['prior'] is None:continue
            reset=resets.pop((row['player_id'],int(row['hand_id'].split('H')[1])),None)
            if reset:assert reset['before']==row['prior'] and reset['after']==row['starting_stack']
            else:assert row['starting_stack']==row['prior']
            transitions+=1;continuous+=row['prior']==row['starting_stack']
        assert not resets
        acts={(r['hand_id'],r['action_no']):r for r in a.iter_rows(named=True)};supported=set()
        for row in trace:
            for t in row['interventions']:
                assert t['reason'] in allowed and t['chosen_intent']!=t['base_intent']
                actual=acts[(row['hand_id'],t['action_no'])]
                assert actual['player_id'] in pr['target'] and actual['action']==t['actual_action']
                assert actual['action'] in ['call','check','all_in']
                reasons[t['reason']]+=1;supported.add((row['pair_id'],row['hand_id']))
        for row in ev.iter_rows(named=True):assert (row['pair_id'],row['hand_id']) in supported
        for row in lab.iter_rows(named=True):
            assert pr['shared_hands'][row['pair_id']]>=50
            if row['label']:assert ev.filter(pl.col('pair_id')==row['pair_id']).height
        hands+=h.height;pairs+=lab.height;positive+=int(lab['label'].sum())
        if lab['label'].sum():cells.append({'policy_family':r['policy_family'],'soft_variant':pr['spec']['soft_variant'],'pool_id':r['pool_id'],'evidence_hands':ev.height})
    result={'hands':hands,'pairs':pairs,'positive_pairs':positive,'players':len(gen['pools'])*30,'all_accounting_cards_transitions_and_supported_labels_passed':True,'unique_hand_payloads':len(fingerprints),'exact_balance_continuation':continuous/transitions,'transitions':transitions,'supported_cells':cells,'policy_families':len({r['policy_family'] for r in cells}),'reasons':dict(reasons),'prototype_sources':len(source_players),'maximum_prototype_reuse':max(source_players.values()),'source_sha256':sha(__file__),'not_a_full_behavior_coverage_test':True}
    (D/'pilot-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='supported_cells'},indent=2))
if __name__=='__main__':main()
