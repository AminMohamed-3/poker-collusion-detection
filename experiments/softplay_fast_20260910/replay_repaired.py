"""Independent replay of serialized sampled hands, without calling any player policy."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import hashlib,json,sys
import polars as pl
H=Path(__file__).resolve().parent;D=H/'repaired'
sys.path.insert(0,str(H.parent/'history_realism_20260910/vendor'))
from pokerkit import Automation,NoLimitTexasHoldem
AUTO=(Automation.ANTE_POSTING,Automation.BET_COLLECTION,Automation.BLIND_OR_STRADDLE_POSTING,Automation.HOLE_CARDS_SHOWING_OR_MUCKING,Automation.HAND_KILLING,Automation.CHIPS_PUSHING,Automation.CHIPS_PULLING)
actions_checked=0;hands_checked=0
for folder in sorted((D/'pilot').glob('pool-*')):
    h=pl.read_parquet(folder/'hands.parquet').with_columns(pl.col('started_at').cast(pl.String)).with_columns(pl.col('hand_id').hash(seed=71583).alias('_sample')).sort('_sample').head(10)
    s=pl.read_parquet(folder/'seats.parquet');a=pl.read_parquet(folder/'actions.parquet')
    for hand in h.iter_rows(named=True):
        seats=s.filter(pl.col('hand_id')==hand['hand_id']).sort('seat_no').to_dicts();records=a.filter(pl.col('hand_id')==hand['hand_id']).sort('action_no').to_dicts()
        seats.sort(key=lambda r:(r['seat_no']-hand['button_seat']-1)%6)
        stacks=[r['starting_stack'] for r in seats];players=[r['player_id'] for r in seats];blind=hand['big_blind'];board=hand['board_cards'].split()
        state=NoLimitTexasHoldem.create_state(AUTO,True,0,(blind//2,blind),blind,stacks,6)
        for i,seat in enumerate(seats):state.deal_hole(seat['hole_card_1']+seat['hole_card_2'],i)
        known={c for s in seats for c in [s['hole_card_1'],s['hole_card_2']]}|set(board)
        burns=iter(c for r in '23456789TJQKA' for suit in 'cdhs' if (c:=r+suit) not in known)
        used_board=0;folded=set();index=0
        while state.status:
            if state.can_burn_card():state.burn_card(next(burns));continue
            if state.can_deal_board():
                n=state.board_dealing_count;state.deal_board(''.join(board[used_board:used_board+n]));used_board+=n;continue
            row=records[index];actor=state.actor_index;assert players[actor]==row['player_id']
            assert row['street']==['preflop','flop','turn','river'][state.street_index]
            assert row['pot_before']==state.total_pot_amount and row['stack_before']==state.stacks[actor]
            assert row['to_call']==max(state.bets)-state.bets[actor] and row['players_active']==6-len(folded)
            old=state.stacks[actor];before=state.bets[actor]
            if row['action']=='fold':state.fold();folded.add(actor);paid=0
            elif row['action'] in ['bet','raise'] or (row['action']=='all_in' and row['amount']>row['to_call']):
                op=state.complete_bet_or_raise_to(row['amount_to']);paid=op.amount-before
            else:paid=state.check_or_call().amount
            assert paid==row['amount'] and row['amount_to']==before+paid
            actions_checked+=1;index+=1
        assert index==len(records) and used_board==len(board)
        for i,seat in enumerate(seats):assert state.stacks[i]-stacks[i]==seat['net_chips']
        assert sum(r['amount'] for r in records)+blind+blind//2==hand['final_pot']
        hands_checked+=1
result={'hands':hands_checked,'pools':len(list((D/'pilot').glob('pool-*'))),'actions':actions_checked,'all_serialized_actions_replayed_legally':True,'all_preaction_states_and_final_net_chips_exact':True,'original_gross_pot_convention_verified':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(D/'replay-counted-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
