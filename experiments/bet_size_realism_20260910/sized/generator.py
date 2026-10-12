"""Fresh, rules-engine-checked six-player histories with explicit interventions.

All ordinary policies receive an Observation containing only their own cards
and public state. The coordination controller is separate and can inspect only
the cooperating player's present cards, never the remaining deck or future board.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'history_realism_20260910/vendor'))
from pokerkit import Automation, NoLimitTexasHoldem, BetCollection, ChipsPushing
from phevaluator import evaluate_cards

DECK = [r+s for r in '23456789TJQKA' for s in 'cdhs']
STREETS = ['preflop','flop','turn','river']
AUTOMATIONS = (Automation.ANTE_POSTING, Automation.BET_COLLECTION,
              Automation.BLIND_OR_STRADDLE_POSTING,
              Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
              Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING)

@dataclass(frozen=True)
class Observation:
    hole: tuple[str,str]
    board: tuple[str,...]
    street: int
    position: int
    pot: int
    stack: int
    to_call: int
    blind: int
    active: int
    raises: int
    previous_aggressor: int | None
    actor: int
    prior_actions: int = 0
    own_history: tuple = ()
    public_history: tuple = ()
    base_action_probabilities: tuple = ()
    can_raise: bool = True
    own_past_style: tuple = (.25,.25,.25,.25)

@dataclass(frozen=True)
class Profile:
    family: int
    tightness: float
    aggression: float
    noise: float
    bluff: float
    sizing: float
    style: tuple = (.6,.07,.13,.2)

def strength(hole, board):
    ranks = sorted(['23456789TJQKA'.index(c[0])+2 for c in hole])
    lo,hi=ranks
    if not board:
        return min(.98, .5+hi/30) if lo==hi else max(.03,min(.93,(hi+lo)/32-max(hi-lo-1,0)/40+.07*(hole[0][1]==hole[1][1])))
    return (7463-evaluate_cards(*hole,*board))/7462

def equity_proxy(obs, rng, draws=12):
    """Monte Carlo against unknown random opponents; never actual other cards."""
    unknown=[c for c in DECK if c not in obs.hole and c not in obs.board]
    opponents=max(1,min(3,obs.active-1)); win=0.
    for _ in range(draws):
        sample=rng.sample(unknown,5-len(obs.board)+2*opponents)
        board=(*obs.board,*sample[:5-len(obs.board)])
        offset=5-len(obs.board); own=evaluate_cards(*obs.hole,*board)
        ranks=[evaluate_cards(*sample[offset+2*j:offset+2*j+2],*board) for j in range(opponents)]
        if own<=min(ranks): win+=1/(1+sum(r==own for r in ranks))
    return win/draws

def base_decision(obs, profile, rng):
    from learned_policy import decision_with_probabilities
    return decision_with_probabilities(obs,profile,rng)


def intervene(obs, proposal, partner_hole, partner_live, partner_index, family, intensity, rng, donor):
    """Return action intent and an intervention reason, separate from normal policy."""
    if not partner_live or rng.random()>intensity:
        return proposal,None
    action,size=proposal
    facing=obs.previous_aggressor==partner_index
    # Soft play below uses only public state. Private partner strength is used
    # solely by the inherited directed-transfer controller.
    own=partner=0.
    if family=='directed_transfer':
        own=strength(obs.hole,obs.board); partner=strength(partner_hole,obs.board)
    if family=='directed_transfer':
        if donor and facing and partner>own+.03 and obs.to_call>0:
            return ('call',size),'donor_calls_partner_while_behind'
        if not donor and partner<own-.03 and obs.raises<3:
            return ('raise',max(size,.65)),'recipient_value_bet'
        if donor and facing and own>partner+.15 and obs.to_call>0 and obs.street>=1:
            return ('fold',size),'donor_folds_stronger_hand'
    elif family=='soft_play':
        from learned_policy import SOFT_VARIANT
        from soft_controller import decide
        return decide(obs,proposal,partner_index,rng,SOFT_VARIANT,'learned')
    elif family=='coordinated_isolation':
        if obs.active>=3 and obs.raises<3:
            return ('raise',max(size,.6)),'pressure_outsiders'
        if obs.active==2 and obs.to_call==0:
            return ('call',size),'partner_conflict_suppressed'
    return proposal,None

def resolve_decision(state, decision):
    action,fraction=decision
    i=state.actor_index
    assert i is not None
    to_call=state.checking_or_calling_amount
    if action=='fold' and state.can_fold() and to_call:
        return 'fold',0
    if action=='raise' and state.can_complete_bet_or_raise_to():
        low=state.min_completion_betting_or_raising_to_amount
        high=state.max_completion_betting_or_raising_to_amount
        desired=max(state.bets)+max(1,round(fraction*(state.total_pot_amount+to_call)))
        amount=min(high,max(low,desired))
        return 'raise',amount
    return 'call',to_call

def execute_decision(state, resolved):
    action,amount=resolved
    if action=='fold': return state.fold()
    if action=='raise': return state.complete_bet_or_raise_to(amount)
    return state.check_or_call()

def simulate_hand(seed, pool_id, hand_number, players, profiles, stacks, blind,
                  target=None, family='none', intensity=.5, active_episode=True, button_seat=5, public_styles=None):
    rng=random.Random(seed)
    # The engine's own internal shuffle is irrelevant: every dealt/burnt card is explicit.
    state=NoLimitTexasHoldem.create_state(AUTOMATIONS,True,0,(blind//2,blind),blind,tuple(stacks),6)
    deck=DECK.copy(); rng.shuffle(deck)
    while state.can_deal_hole(): state.deal_hole(deck.pop())
    holes=[tuple(repr(c) for c in cards) for cards in state.hole_cards]
    assert all(len(h)==2 for h in holes)
    hand_id=f'SS{pool_id:05d}H{hand_number:05d}'; table_id=f'SSPOOL{pool_id:05d}'
    actions=[]; intervention=[]; folded=[False]*6; raises=0; previous=None; last_street=0
    showdown_live=None
    while state.status:
        if state.can_burn_card(): state.burn_card(deck.pop()); continue
        if state.can_deal_board():
            count=state.board_dealing_count
            state.deal_board(''.join(deck.pop() for _ in range(count)))
            continue
        actor=state.actor_index
        if actor is None:
            raise RuntimeError('Unexpected unfinished engine state')
        street=state.street_index
        if street!=last_street: raises=0; previous=None; last_street=street
        board=tuple(repr(c) for c in state.get_board_cards(0))
        pot=state.total_pot_amount; stack=state.stacks[actor]; call=max(state.bets)-state.bets[actor]
        active=sum(not f for f in folded)
        obs=Observation(holes[actor],board,street,(actor-5)%6,pot,stack,call,blind,active,raises,previous,actor,len(actions),tuple((0 if a['action']=='fold' else 1 if a['action']=='check' else 2 if a['action']=='call' or (a['action']=='all_in' and a['amount']<=a['to_call']) else 3,a['amount'],STREETS.index(a['street'])) for a in actions if a['player_id']==players[actor]))
        from dataclasses import replace
        obs=replace(obs,public_history=tuple((players.index(a['player_id']),0 if a['action']=='fold' else 1 if a['action']=='check' else 2 if a['action']=='call' or (a['action']=='all_in' and a['amount']<=a['to_call']) else 3,STREETS.index(a['street']),a['amount']) for a in actions))
        obs=replace(obs,own_past_style=tuple((public_styles or {}).get(players[actor],(.25,.25,.25,.25))))
        proposal,base_probabilities=base_decision(obs,profiles[actor],rng)
        obs=replace(obs,base_action_probabilities=base_probabilities,can_raise=state.can_complete_bet_or_raise_to())
        chosen,reason=proposal,None
        if active_episode and target and players[actor] in target and all(p in players for p in target):
            partner=players.index(target[1] if players[actor]==target[0] else target[0])
            controller_rng=random.Random(seed*1000003+len(actions)*9176+27183)
            chosen,reason=intervene(obs,proposal,holes[partner],not folded[partner],partner,family,intensity,controller_rng,players[actor]==target[0])
        before_bet=state.bets[actor]
        base_resolved=resolve_decision(state,proposal)
        chosen_resolved=resolve_decision(state,chosen)
        op=execute_decision(state,chosen_resolved)
        if hasattr(op,'amount'): paid=op.amount
        else: paid=0
        if op.__class__.__name__=='CompletionBettingOrRaisingTo': paid=op.amount-before_bet
        name='fold' if op.__class__.__name__=='Folding' else ('raise' if op.__class__.__name__=='CompletionBettingOrRaisingTo' else ('call' if paid else 'check'))
        if name=='raise' and call==0: name='bet'
        if paid==stack and paid>0: name='all_in'
        assert 0<=paid<=stack
        if name=='fold': folded[actor]=True
        if paid>call and name in ['raise','bet','all_in']: raises+=1; previous=actor
        row={'hand_id':hand_id,'action_no':len(actions),'street':STREETS[street],
             'player_id':players[actor],'action':name,'amount':int(paid),'amount_to':int(before_bet+paid),
             'pot_before':int(pot),'stack_before':int(stack),'to_call':int(call),'players_active':active}
        actions.append(row)
        if reason and chosen_resolved!=base_resolved:
            intervention.append({'action_no':row['action_no'],'reason':reason,'base_intent':proposal,'chosen_intent':chosen,'actual_action':name,'own_past_style':obs.own_past_style})
        assert len(actions)<160
        if sum(not f for f in folded)>1 and (street==3 or all(state.stacks[i]==0 for i in range(6) if not folded[i])):
            showdown_live=[not f for f in folded]
    assert sum(state.stacks)==sum(stacks) and all(s>=0 for s in state.stacks)
    payouts=[0]*6; contribution=[0]*6
    for op in state.operations:
        if isinstance(op,BetCollection):
            contribution=[a+b for a,b in zip(contribution,op.bets)]
        elif isinstance(op,ChipsPushing):
            payouts=[a+b for a,b in zip(payouts,op.amounts)]
    # Uncalled winning bets stay outside the pot and are not a contribution.
    final_pot=sum(payouts)
    assert final_pot==sum(contribution)
    assert all(payouts[i]-contribution[i]==state.stacks[i]-stacks[i] for i in range(6))
    # Match the organizer's gross-contribution convention: uncalled winning
    # wagers count in the displayed pot and are returned in the winner's payout.
    # Engine net winnings remain authoritative and unchanged.
    contribution=[sum(a['amount'] for a in actions if a['player_id']==players[i])+(blind//2 if i==0 else blind if i==1 else 0) for i in range(6)]
    payouts=[state.stacks[i]-stacks[i]+contribution[i] for i in range(6)]
    assert min(payouts)>=0
    final_pot=sum(contribution)
    assert sum(payouts)==final_pot
    board=tuple(repr(c) for c in state.get_board_cards(0))
    if len(board)==5 and sum(not f for f in folded)>1: showdown_live=[not f for f in folded]
    else: showdown_live=[False]*6
    visible=[c for hole in holes for c in hole]+list(board)
    assert len(visible)==len(set(visible)) and len(board) in [0,3,4,5]
    hand={'hand_id':hand_id,'table_id':table_id,'started_at':hand_number,'phase':'development','button_seat':button_seat,
          'small_blind':blind//2,'big_blind':blind,'board_cards':' '.join(board),'final_pot':final_pot,
          'players_dealt':6,'players_at_showdown':sum(showdown_live)}
    seats=[{'hand_id':hand_id,'player_id':players[i],'seat_no':(i+button_seat+1)%6,'starting_stack':stacks[i],
            'hole_card_1':holes[i][0],'hole_card_2':holes[i][1],'total_contribution':contribution[i],
            'net_chips':state.stacks[i]-stacks[i],'folded':folded[i],'went_to_showdown':showdown_live[i],
            'won_share':payouts[i]/final_pot if final_pot else 0.} for i in range(6)]
    return hand,seats,actions,intervention
