"""Observed-card multiway showdown-share descriptors, without future board use.

All disclosed seat cards block runouts. Active seats share a single hypothetical
pot with no further betting. This is not player knowledge or side-pot utility.
"""
from functools import lru_cache
from itertools import combinations
import math
import numpy as np
from phevaluator import evaluate_cards

SAMPLES=128
SEED=472711
NAMES=['partner_call_deficit_bb','partner_call_surplus_bb','partner_fold_surplus_bb',
       'partner_fold_own_share','partner_fold_partner_gain','partner_fold_outsider_gain',
       'partner_call_outsider_share_cost','late_check_own_share','multiway_check_joint_share',
       'multiway_raise_weak_joint_cost','multiway_raise_partner_advantage_cost','multiway_reraise_joint_share_cost']
FEATURES=['mw_'+name+suffix for suffix in ['_sum','_max'] for name in NAMES]


@lru_cache(maxsize=64)
def sample_indices(count,draws):
    return np.argsort(np.random.default_rng(SEED).random((SAMPLES,count)),axis=1)[:,:draws]


class HandShares:
    def __init__(self,seats,board):
        if any(len(c)!=2 for c in seats.values()): raise ValueError('Two cards per seat required')
        self.players=sorted(seats,key=lambda p:tuple(sorted(seats[p])))
        self.holes=[tuple(sorted(seats[p])) for p in self.players]
        self.index={p:i for i,p in enumerate(self.players)}
        self.board=tuple(board); self.ranks={}; self.cache={}
        known=[c for h in self.holes for c in h]
        if len(set(known))!=len(known): raise ValueError('Duplicated hole card')

    def street_ranks(self,street):
        if street not in self.ranks:
            n={'preflop':0,'flop':3,'turn':4,'river':5}[street]
            if len(self.board)<n: raise ValueError('Street exceeds observed board')
            board=tuple(sorted(self.board[:n])); known=set(board)
            for h in self.holes:
                if known.intersection(h): raise ValueError('Duplicated card')
                known.update(h)
            if len(set(board))!=n: raise ValueError('Duplicated board card')
            remaining=tuple(c for c in range(52) if c not in known); draws=5-n
            future=(tuple(remaining[i] for i in row) for row in sample_indices(len(remaining),draws)) if n<4 else combinations(remaining,draws)
            self.ranks[street]=np.array([[evaluate_cards(*h,*board,*runout) for h in self.holes] for runout in future],dtype=np.int32)
        return self.ranks[street]

    def shares(self,live,street):
        indices=tuple(sorted(self.index[p] for p in live)); key=(street,indices)
        if not indices: raise ValueError('No active players')
        if key not in self.cache:
            ranks=self.street_ranks(street)[:,indices]
            wins=ranks==ranks.min(axis=1,keepdims=True)
            share=(wins/wins.sum(axis=1,keepdims=True)).mean(axis=0)
            self.cache[key]={self.players[i]:float(q) for i,q in zip(indices,share)}
        return self.cache[key]


def extract_pair_hand(first,second,big_blind,seats,board,actions,engine=None):
    pair={first,second}
    if len(pair)!=2 or not pair.issubset(seats) or big_blind<=0: raise ValueError('Invalid pair/blind')
    engine=engine or HandShares(seats,board)
    live=set(seats); values={p:dict.fromkeys(NAMES,0.) for p in pair}; previous=None; street=None
    ordered=sorted(actions,key=lambda a:a['action_no'])
    if len({a['action_no'] for a in ordered})!=len(ordered): raise ValueError('Duplicate action number')
    for a in ordered:
        if a['street']!=street: street=a['street']; previous=None
        actor=a['player_id']; kind=a['action']
        if actor not in live: raise ValueError('Action after fold')
        aggressive=kind in ('bet','raise','all_in') and a['amount']>a['to_call']
        calling=kind in ('call','all_in') and not aggressive
        if actor in pair and pair.issubset(live):
            partner=second if actor==first else first; to_partner=previous==partner
            if kind=='check' or aggressive or (to_partner and (calling or kind=='fold')):
                share=engine.shares(live,street); own=share[actor]; other=share[partner]; joint=own+other
                outsider=max(0.,1-joint); cost=min(a['to_call'],a['stack_before']); pot=a['pot_before']; v=values[actor]
                surplus=max(0.,own*(pot+cost)-cost)/big_blind
                deficit=max(0.,cost-own*(pot+cost))/big_blind
                if to_partner and calling:
                    v['partner_call_deficit_bb']+=deficit; v['partner_call_surplus_bb']+=surplus
                    v['partner_call_outsider_share_cost']+=outsider*a['amount']/big_blind
                if to_partner and kind=='fold':
                    after=engine.shares(live-{actor},street)
                    v['partner_fold_surplus_bb']+=surplus; v['partner_fold_own_share']+=own
                    v['partner_fold_partner_gain']+=max(0.,after[partner]-other)
                    v['partner_fold_outsider_gain']+=max(0.,1-after[partner]-outsider)
                if kind=='check':
                    v['late_check_own_share']+=own*(street in ('turn','river'))
                    v['multiway_check_joint_share']+=joint*(len(live)>=3)
                if aggressive and len(live)>=3:
                    weight=math.log1p(a['amount']/big_blind)
                    v['multiway_raise_weak_joint_cost']+=outsider*weight
                    v['multiway_raise_partner_advantage_cost']+=max(0.,other-own)*weight
                    v['multiway_reraise_joint_share_cost']+=joint*weight*to_partner
        if aggressive: previous=actor
        if kind=='fold': live.remove(actor)
    first_values,second_values=values[first],values[second]
    result=[first_values[n]+second_values[n] for n in NAMES]+[max(first_values[n],second_values[n]) for n in NAMES]
    if not all(math.isfinite(x) and x>=0 for x in result): raise ValueError('Invalid multiway feature')
    return result
