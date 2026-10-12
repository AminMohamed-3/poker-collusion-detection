"""Own-card/public-history soft-play controller; no partner private cards."""
from pathlib import Path
import math
import joblib
import numpy as np
from conditional_action_features import card_features
from advanced_features import CARDS
BUNDLE=None
def bundle():
    global BUNDLE
    if BUNDLE is None:BUNDLE=joblib.load(Path(__file__).resolve().parent/'controller.joblib')
    return BUNDLE
def features(obs,partner):
    own=card_features(tuple(sorted(CARDS[c] for c in obs.hole)),tuple(CARDS[c] for c in obs.board))
    a=[r for r in obs.public_history if r[0]==obs.actor];b=[r for r in obs.public_history if r[0]==partner]
    values=[obs.street,own[0],own[3],own[4],own[1],own[2],obs.position,obs.to_call/max(1,obs.pot+obs.to_call),math.log1p(obs.pot/obs.blind),math.log1p(obs.stack/obs.blind),obs.active,obs.prior_actions,len(a),sum(r[1]==3 for r in a),sum(r[1]==3 and r[2]==obs.street for r in a),len(b),sum(r[1]==3 for r in b),sum(r[1]==3 and r[2]==obs.street for r in b),b[-1][1] if b else -1,sum(r[1]==3 for r in obs.public_history if r[0] not in [obs.actor,partner]),int(obs.previous_aggressor==partner)]
    assert len(values)==len(bundle()['inputs'])
    return np.array([values],dtype=np.float32)
def decide(obs,proposal,partner,rng,variant,mode):
    action,size=proposal;facing=obs.previous_aggressor==partner;price=obs.to_call/max(1,obs.pot+obs.to_call)
    if mode=='learned':
        x=features(obs,partner);p=bundle()['soft'].predict(x,num_threads=1)[0];normal=bundle()['normal'].predict(x,num_threads=1)[0]
        passive=1 if obs.to_call==0 else 2
        p[0 if passive==1 else 1]=0;p/=p.sum()
        choice=rng.choices(range(4),p.tolist())[0]
        if choice!=passive or p[passive]<=normal[passive]:return proposal,None
        if action=='raise':return ('call',size),'partner_conditional_passivity'
        if action=='fold' and price<=.5 and (obs.street==0 or facing):return ('call',size),'partner_conditional_continuation'
        return proposal,None
    cap=(.25,.40,.40,.25)[variant]
    own=card_features(tuple(sorted(CARDS[c] for c in obs.hole)),tuple(CARDS[c] for c in obs.board))[0]
    allow_fold=mode=='baseline' or own>=bundle()['minimum_call_strength'][str(obs.street)]
    if obs.street==0 and price<=cap and (variant!=3 or facing) and (action!='fold' or allow_fold):return ('call',size),'partner_low_pressure_preflop'
    if obs.active==2 and obs.to_call==0:return ('call',size),'partner_check'
    if facing and obs.to_call>0:
        if action=='raise':return ('call',size),'partner_raise_suppressed'
        if price<=cap and action=='fold' and allow_fold:return ('call',size),'partner_low_pressure_call'
    if obs.street>0 and obs.to_call==0 and variant in (0,1):return ('call',size),'partner_multiway_check'
    return proposal,None
