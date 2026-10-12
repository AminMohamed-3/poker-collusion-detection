"""Transport ordinary action draws to a learned soft policy, using own/public inputs."""
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
def target_distribution(base,soft,to_call,can_raise=True):
    q=np.array(base,dtype=float);p=np.array(soft,dtype=float)
    assert q.shape==p.shape==(4,) and np.isfinite(q).all() and np.isfinite(p).all()
    assert np.min(q)>=0 and np.min(p)>=0
    invalid=0 if to_call==0 else 1
    q[invalid]=0;p[invalid]=0;q/=q.sum();p/=p.sum()
    if not can_raise:
        passive=1 if to_call==0 else 2
        q[passive]+=q[3];q[3]=0;p[passive]+=p[3];p[3]=0
    # Soft-play transport never introduces an extra aggressive action. Preserve
    # the learned non-aggressive class proportions when clipping raise mass.
    if p[3]>q[3]:
        p[:3]*=(1-q[3])/p[:3].sum();p[3]=q[3]
    return q,p
def coupled_class(base_class,q,p,rng):
    # q[a] * min(1,p[a]/q[a]) = min(q[a],p[a]). Rejected mass
    # equals sum((p-q)+), so the resulting marginal is exactly p.
    if q[base_class]<=0:raise ValueError('Impossible ordinary action class')
    if rng.random()<min(1.,p[base_class]/q[base_class]):return base_class
    extra=np.maximum(p-q,0);assert extra.sum()>0
    return rng.choices(range(4),weights=extra.tolist())[0]
def decide(obs,proposal,partner,rng,variant,mode):
    action,size=proposal
    source=0 if action=='fold' else 3 if action=='raise' else 1 if obs.to_call==0 else 2
    if source==3 and not obs.can_raise:source=1 if obs.to_call==0 else 2
    q,p=target_distribution(obs.base_action_probabilities,bundle()['soft'].predict(features(obs,partner),num_threads=1)[0],obs.to_call,obs.can_raise)
    chosen=coupled_class(source,q,p,rng)
    if chosen==source:return proposal,None
    assert chosen!=3
    if chosen==0:return ('fold',size),'partner_distribution_fold'
    return ('call',size),'partner_distribution_passivity' if source==3 else 'partner_distribution_continuation'
