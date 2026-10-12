"""Only own cards, public actions, and an assigned ordinary-player style enter here."""
from collections import Counter
from pathlib import Path
import math
import joblib
import numpy as np
from conditional_action_features import card_features
from advanced_features import CARDS

HERE=Path(__file__).resolve().parent
BUNDLE=None
def bundle():
    global BUNDLE
    if BUNDLE is None:BUNDLE=joblib.load(HERE/'models.joblib')
    return BUNDLE

def sample_style(rng):return tuple(bundle()['styles'][rng.randrange(len(bundle()['styles']))])

def features(obs,profile):
    cards=tuple(sorted(CARDS[c] for c in obs.hole));board=tuple(CARDS[c] for c in obs.board)
    own=card_features(cards,board);ranks=Counter(c//4 for c in board);suits=Counter(c%4 for c in board)
    hist=obs.own_history
    values=[obs.street,*own,max((c//4+2 for c in board),default=0),sum(n-1 for n in ranks.values()),max(suits.values(),default=0),obs.position,obs.to_call/max(1,obs.pot+obs.to_call),math.log1p(obs.pot/obs.blind),math.log1p(obs.stack/obs.blind),obs.active,obs.prior_actions,len(hist),sum(a[0]==3 for a in hist),sum(a[0]==3 and a[2]==obs.street for a in hist),math.log1p(sum(a[1] for a in hist)/obs.blind),*profile.style]
    assert len(values)==len(bundle()['inputs'])
    return np.array([values],dtype=np.float32)

def decision(obs,profile,rng):
    x=features(obs,profile);model=bundle()['models'][profile.family]
    p=model['classifier'].predict(x,num_threads=1)[0]
    # Fixed moderate temperature variety; never use a target label or partner card.
    p=np.maximum(p,1e-8)**(1/[.85,1.,1.15,1.][profile.family])
    # Public ordinary-player variation, including the runner's episodic tilt.
    p[0]*=math.exp(1.5*(profile.tightness-.08))
    p[3]*=math.exp(2*profile.aggression)
    p=p**(1/max(.8,1+(profile.noise-.12)))
    p[0 if obs.to_call==0 else 1]=0
    p/=p.sum();kind=rng.choices(range(4),p.tolist())[0]
    if kind!=3:return ('fold' if kind==0 else 'call'),.75
    log_amount=float(model['regressor'].predict(x,num_threads=1)[0])
    amount=max(obs.blind,math.expm1(log_amount)*obs.blind)*rng.uniform(.8,1.2)
    size=max(.1,min(3.,(amount-obs.to_call)/max(1,obs.pot+obs.to_call)))
    return 'raise',size
