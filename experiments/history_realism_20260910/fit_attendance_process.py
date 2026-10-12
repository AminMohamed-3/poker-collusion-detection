"""Fit a small mixture of seat persistence and complete table refresh."""
from pathlib import Path
import hashlib,json,math
from datetime import datetime,timezone
import numpy as np
from scipy.optimize import minimize
H=Path(__file__).resolve().parent
d=json.loads((H/'attendance-diagnosis.json').read_text())
counts=np.array([r['len'] for r in d['turnover_counts']],dtype=float)
assert [r['n_changed'] for r in d['turnover_counts']]==list(range(7))

def perm(n,k):return math.factorial(n)//math.factorial(n-k)

# A fresh ordered uniform six-player table from30 can coincidentally retain
# some prior occupants in the same seat. Count these cases exactly.
refresh=np.zeros(7)
for matches in range(7):
    configurations=math.comb(6,matches)*sum((-1)**j*math.comb(6-matches,j)*perm(30-matches-j,6-matches-j) for j in range(7-matches))
    refresh[6-matches]=configurations/perm(30,6)
assert abs(refresh.sum()-1)<1e-12 and min(refresh)>=0

def probs(params):
    p,q=params;single=np.array([math.comb(6,k)*p**k*(1-p)**(6-k) for k in range(7)])
    return (1-q)*single+q*refresh

fit=minimize(lambda z:-float(counts@np.log(probs(z))),[.035,.015],bounds=[(.0001,.2),(.0001,.1)],method='L-BFGS-B')
assert fit.success
p,q=fit.x
r={'created_at':datetime.now(timezone.utc).isoformat(),'seat_exit_probability':float(p),'table_refresh_probability':float(q),'observed_counts':counts.astype(int).tolist(),'predicted_counts':(probs(fit.x)*counts.sum()).tolist(),'absolute_probability_error':np.abs(counts/counts.sum()-probs(fit.x)).tolist(),'total_variation':float(.5*np.abs(counts/counts.sum()-probs(fit.x)).sum()),'button':'Independent uniform physical seat, matching observed training button transition frequencies, not assumed real-world dealer rotation.','seating':'Uniform initial and refresh seating. On ordinary transitions each seat independently exits; refill from the24previously unseated players without replacement. Departing players cannot immediately reenter in the same transition.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_original_seat_schedule_or_game_outcomes_copied':True,'interpretation':'Fitted stochastic approximation; not recovered original source code.'}
(H/'attendance-process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
