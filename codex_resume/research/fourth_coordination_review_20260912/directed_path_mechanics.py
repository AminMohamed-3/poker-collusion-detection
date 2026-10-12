"""Synthetic mechanics only; no poker observations, family labels, or model fitting."""
import json
from pathlib import Path
import numpy as np

def chains(flow):
    assert flow.ndim==2 and flow.shape[0]==flow.shape[1]
    assert np.allclose(flow,-flow.T) and np.allclose(np.diag(flow),0)
    positive=np.maximum(flow,0)
    n=len(flow);out=np.zeros((n,n))
    for c in range(n):
        np.maximum(out,np.minimum(positive[c,:][:,None],positive[:,c][None,:]),out=out)
    np.fill_diagonal(out,0)
    return {'directed':out,'old_both_directions':np.minimum(out,out.T),'either_direction':np.maximum(out,out.T)}

def make_flow(n,edges):
    flow=np.zeros((n,n))
    for payer,receiver,amount in edges:
        assert payer!=receiver and amount>0
        flow[receiver,payer]+=amount;flow[payer,receiver]-=amount
    return flow

def run():
    results=[]
    for strength in [1.,2.,4.]:
        x=chains(make_flow(4,[(0,1,4*strength),(1,2,3*strength)]))
        assert x['directed'][0,2]==3*strength and x['directed'][2,0]==0
        assert x['old_both_directions'][0,2]==0 and x['either_direction'][0,2]==3*strength
        results.append({'strength':strength,'old':float(x['old_both_directions'][0,2]),'corrected':float(x['either_direction'][0,2])})
    for edges in [[],[(0,2,4)],[(0,1,4),(3,2,3)],[(0,1,4),(2,1,3)]]:
        assert chains(make_flow(4,edges))['either_direction'][0,2]==0
    f=make_flow(4,[(0,1,4),(1,2,3)]);swap=[2,1,0,3]
    original=chains(f)['either_direction'];permuted=chains(f[np.ix_(swap,swap)])['either_direction']
    assert np.array_equal(permuted,original[np.ix_(swap,swap)])
    assert np.array_equal(chains(-f)['either_direction'],original)
    return {'one_way_strengths':results,'zero_direct_only_disconnected_shared_receiver_controls_pass':True,'player_relabel_equivariance':True,'global_direction_reversal_invariant':True,'scope':'Arithmetic counterexample only; matrices are not legal hand-history simulations. No real-data power, fourth-family identity, or public gain established.'}
if __name__=='__main__':
    result=run();p=Path(__file__).with_name('mechanics-checks.json')
    with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
