"""Exact fixed-cardinality conditional-Bernoulli subset NLL for one query.

Targets are observed evidence-list membership, not genuine manipulation labels.
No project data is imported. The diagonal Hessian is exact; a tree learner using
only that diagonal would still approximate the full, coupled query Hessian.
"""
import numpy as np


def subset_nll(labels, scores):
    y=np.asarray(labels,dtype=float);s=np.asarray(scores,dtype=float)
    if y.ndim!=1 or s.shape!=y.shape or len(y)==0 or not np.isin(y,[0,1]).all() or not np.isfinite(s).all():
        raise ValueError('Expected nonempty binary membership and matching finite scores')
    n=len(y);k=int(y.sum())
    if k in (0,n):
        return 0.0,np.zeros(n),np.zeros(n),y.copy()
    # A common offset cancels in a fixed-size subset distribution. Centering
    # before the log recursion also avoids cancellation in the returned loss.
    z=s-s.max()
    forward=np.full((n+1,k+1),-np.inf);backward=np.full_like(forward,-np.inf)
    forward[:,0]=0.;backward[:,0]=0.
    for i in range(n):
        forward[i+1,1:]=np.logaddexp(forward[i,1:],z[i]+forward[i,:-1])
    for i in range(n-1,-1,-1):
        backward[i,1:]=np.logaddexp(backward[i+1,1:],z[i]+backward[i+1,:-1])
    log_partition=forward[n,k]
    mu=np.empty(n)
    for i in range(n):
        log_excluding=np.logaddexp.reduce(forward[i,:k]+backward[i+1,k-1::-1])
        mu[i]=np.exp(z[i]+log_excluding-log_partition)
    assert np.isfinite(mu).all() and np.min(mu)>=0 and np.max(mu)<=1+1e-10
    mu=np.clip(mu,0,1)
    loss=float(log_partition-np.dot(y,z))
    gradient=mu-y
    hessian_diagonal=mu*(1-mu)
    return loss,gradient,hessian_diagonal,mu
