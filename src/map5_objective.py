"""Binary LambdaRank pseudo-gradients weighted by exact AP@5 swap changes.

The ranking metric is discrete: these are metric-weighted RankNet derivatives,
not derivatives of AP itself. No cross-query pairs or learned ID inputs.
"""
import numpy as np


def swap_weights(sorted_labels, positive, negative, cutoff=5):
    """Absolute AP@cutoff change for each positive/negative position swap."""
    y=np.asarray(sorted_labels,dtype=np.float64)
    p=np.asarray(positive,dtype=np.int64)[:,None]
    n=np.asarray(negative,dtype=np.int64)[None,:]
    if p.size==0 or n.size==0: return np.zeros((p.shape[0],n.shape[1]))
    a=np.minimum(p,n); b=np.maximum(p,n)
    cumulative=np.cumsum(y)
    term=y/np.arange(1,len(y)+1); term[cutoff:]=0
    prefix=np.r_[0.,np.cumsum(term)]
    first=np.where(a<cutoff,(cumulative[a]+1-y[a])/(a+1),0.)
    last=np.where(b<cutoff,cumulative[b]/(b+1),0.)
    between=prefix[b]-prefix[a+1]
    result=(first-last+between)/min(cutoff,int(y.sum()))
    if result.min(initial=0)<-1e-12: raise ValueError('Negative absolute swap weight')
    return np.maximum(result,0.)


def query_derivatives(labels,scores,cutoff=5):
    y=np.asarray(labels); s=np.asarray(scores,dtype=np.float64)
    order=np.argsort(-s,kind='stable'); y=y[order]; s=s[order]
    positive=np.flatnonzero(y==1); negative=np.flatnonzero(y==0)
    grad=np.zeros(len(y)); hess=np.zeros(len(y))
    if len(positive) and len(negative):
        delta=swap_weights(y,positive,negative,cutoff)
        rho=1/(1+np.exp(np.clip(s[positive,None]-s[None,negative],-50,50)))
        pairgrad=delta*rho; pairhess=pairgrad*(1-rho)
        grad[positive]=-pairgrad.sum(axis=1); grad[negative]=pairgrad.sum(axis=0)
        hess[positive]=pairhess.sum(axis=1); hess[negative]=pairhess.sum(axis=0)
        # Frozen soft query normalization; avoids linear domination by long histories.
        magnitude=2*pairgrad.sum()
        if magnitude>0:
            norm=np.log2(1+magnitude)/magnitude
            grad*=norm; hess*=norm
    restored_grad=np.empty(len(y)); restored_hess=np.empty(len(y))
    restored_grad[order]=grad; restored_hess[order]=hess
    return restored_grad,restored_hess


def map5_lambdarank(y_true,y_pred,weight,group):
    y=np.asarray(y_true); scores=np.asarray(y_pred)
    if group is None: raise ValueError('Explicit query boundaries required')
    groups=np.asarray(group)
    if np.any(groups<1) or np.any(groups!=groups.astype(np.int64)) or int(groups.sum())!=len(y):
        raise ValueError('Invalid query boundaries')
    if y.ndim!=1 or scores.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(scores).all():
        raise ValueError('Expected binary relevance and finite scalar scores')
    if weight is not None and not np.all(np.asarray(weight)==1):
        raise ValueError('This frozen experiment uses equal query treatment and no sample weights')
    grad=np.empty(len(y)); hess=np.empty(len(y)); start=0
    for count in groups:
        end=start+int(count)
        grad[start:end],hess[start:end]=query_derivatives(y[start:end],scores[start:end])
        start=end
    return grad,hess
