"""Small exact enumeration/derivative checks; no poker data or model fitting."""
import sys,json,hashlib,time,itertools
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
from subset_likelihood import subset_nll

def enumeration(y,s):
    subsets=list(itertools.combinations(range(len(y)),int(sum(y))))
    membership=np.array([[i in a for i in range(len(y))] for a in subsets],dtype=float)
    energy=membership@s;z=np.logaddexp.reduce(energy)
    weights=np.exp(energy-z);mu=weights@membership
    return float(z-y@s),mu,((membership-mu).T*weights)@(membership-mu)

def main():
    start=time.perf_counter();rng=np.random.default_rng(92612)
    counts={'enumerated_queries':0,'derivative_queries':0,'offset_and_permutation_queries':0,'softmax_equivalence':0,'extreme_score_queries':0,'degenerate_queries':0}
    max_g=max_h=0.
    for n in range(2,9):
        for k in range(1,n):
            s=rng.normal(size=n)*2;y=np.zeros(n);y[rng.choice(n,k,replace=False)]=1
            loss,g,h,mu=subset_nll(y,s);el,em,eh=enumeration(y,s)
            np.testing.assert_allclose(loss,el,rtol=0,atol=1e-11)
            np.testing.assert_allclose(mu,em,rtol=0,atol=1e-11)
            np.testing.assert_allclose(h,np.diag(eh),rtol=0,atol=1e-11)
            np.testing.assert_allclose(mu.sum(),k,rtol=0,atol=1e-10)
            np.testing.assert_allclose(g.sum(),0,rtol=0,atol=1e-10)
            counts['enumerated_queries']+=1
            step=1e-5;fg=np.empty(n);fh=np.empty(n)
            for i in range(n):
                plus=s.copy();minus=s.copy();plus[i]+=step;minus[i]-=step
                pl,pg,_,_=subset_nll(y,plus);ml,mg,_,_=subset_nll(y,minus)
                fg[i]=(pl-ml)/(2*step);fh[i]=(pg[i]-mg[i])/(2*step)
            max_g=max(max_g,float(np.max(np.abs(fg-g))));max_h=max(max_h,float(np.max(np.abs(fh-h))))
            np.testing.assert_allclose(fg,g,rtol=0,atol=2e-8);np.testing.assert_allclose(fh,h,rtol=0,atol=2e-8)
            counts['derivative_queries']+=1
            order=rng.permutation(n);l2,g2,h2,m2=subset_nll(y[order],s[order]+100)
            np.testing.assert_allclose([loss], [l2],rtol=0,atol=1e-10)
            for a,b in ((g,g2),(h,h2),(mu,m2)):np.testing.assert_allclose(a[order],b,rtol=0,atol=1e-10)
            counts['offset_and_permutation_queries']+=1
            if k==1:
                lognorm=np.logaddexp.reduce(s);np.testing.assert_allclose(loss,lognorm-y@s,atol=1e-11)
                np.testing.assert_allclose(mu,np.exp(s-lognorm),atol=1e-11);counts['softmax_equivalence']+=1
    for y in (np.zeros(5),np.ones(5)):
        loss,g,h,m=subset_nll(y,rng.normal(size=5));assert loss==0 and np.all(g==0) and np.all(h==0) and np.array_equal(m,y)
        counts['degenerate_queries']+=1
    for k in (1,2,3):
        y=np.zeros(4);y[:k]=1;s=np.array([-1000.,-300.,300.,1000.])
        loss,g,h,m=subset_nll(y,s)
        assert np.isfinite(loss) and np.isfinite(g).all() and np.isfinite(h).all() and np.isfinite(m).all()
        np.testing.assert_allclose(m.sum(),k,atol=1e-10);counts['extreme_score_queries']+=1
    out=HERE/'research/subset_likelihood_20260912';out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),HERE/'src/subset_likelihood.py']
    receipt={'status':'mechanics_passed_no_model_fits','checks':counts,'max_finite_difference_gradient_error':max_g,'max_finite_difference_hessian_error':max_h,'seconds':time.perf_counter()-start,'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'poker_labels_read':False,'model_fits':0}
    (out/'mechanical-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
