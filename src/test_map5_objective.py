"""Independent brute-force metric and grouped learner checks before research use."""
import itertools
import unittest
import numpy as np
from map5_objective import swap_weights,query_derivatives,map5_lambdarank


def brute_ap(y,k):
    hits=0; value=0.
    for rank,label in enumerate(y[:k],1):
        if label: hits+=1; value+=hits/rank
    return value/min(sum(y),k) if sum(y) else 0.


class ObjectiveTests(unittest.TestCase):
    def test_all_binary_rankings_swap_deltas(self):
        checked=0
        for size in range(2,11):
            for bits in itertools.product([0,1],repeat=size):
                y=np.array(bits); pos=np.flatnonzero(y); neg=np.flatnonzero(1-y)
                for k in range(1,min(size,5)+1):
                    delta=swap_weights(y,pos,neg,k); before=brute_ap(y,k)
                    for a,p in enumerate(pos):
                        for b,n in enumerate(neg):
                            swapped=y.copy(); swapped[p],swapped[n]=swapped[n],swapped[p]
                            self.assertAlmostEqual(delta[a,b],abs(brute_ap(swapped,k)-before),places=12)
                            checked+=1
        self.assertGreater(checked,100000)

    def test_direction_conservation_and_extremes(self):
        y=np.array([0,0,0,0,0,1,1,0])
        for scores in [np.zeros(8),np.arange(8.),np.arange(8.)*10000]:
            g,h=query_derivatives(y,scores)
            self.assertTrue(np.isfinite(g).all() and np.isfinite(h).all())
            self.assertTrue(np.all(g[y==1]<=0) and np.all(g[y==0]>=0) and np.all(h>=0))
            self.assertAlmostEqual(g.sum(),0.,places=12)
        g,_=query_derivatives(y,np.zeros(8)); self.assertGreater(np.linalg.norm(g),0)

    def test_query_boundaries_and_degenerate_queries(self):
        a=np.array([1,0,0,1,0,0]); b=np.array([0,1,0,0]); scores=np.arange(10.)
        g,h=map5_lambdarank(np.r_[a,b],scores,None,np.array([6,4]))
        ga,ha=query_derivatives(a,scores[:6]); gb,hb=query_derivatives(b,scores[6:])
        np.testing.assert_array_equal(g,np.r_[ga,gb]); np.testing.assert_array_equal(h,np.r_[ha,hb])
        for labels in [np.zeros(8),np.ones(8)]:
            x,z=query_derivatives(labels,np.zeros(8)); self.assertFalse(x.any() or z.any())
        with self.assertRaises(ValueError): map5_lambdarank(a,scores[:6],None,[5])

    def test_lightgbm_learns_separable_synthetic_queries(self):
        import lightgbm as lgb
        rng=np.random.default_rng(731); x=rng.random((320,2)); y=(x[:,0]>.75).astype(int)
        model=lgb.LGBMRanker(objective=map5_lambdarank,n_estimators=30,learning_rate=.1,num_leaves=7,min_child_samples=2,reg_lambda=.1,n_jobs=2,verbosity=-1,random_state=31)
        model.fit(x,y,group=np.full(16,20))
        score=model.predict(x)
        values=[brute_ap(y[i:i+20][np.argsort(-score[i:i+20])],5) for i in range(0,320,20)]
        self.assertGreater(np.mean(values),.99)


if __name__=='__main__': unittest.main(verbosity=2)
