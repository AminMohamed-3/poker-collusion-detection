"""Small analytic checks for averaging, paired uncertainty and strict gates."""
from pathlib import Path
import json
import numpy as np
from replication_statistics import mean_metrics,deltas,pool_interval,gate
from prepare_trial import select_balanced
def main():
    # A constant positive loss contrast must remain constant under whole-pool resampling.
    np.testing.assert_allclose(pool_interval([.2]*6,['a','a','b','c','c','c']),[.2,.2])
    # Repeating every observation/history cannot narrow this pooled-bootstrap interval.
    v=np.array([.1,-.1,.4,.3]);g=np.array(['a','a','b','c'])
    np.testing.assert_allclose(pool_interval(v,g),pool_interval(np.tile(v,3),np.tile(g,3)))
    a={'log_loss':.02,'pair_ap':.91,'sparse_ap':.7,'soft_play_ap':.8};b={**a,'log_loss':.01}
    assert abs(deltas(b,mean_metrics([a,b]))['logloss_improvement']-.005)<1e-12
    m={'logloss_improvement':.002,'pair_ap':.0001,'sparse_ap':.0001}
    assert gate(m,0,[.001,.003],[.001,.003],[.001,.002,.003])['passed']
    assert not gate(m,0,[.001,.003],[-.001,.005],[.001,.002,.003])['passed']
    assert not gate(m,0,[.001,.003],[.001,.003],[.001,-.0001,.006])['passed']
    assert not gate({**m,'sparse_ap':-.00001},0,[.001,.003],[.001,.003],[.001,.002,.003])['passed']
    ids=[str(i) for i in range(20)];families={p:int(p)%4 for p in ids};s=select_balanced(ids,families,13)
    assert len(s)==len(set(s))==13 and len({families[p] for p in s})==4
    assert s==select_balanced(list(reversed(ids)),families,13)
    out={'analytic_statistics_checks':9,'passed':True,'held_data_used':False}
    (Path(__file__).resolve().parent/'statistics-tests.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
