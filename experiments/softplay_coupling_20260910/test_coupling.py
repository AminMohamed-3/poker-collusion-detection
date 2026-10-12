"""Algebraic and simulation checks for a causal change in policy distribution."""
from pathlib import Path
import sys,importlib.util,json,random,hashlib
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'b0_coupled'))
import generator as new
import soft_controller as controller
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def main():
    oldroot=H.parent/'softplay_replication_20260910/b0_continuous';old=load('old_generation',oldroot/'generator.py');oldc=load('old_controller',oldroot/'soft_controller.py')
    rng=np.random.default_rng(55009);checked=0
    for owed in [0,4]:
        for can_raise in [False,True]:
            for _ in range(100):
                q,p=controller.target_distribution(rng.dirichlet(np.ones(4)),rng.dirichlet(np.ones(4)),owed,can_raise)
                extra=np.maximum(p-q,0);total=extra.sum();joint=np.diag(np.minimum(q,p))
                if total>1e-14:joint+=np.outer(np.maximum(q-p,0),extra)/total
                np.testing.assert_allclose(joint.sum(1),q,atol=1e-12);np.testing.assert_allclose(joint.sum(0),p,atol=1e-12)
                assert p[3]<=q[3]+1e-12 and p[0 if owed==0 else 1]==0
                if not can_raise:assert q[3]==p[3]==0
                checked+=1
    q,p=controller.target_distribution([.6,0,.15,.25],[.5,0,.4,.1],4);rr=random.Random(67391);counts=np.zeros(4)
    for _ in range(50000):
        a=rr.choices(range(4),weights=q.tolist())[0];counts[controller.coupled_class(a,q,p,rr)]+=1
    np.testing.assert_allclose(counts/counts.sum(),p,atol=.008)
    players=[f'test{i}' for i in range(6)];profiles=[new.Profile(i%4,.08,.0,.12,.03,1.,(.6,.07,.13,.2)) for i in range(6)];stacks=[200]*6;saved=controller.decide;changed=0
    for seed in range(1500031,1500063):
        a=old.simulate_hand(seed,0,0,players,profiles,stacks,2)
        b=new.simulate_hand(seed,0,0,players,profiles,stacks,2);assert a==b
        z=new.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',0,True);assert a==z
        try:
            controller.decide=lambda obs,proposal,partner,rng,variant,mode:(proposal,None)
            z=new.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True);assert a==z
            controller.decide=oldc.decide
            c=old.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True)
            d=new.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True);assert c==d
        finally:controller.decide=saved
        z=new.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True);changed+=bool(z[-1])
    assert changed>0
    result={'algebraic_distributions_verified':checked,'empirical_sampling_draws':50000,'empirical_class_probabilities':(counts/50000).tolist(),'expected_class_probabilities':p.tolist(),'independent_seeds_checked':32,'ordinary_trace_exact':True,'legacy_controller_trace_exact':True,'zero_activation_trace_exact':True,'noop_trace_exact':True,'changed_hands_with_new_controller':changed,'no_new_aggression':True,'before_generation':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (H/'coupling-tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
