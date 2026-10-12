"""Causal no-op invariance before generation; no learned detector tests."""
from pathlib import Path
import sys,json,hashlib,random
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'calibrated'))
import generator as g
import soft_controller as c
def main():
    players=[f'test{i}' for i in range(6)];profiles=[g.Profile(i%4,.08,.0,.12,.03,1.,(.6,.07,.13,.2)) for i in range(6)];stacks=[200]*6;checks=0;changed=0
    saved=c.decide
    for seed in range(900031,900063):
        a=g.simulate_hand(seed,0,0,players,profiles,stacks,2)
        b=g.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',0,True)
        assert a==b
        try:
            c.decide=lambda obs,proposal,partner,rng,variant,mode:(proposal,None)
            z=g.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True)
            assert a==z
        finally:c.decide=saved
        d=g.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True)
        changed+=bool(d[-1]);checks+=1
    assert changed>0
    out={'seeds_checked':checks,'zero_activation_history_exact':True,'active_noop_history_exact':True,'real_controller_changed_hands':changed,'generator_sha256':hashlib.sha256((H/'calibrated/generator.py').read_bytes()).hexdigest()};(H/'randomness-tests.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
