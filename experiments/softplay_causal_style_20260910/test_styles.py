"""Past-only style reconstruction and unchanged ordinary-policy trace checks."""
from pathlib import Path
import sys,importlib.util,json,hashlib
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'b0_style'))
import generator as new
import soft_controller as controller
from causal_styles import attach_styles
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def canonical(result):
    h,s,a,t=result
    return h,s,a,[{k:v for k,v in r.items() if k!='own_past_style'} for r in t]
def main():
    h=pl.DataFrame({'hand_id':['h1','h2','h3'],'started_at':[1,2,3]});s=pl.DataFrame({'hand_id':['h1','h1','h2','h2','h3','h3'],'player_id':['a','b']*3});a=pl.DataFrame({'hand_id':['h1','h1','h2','h2','h3','h3'],'player_id':['a','b']*3,'action':['fold','raise','check','call','raise','fold'],'amount':[0,10,0,10,100,0],'to_call':[2,2,0,10,2,100]});lab=pl.DataFrame({'pair_id':['p'],'player_1':['a'],'player_2':['b']});f=pl.DataFrame({'pair_id':['p','p'],'hand_id':['h1','h2'],'direction':[0,0]})
    _,prefix=attach_styles(f,h.head(2),s.head(4),a.head(4),lab);_,full=attach_styles(f,h,s,a,lab)
    assert all(prefix[k]==full[k] for k in prefix);np.testing.assert_allclose(full['h1','a'],[.25]*4);np.testing.assert_allclose(full['h2','a'],[.4,.2,.2,.2]);np.testing.assert_allclose(full['h2','b'],[.2,.2,.2,.4])
    oldroot=H.parent/'softplay_coupling_20260910/b0_coupled';old=load('pre_style_generation',oldroot/'generator.py');oldc=load('pre_style_controller',oldroot/'soft_controller.py')
    players=[f'test{i}' for i in range(6)];profiles=[new.Profile(i%4,.08,.0,.12,.03,1.,(.6,.07,.13,.2)) for i in range(6)];stacks=[200]*6;styles={p:(.6,.07,.13,.2) for p in players};saved=controller.decide;changed=0
    for seed in range(1700031,1700063):
        a=old.simulate_hand(seed,0,0,players,profiles,stacks,2);b=new.simulate_hand(seed,0,0,players,profiles,stacks,2,public_styles=styles);assert a==b
        z=new.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',0,True,public_styles=styles);assert a==z
        try:
            controller.decide=lambda obs,proposal,partner,rng,variant,mode:(proposal,None)
            z=new.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True,public_styles=styles);assert a==z
            controller.decide=oldc.decide
            c=old.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True)
            d=new.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True,public_styles=styles);assert canonical(c)==canonical(d)
        finally:controller.decide=saved
        d=new.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True,public_styles=styles);changed+=bool(d[-1])
        for t in d[-1]:np.testing.assert_array_equal(t['own_past_style'],styles[players[0]])
    assert changed>0
    result={'before_generation':True,'earlier_style_values_invariant_to_future_actions':True,'uniform_first_hand_prior':True,'current_hand_excluded':True,'ordinary_legacy_noop_zero_activation_trace_parity':True,'independent_seeds':32,'new_controller_changed_hands':changed,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(H/'causal-style-tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
