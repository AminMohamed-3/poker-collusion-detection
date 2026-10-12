"""Repair class representation and test whether it changes actual gameplay."""
from pathlib import Path
import sys,importlib.util,json,hashlib,shutil,random
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent/'softplay_causal_style_20260910/b0_style'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    d=H/'candidate';d.mkdir(exist_ok=False);pp=json.loads((P/'pilot-plan.json').read_text())
    for n,h in pp['source_hashes'].items():assert sha(P/n)==h;shutil.copy2(P/n,d/n)
    s=(d/'soft_controller.py').read_text();old='q[invalid]=0;p[invalid]=0;q/=q.sum();p/=p.sum()';new=old+'\n    if to_call==0:\n        q[1]+=q[2];q[2]=0;p[1]+=p[2];p[2]=0'
    assert s.count(old)==1;s=s.replace(old,new);(d/'soft_controller.py').write_text(s,encoding='utf-8')
    sys.path.insert(0,str(d));import generator as gen;import soft_controller as fixed
    legacy=load('legacy_probability_controller',P/'soft_controller.py');rng=np.random.default_rng(88901);distributions=0
    for owed in [0,2]:
        for can_raise in [False,True]:
            for _ in range(100):
                b=rng.dirichlet(np.ones(4));target=rng.dirichlet(np.ones(4));q,p=fixed.target_distribution(b,target,owed,can_raise);lq,lp=legacy.target_distribution(b,target,owed,can_raise)
                assert p[0]==q[0]==p[2]==q[2]==0 if owed==0 else p[1]==q[1]==0
                np.testing.assert_allclose(p[3],lp[3],atol=1e-14)
                if owed>0:np.testing.assert_allclose(p,lp,atol=1e-14)
                extra=np.maximum(p-q,0);joint=np.diag(np.minimum(q,p))
                if extra.sum()>1e-14:joint+=np.outer(np.maximum(q-p,0),extra)/extra.sum()
                np.testing.assert_allclose(joint.sum(1),q,atol=1e-12);np.testing.assert_allclose(joint.sum(0),p,atol=1e-12);distributions+=1
    players=[f'test{i}' for i in range(6)];profiles=[gen.Profile(i%4,.08,.0,.12,.03,1.,(.6,.07,.13,.2)) for i in range(6)];styles={p:(.6,.07,.13,.2) for p in players};saved=fixed.decide;checked=0
    for seed in range(1900031,1900287):
        try:
            fixed.decide=legacy.decide;a=gen.simulate_hand(seed,0,0,players,profiles,[200]*6,2,(players[0],players[1]),'soft_play',1,True,public_styles=styles)
            fixed.decide=saved;b=gen.simulate_hand(seed,0,0,players,profiles,[200]*6,2,(players[0],players[1]),'soft_play',1,True,public_styles=styles)
            assert a==b;checked+=1
        finally:fixed.decide=saved
    out={'corrected_controller_sha256':sha(d/'soft_controller.py'),'legal_distribution_cases_verified':distributions,'complete_gameplay_equivalence_seeds':checked,'all_generated_records_and_traces_identical':True,'interpretation':'The abstract class bookkeeping was inconsistent, but it did not change resolved gameplay in this controller. With no extra aggression permitted, both free-call classes resolve to check, and retained raise probability is identical. Algebra and256 complete seeded histories agree. Keep the correction for future code; do not claim it repairs the observed realism gap or spend on a new population for this change alone.','no_detector_or_old_artifact_modified':True};(H/'projection-result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
