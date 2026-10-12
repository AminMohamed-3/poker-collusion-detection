"""Check size sampler, disabled-path parity and absence of controller RNG effects."""
from pathlib import Path
import json,sys,importlib.util,hashlib
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'sized'))
import learned_policy as policy
import generator as gen
import soft_controller as controller
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    plan=json.loads((H/'study-plan.json').read_text());old=load('old_size_policy',Path(plan['batches'][0]['existing_control'])/'learned_policy.py')
    q=policy.quantiles();assert q['median_model_source_sha256']==hashlib.sha256((H/'sized/models.joblib').read_bytes()).hexdigest();assert q['inputs']==policy.bundle()['inputs']
    samples=np.array([policy.sample_log_size([1,2,3],u) for u in np.linspace(0,1,10001)]);np.testing.assert_allclose(np.quantile(samples,[.1,.5,.9]),[1,2,3],atol=1e-12);assert np.diff(samples).min()>=0
    players=[f'test{i}' for i in range(6)];profiles=[gen.Profile(i%4,.08,.0,.12,.03,1.,(.6,.07,.13,.2)) for i in range(6)];stacks=[200]*6;styles={p:(.6,.07,.13,.2) for p in players};saved=gen.base_decision;saved_c=controller.decide;changed=0
    for seed in range(1800031,1800063):
        try:
            gen.base_decision=lambda obs,profile,rng:old.decision_with_probabilities(obs,profile,rng)
            baseline=gen.simulate_hand(seed,0,0,players,profiles,stacks,2,public_styles=styles)
            gen.base_decision=saved;policy.USE_QUANTILE_SIZING=False
            disabled=gen.simulate_hand(seed,0,0,players,profiles,stacks,2,public_styles=styles);assert baseline==disabled
            policy.USE_QUANTILE_SIZING=True
            candidate=gen.simulate_hand(seed,0,0,players,profiles,stacks,2,public_styles=styles);changed+=candidate!=baseline
            zero=gen.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',0,True,public_styles=styles);assert candidate==zero
            controller.decide=lambda obs,proposal,partner,rng,variant,mode:(proposal,None)
            noop=gen.simulate_hand(seed,0,0,players,profiles,stacks,2,(players[0],players[1]),'soft_play',1,True,public_styles=styles);assert candidate==noop
        finally:gen.base_decision=saved;controller.decide=saved_c;policy.USE_QUANTILE_SIZING=True
    assert changed>0
    receipt={'before_generation':True,'disabled_path_identical_32_seeds':True,'enabled_noop_zero_activation_trace_parity':True,'sampling_quantiles_exact':True,'ordinary_histories_changed_intentionally':changed,'inherited_style_prefix_checks_sha256':hashlib.sha256((H/'inherited-causal-style-tests.json').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(H/'causal-style-tests.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
