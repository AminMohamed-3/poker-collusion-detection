"""Add legality-aware class transport before generation, preserving design history."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());assert not any((H/a/'pilot').exists() for a in plan['generator_arms'])
    (H/'design-before-legality-review.json').write_bytes((H/'study-plan.json').read_bytes())
    for arm in plan['generator_arms']:
        d=H/arm;s=(d/'soft_controller.py').read_text()
        s=s.replace('def target_distribution(base,soft,to_call):','def target_distribution(base,soft,to_call,can_raise=True):')
        s=s.replace('q[invalid]=0;p[invalid]=0;q/=q.sum();p/=p.sum()',"q[invalid]=0;p[invalid]=0;q/=q.sum();p/=p.sum()\n    if not can_raise:\n        passive=1 if to_call==0 else 2\n        q[passive]+=q[3];q[3]=0;p[passive]+=p[3];p[3]=0")
        s=s.replace("q,p=target_distribution(obs.base_action_probabilities,bundle()['soft'].predict(features(obs,partner),num_threads=1)[0],obs.to_call)","if source==3 and not obs.can_raise:source=1 if obs.to_call==0 else 2\n    q,p=target_distribution(obs.base_action_probabilities,bundle()['soft'].predict(features(obs,partner),num_threads=1)[0],obs.to_call,obs.can_raise)")
        (d/'soft_controller.py').write_text(s,encoding='utf-8')
        s=(d/'generator.py').read_text().replace('    base_action_probabilities: tuple = ()','    base_action_probabilities: tuple = ()\n    can_raise: bool = True')
        s=s.replace('obs=replace(obs,base_action_probabilities=base_probabilities)','obs=replace(obs,base_action_probabilities=base_probabilities,can_raise=state.can_complete_bet_or_raise_to())')
        (d/'generator.py').write_text(s,encoding='utf-8')
        pp=json.loads((d/'pilot-plan.json').read_text());pp['source_hashes']={n:sha(d/n) for n in pp['source_hashes']};(d/'pilot-plan.json').write_text(json.dumps(pp,indent=2));plan['generator_plan_sha256'][arm]=sha(d/'pilot-plan.json')
    (H/'soft_controller.py').write_bytes((H/plan['generator_arms'][0]/'soft_controller.py').read_bytes())
    plan['legality_review']='Before generating: transport ordinary and soft probabilities after folding illegal raise mass into check/call; never add aggression. No result-dependent change.'
    (H/'study-plan.json').write_text(json.dumps(plan,indent=2));print('Pre-generation legality review frozen.')
if __name__=='__main__':main()
