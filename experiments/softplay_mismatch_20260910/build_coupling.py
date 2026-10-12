"""Build a causal distribution-coupled soft-policy experiment from frozen models."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).resolve().parent;P=H.parent/'softplay_replication_20260910';N=H.parent/'softplay_coupling_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
CONTROLLER='''"""Transport ordinary action draws to a learned soft policy, using own/public inputs."""
from pathlib import Path
import math
import joblib
import numpy as np
from conditional_action_features import card_features
from advanced_features import CARDS
BUNDLE=None
def bundle():
    global BUNDLE
    if BUNDLE is None:BUNDLE=joblib.load(Path(__file__).resolve().parent/'controller.joblib')
    return BUNDLE
def features(obs,partner):
    own=card_features(tuple(sorted(CARDS[c] for c in obs.hole)),tuple(CARDS[c] for c in obs.board))
    a=[r for r in obs.public_history if r[0]==obs.actor];b=[r for r in obs.public_history if r[0]==partner]
    values=[obs.street,own[0],own[3],own[4],own[1],own[2],obs.position,obs.to_call/max(1,obs.pot+obs.to_call),math.log1p(obs.pot/obs.blind),math.log1p(obs.stack/obs.blind),obs.active,obs.prior_actions,len(a),sum(r[1]==3 for r in a),sum(r[1]==3 and r[2]==obs.street for r in a),len(b),sum(r[1]==3 for r in b),sum(r[1]==3 and r[2]==obs.street for r in b),b[-1][1] if b else -1,sum(r[1]==3 for r in obs.public_history if r[0] not in [obs.actor,partner]),int(obs.previous_aggressor==partner)]
    assert len(values)==len(bundle()['inputs'])
    return np.array([values],dtype=np.float32)
def target_distribution(base,soft,to_call):
    q=np.array(base,dtype=float);p=np.array(soft,dtype=float)
    assert q.shape==p.shape==(4,) and np.isfinite(q).all() and np.isfinite(p).all()
    assert np.min(q)>=0 and np.min(p)>=0
    invalid=0 if to_call==0 else 1
    q[invalid]=0;p[invalid]=0;q/=q.sum();p/=p.sum()
    # Soft-play transport never introduces an extra aggressive action. Preserve
    # the learned non-aggressive class proportions when clipping raise mass.
    if p[3]>q[3]:
        p[:3]*=(1-q[3])/p[:3].sum();p[3]=q[3]
    return q,p
def coupled_class(base_class,q,p,rng):
    # q[a] * min(1,p[a]/q[a]) = min(q[a],p[a]). Rejected mass
    # equals sum((p-q)+), so the resulting marginal is exactly p.
    if q[base_class]<=0:raise ValueError('Impossible ordinary action class')
    if rng.random()<min(1.,p[base_class]/q[base_class]):return base_class
    extra=np.maximum(p-q,0);assert extra.sum()>0
    return rng.choices(range(4),weights=extra.tolist())[0]
def decide(obs,proposal,partner,rng,variant,mode):
    action,size=proposal
    source=0 if action=='fold' else 3 if action=='raise' else 1 if obs.to_call==0 else 2
    q,p=target_distribution(obs.base_action_probabilities,bundle()['soft'].predict(features(obs,partner),num_threads=1)[0],obs.to_call)
    chosen=coupled_class(source,q,p,rng)
    if chosen==source:return proposal,None
    assert chosen!=3
    if chosen==0:return ('fold',size),'partner_distribution_fold'
    return ('call',size),'partner_distribution_passivity' if source==3 else 'partner_distribution_continuation'
'''
def main():
    N.mkdir(exist_ok=False);arms=[];planhash={}
    for b in range(2):
        arm=f'b{b}_coupled';arms.append(arm);d=N/arm;d.mkdir();src=P/f'b{b}_continuous';pp=json.loads((src/'pilot-plan.json').read_text())
        for name,hash in pp['source_hashes'].items():assert sha(src/name)==hash;shutil.copy2(src/name,d/name)
        text=(d/'learned_policy.py').read_text()
        text=text.replace('def decision(obs,profile,rng):','def decision_with_probabilities(obs,profile,rng):')
        text=text.replace("if kind!=3:return ('fold' if kind==0 else 'call'),.75","if kind!=3:return (('fold' if kind==0 else 'call'),.75),tuple(p)")
        text=text.replace("return 'raise',size","return ('raise',size),tuple(p)")
        text+='\ndef decision(obs,profile,rng):\n    return decision_with_probabilities(obs,profile,rng)[0]\n'
        (d/'learned_policy.py').write_text(text,encoding='utf-8')
        text=(d/'generator.py').read_text().replace('    public_history: tuple = ()','    public_history: tuple = ()\n    base_action_probabilities: tuple = ()')
        text=text.replace('from learned_policy import decision\n    return decision(obs,profile,rng)','from learned_policy import decision_with_probabilities\n    return decision_with_probabilities(obs,profile,rng)')
        text=text.replace('proposal=base_decision(obs,profiles[actor],rng)','proposal,base_probabilities=base_decision(obs,profiles[actor],rng)\n        obs=replace(obs,base_action_probabilities=base_probabilities)')
        assert 'proposal,base_probabilities' in text
        (d/'generator.py').write_text(text,encoding='utf-8');(d/'soft_controller.py').write_text(CONTROLLER,encoding='utf-8')
        text=(d/'run_pilot.py').read_text().replace("'partner_conditional_continuation'","'partner_conditional_continuation','partner_distribution_fold','partner_distribution_passivity','partner_distribution_continuation'")
        (d/'run_pilot.py').write_text(text,encoding='utf-8')
        pp.update(created_at=datetime.now(timezone.utc).isoformat(),source_hashes={n:sha(d/n) for n in pp['source_hashes']},paired_existing_control=str(src.resolve()),counterfactual_development_replay=True)
        (d/'pilot-plan.json').write_text(json.dumps(pp,indent=2));planhash[arm]=sha(d/'pilot-plan.json')
    for name in ['controller.joblib','conditional_action_features.py','advanced_features.py','soft_controller.py']:shutil.copy2(N/arms[0]/name,N/name)
    for name in ['audit_batch.py','replay_batch.py','prepare_features.py','run_generation.py']:
        text=(P/name).read_text()
        if name=='audit_batch.py':
            text=text.replace("'partner_conditional_continuation'","'partner_conditional_continuation','partner_distribution_fold','partner_distribution_passivity','partner_distribution_continuation'")
            text=text.replace("assert actual['action'] in ['call','check','all_in']","assert actual['action'] in ['fold','call','check','all_in']")
        (N/name).write_text(text,encoding='utf-8')
    # Source diagnostics keep their classifier/threshold settings unchanged.
    text=(P/'diagnostics.py').read_text()
    start=text.index('    for batch,i in ');end=text.index('    original=np.load',start)
    text=text[:start]+'''    for batch in plan['batches']:
        for i in batch['pool_ids']:
            paths=[Path(batch['existing_control'])/'pilot'/f'pool-{i:04d}',H/batch['arm']/'pilot'/f'pool-{i:04d}']
            prs=[json.loads((p/'provenance.json').read_text()) for p in paths]
            for key in ['profiles','prototype_sources','target','nominal_intensity','onset','duration','blind','tilt_player','tilt_onset','shared_hands']:assert prs[0][key]==prs[1][key],(i,key)
            assert sha(paths[0]/'seat-events.json')==sha(paths[1]/'seat-events.json')
            sets=[pl.read_parquet(p/'seats.parquet').select('hand_id','player_id','seat_no','hole_card_1','hole_card_2').sort('hand_id','seat_no') for p in paths]
            assert sets[0].equals(sets[1]);paired.append(i)
'''+text[end:]
    start=text.index('    common_counts={}');end=text.index("    (H/'diagnostics.json')",start)
    text=text[:start]+'''    result={'training_only':True,'paired_scenarios_verified':paired,'paired_seats_hole_cards_profiles_episodes_exact':True,'controller_feature_parity_rows':parity,'results':output,'source_sha256':sha(__file__),'limitations':'Development replay of previously diagnosed scenarios, not fresh-seed confirmation. Same policy source. Original 100 held pools not loaded.'}
'''+text[end:]
    (N/'diagnostics.py').write_text(text,encoding='utf-8')
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'hypothesis':'One-way replacement loses conditional soft-play folds and distorts the learned action distribution. Exact probability coupling to the own-card/public learned soft policy, with non-increasing aggression, can restore conditional behavior.',
      'generator_arms':arms,'generator_plan_sha256':planhash,'batches':[{'arm':f'b{b}_coupled','existing_control':str((P/f'b{b}_continuous').resolve()),'pool_ids':list(range(start,start+32))} for b,start in enumerate([640,704])],
      'controller_sha256':sha(N/'controller.joblib'),'no_new_model_fit':True,'activation_probability':1.0,
      'purpose':'Training-only mechanism diagnosis and development replay; never call reused scenarios independent confirmation. No outer-held utility evaluation in this stage.',
      'realism_gate':'Same source classifier and seed19083 as previous studies, source AUC<=.85, preflop calling and call-strength gaps<=.05, >=12 supported targets and all4 ordinary-policy families in EACH development batch.',
      'mechanism_check':'Record partner-facing action probabilities and fold-related feature gaps. Verify distribution transport algebra and sampling, no-new-aggression, ordinary/no-op trace parity, exact known-seed exogenous pairing, and legal changed-action evidence.',
      'next_step':'If both development batches pass realism, freeze a fresh-seed confirmation before any new outer-held utility trial. If not, diagnose the remaining training-only mismatch and make a new frozen repair. Never modify completed runs.',
      'protected':json.loads((P/'study-plan.json').read_text())['protected']}
    (N/'study-plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps({'directory':str(N),'plan_sha256':sha(N/'study-plan.json')}))
if __name__=='__main__':main()
