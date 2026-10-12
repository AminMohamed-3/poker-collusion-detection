"""Freeze three paired generators and six small detector fits at two data doses."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).resolve().parent;OLD=H.parent/'softplay_fast_20260910';SRC=OLD/'repaired'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (H/'study-plan.json').exists()
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'hypothesis':'Condition soft-play on own-card strength and public pressure. Test whether more unique eligible synthetic pairs improve utility beyond equal-sized real resampling.',
      'generator_arms':['baseline','strength','learned'],'scenario_pools':32,'hands_per_pool':3000,'raw_histories_total':288000,'independent_exogenous_scenarios':96000,'paired_note':'Same seeds, initial cards, profiles, seats and episode plans across generator arms. Balance trajectories may diverge. Paired arms are counterfactual histories, not independent new scenarios.',
      'controller_parameters':{'strength':'Only replace ordinary folds with calls above the original soft-call 25th-percentile own-strength threshold within street. Existing check/raise suppression retained.','learned':'Fixed depth-3 soft and normal action models; propose only passive choices whose predicted probability increases over normal. No private partner cards. Original episode/intensity gate retained.'},
      'controller_training':'Original 297 training pools only; three-fold pool OOF comparison with a normal-action model; final models fit only these pools. No validation feedback used to choose controller parameters.',
      'cohort':'Intersection of supported soft targets in strength and learned arms. Low dose is common-supported scenarios 0..15; high is scenarios 0..31. All included targets are unique; each contributes its full/70/40 history views. At least 6 low and 12 high targets, all four ordinary policies represented at high dose, or stop before utility fits.',
      'detector_arms':['resampled_low','strength_low','learned_low','resampled_high','strength_high','learned_high'],'tree_parameters':json.loads((OLD/'study-plan.json').read_text())['tree_parameters'],
      'training':'Same original 4209 rows and 1217 features. Add only soft-play targets. Real controls use unique original soft pairs without replacement, nested by dose, and the identical three views. Dose-specific controls match counts and class weights exactly. No replicated synthetic target to meet a quota.',
      'utility_gate':'Against matched-dose resampling: mean logloss reduction >=.001, mean Pair AP and sparse AP deltas >=0, worst history Pair AP delta >=-.001, mean loss reduction nonnegative in both fixed hashed validation-pool halves. Report high-vs-low increments in both treatment and control. Passing is a replication lead only.',
      'realism_gate':'Grouped training-only soft source AUC <=.85; at high dose at least 12 targets and four ordinary policies. Report call/check rates and conditional strengths; no fitted source-reweighting or source-classifier reward.',
      'evaluation':'All six fits and predictions freeze before labels are scored once. Reuse full/shortened original 100 outer-held pools, same softplay-fast-v1 SHA256 halves. Paired pool bootstrap1000 seed20319. Not untouched, no new independent test claim. No leaderboard feedback or official-evidence-ranker claim.',
      'stop':'Six fixed tree fits maximum. No neural runs or further generation after this frozen study. Retain all failed gates and report every arm; new user instruction authorized this separate follow-up, not revision of prior failures.',
      'protected':json.loads((OLD/'study-plan.json').read_text())['protected'],'controller_fit_sha256':sha(H/'controller-fit.json')}
    for mode in plan['generator_arms']:
        d=H/mode;d.mkdir()
        for n in ['generator.py','run_pilot.py','learned_policy.py','models.joblib','fit.json','sampler.json','profiles.json','training-calibration.json','balance-process.json','attendance-process.json','conditional_action_features.py','advanced_features.py']:shutil.copy2(SRC/n,d/n)
        for n in ['soft_controller.py','controller.joblib']:shutil.copy2(H/n,d/n)
        source=(d/'generator.py').read_text().replace('    own_history: tuple = ()','    own_history: tuple = ()\n    public_history: tuple = ()')
        start=source.index("    elif family=='soft_play':");end=source.index("    elif family=='coordinated_isolation':",start)
        source=source[:start]+f"    elif family=='soft_play':\n        from learned_policy import SOFT_VARIANT\n        from soft_controller import decide\n        return decide(obs,proposal,partner_index,rng,SOFT_VARIANT,'{mode}')\n"+source[end:]
        needle='        proposal=base_decision(obs,profiles[actor],rng)'
        replacement="        from dataclasses import replace\n        obs=replace(obs,public_history=tuple((players.index(a['player_id']),0 if a['action']=='fold' else 1 if a['action']=='check' else 2 if a['action']=='call' or (a['action']=='all_in' and a['amount']<=a['to_call']) else 3,STREETS.index(a['street']),a['amount']) for a in actions))\n"+needle
        assert needle in source;source=source.replace(needle,replacement);(d/'generator.py').write_text(source)
        source=(d/'run_pilot.py').read_text()
        source=source.replace("        for player in sitting:\n            if player in previously_seen:","        reset_rng=random.Random(spec['seed']*10000+h+87300317)\n        for player in sitting:\n            reset_coin=reset_rng.random()\n            reset_value=round(blind*reset_rng.choices(BALANCE['reset_stack_bb'],BALANCE['reset_weights'])[0])\n            if player in previously_seen:")
        source=source.replace("rng.random()<BALANCE['spontaneous_reset_probability']","reset_coin<BALANCE['spontaneous_reset_probability']").replace("round(blind*rng.choices(BALANCE['reset_stack_bb'],BALANCE['reset_weights'])[0])","reset_value")
        source=source.replace("'partner_multiway_check'}","'partner_multiway_check','partner_conditional_passivity','partner_conditional_continuation'}")
        (d/'run_pilot.py').write_text(source)
        pools=[{'pool_id':384+i,'scenario':i,'seed':210319+997*i,'policy_family':i%4,'soft_variant':(i//4)%4,'behavior':'soft_play','hands':3000,'controller':mode} for i in range(32)]
        sources={p.name:sha(p) for p in d.iterdir() if p.is_file()}
        (d/'pilot-plan.json').write_text(json.dumps({'created_at':plan['created_at'],'pools':pools,'source_hashes':sources,'training_only':True},indent=2))
    plan['generator_plan_sha256']={m:sha(H/m/'pilot-plan.json') for m in plan['generator_arms']}
    (H/'study-plan.json').write_text(json.dumps(plan,indent=2));print('Frozen paired 32-pool generators and six detector fits.')
if __name__=='__main__':main()
