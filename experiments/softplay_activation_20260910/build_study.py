"""Freeze activation-only comparisons with independent controller randomness."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).resolve().parent;P=H.parent/'softplay_conditional_20260910';SRC=P/'learned'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (H/'study-plan.json').exists();cal=json.loads((H/'calibration.json').read_text());gamma=cal['calibrated_gamma'];assert .05<gamma<.95
    previous=json.loads((P/'study-plan.json').read_text());plan={'created_at':datetime.now(timezone.utc).isoformat(),'hypothesis':'A learned policy fit to mixed soft/normal actions is attenuated again by episodic and probability gates. Continuous, training-calibrated activation may improve conditional behavior and data utility.',
      'generator_arms':['gated','calibrated','continuous'],'scenario_pools':32,'hands_per_pool':3000,'calibrated_probability':gamma,'calibration_sha256':sha(H/'calibration.json'),
      'activation':{'gated':'Original episode window and sampled .4-.9 activation probability.','calibrated':'Every co-presence opportunity; activation probability fixed by training-only call-rate equation.','continuous':'Every co-presence opportunity; activation probability 1, a fixed sensitivity endpoint.'},
      'new_randomness_guard':'All arms use a separate controller RNG keyed by hand seed and action index. No-op interventions must not advance ordinary-policy RNG or change the history. This fixes a confound shared by prior simulations, whose reported failures remain unchanged.',
      'pairing':'Fresh scenarios 512..543, seeds independent of calibration. Same initial cards, seating, profiles and nominal episode/intensity draws across all arms; only effective activation differs. Three histories per scenario are counterfactual replays, not independent populations.',
      'cohort':'Intersection of supported soft targets in all three arms; at least 12 unique targets, all four ordinary-policy families. Each target contributes full/70/40 views without replacement. Matched resampling uses the same count of unique real training soft pairs and views.',
      'detector_arms':['resampled','gated','calibrated','continuous'],'tree_parameters':previous['tree_parameters'],'detector_fits':4,
      'utility_gate':'Versus matched resampling: mean multiclass logloss reduction >=.001, mean Pair AP and sparse AP deltas >=0, worst history Pair AP delta >=-.001, nonnegative mean loss benefit in each predefined hashed validation-pool half. Also report both activation treatments versus the fresh gated arm. Passing only justifies replication.',
      'realism_gate':'Pool-grouped training-only soft source AUC <=.85; preflop call-rate and conditional call-strength absolute gaps <=.05 versus original training; at least 12 supported targets and four ordinary policies.',
      'evaluation':'Freeze all four models/predictions before scoring reused original 100 held pools / 457 pairs in full and shortened histories. Same softplay-fast-v1 SHA256 halves, paired pool bootstrap1000 seed20319. No untouched-holdout claim, leaderboard feedback or evidence-ranker evaluation.',
      'stop':'Four fixed detector fits; no further generation, neural training, weight tuning or post-score rescue in this study. Report all arms and gate failures.',
      'protected':previous['protected'],'controller_sha256':sha(P/'controller.joblib')}
    shutil.copy2(P/'controller.joblib',H/'controller.joblib');shutil.copy2(P/'soft_controller.py',H/'soft_controller.py')
    for mode in plan['generator_arms']:
        d=H/mode;d.mkdir()
        for n in ['generator.py','run_pilot.py','learned_policy.py','models.joblib','fit.json','sampler.json','profiles.json','training-calibration.json','balance-process.json','attendance-process.json','conditional_action_features.py','advanced_features.py','soft_controller.py','controller.joblib']:shutil.copy2(SRC/n,d/n)
        s=(d/'generator.py').read_text();old="chosen,reason=intervene(obs,proposal,holes[partner],not folded[partner],partner,family,intensity,rng,players[actor]==target[0])";new="controller_rng=random.Random(seed*1000003+len(actions)*9176+27183)\n            chosen,reason=intervene(obs,proposal,holes[partner],not folded[partner],partner,family,intensity,controller_rng,players[actor]==target[0])";assert old in s;s=s.replace(old,new);(d/'generator.py').write_text(s)
        s=(d/'run_pilot.py').read_text();needle="    balances={p:round(BALANCE['first_stack_by_player'][prototypes[p]['source_player_id']]*blind) for p in players}";replacement="    nominal_intensity=intensity\n    if spec['activation_mode']!='gated':intensity=spec['activation_probability']\n"+needle;assert needle in s;s=s.replace(needle,replacement)
        s=s.replace("active=onset<=h/spec['hands']<onset+duration","active=(onset<=h/spec['hands']<onset+duration) if spec['activation_mode']=='gated' else True")
        s=s.replace("'intensity':intensity,'onset':onset", "'intensity':intensity,'nominal_intensity':nominal_intensity,'activation_mode':spec['activation_mode'],'onset':onset");(d/'run_pilot.py').write_text(s)
        pools=[{'pool_id':512+i,'scenario':i,'seed':410719+997*i,'policy_family':i%4,'soft_variant':(i//4)%4,'behavior':'soft_play','hands':3000,'activation_mode':mode,'activation_probability':None if mode=='gated' else gamma if mode=='calibrated' else 1.} for i in range(32)]
        (d/'pilot-plan.json').write_text(json.dumps({'created_at':plan['created_at'],'pools':pools,'source_hashes':{p.name:sha(p) for p in d.iterdir() if p.is_file()},'training_only':True},indent=2))
    plan['generator_plan_sha256']={m:sha(H/m/'pilot-plan.json') for m in plan['generator_arms']};(H/'study-plan.json').write_text(json.dumps(plan,indent=2));print('Frozen four detector fits and three fresh paired activation batches.')
if __name__=='__main__':main()
