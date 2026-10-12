from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((HERE/'generator-tests.json').read_text())['passed']
assert not (HERE/'pilot-plan.json').exists()
families=['none','directed_transfer','soft_play','coordinated_isolation']
specs=[{'pool_id':i,'seed':908117+i*7919,'policy_family':(i//4)%4,
        'behavior':families[i%4],'hands':3000} for i in range(32)]
plan={'created_at':datetime.now(timezone.utc).isoformat(),
      'purpose':'Pilot for diverse valid synthetic histories, not a claim of 10x independent evidence.',
      'source_hashes':{p:sha(HERE/p) for p in ['generator.py','run_pilot.py','test_generator.py']},
      'pokerkit_version':'0.7.5','wheel_sha256':'3e1a0b2a8a9785369c86e2c1392dd2204ee536fbcbe77a779c875bd545eb63e8',
      'training_calibration_sha256':sha(HERE/'training-calibration.json'),
      'pools':specs,'workers':2,
      'design':'32 separate pools; 30 players each; uniform six-player seating; 3000 hands per pool; four structural policy families crossed with none/three disclosed coordination mechanisms. Two pools per cell. Five labelled pairs per pool; no positive when the controller never changes a legal action. Other pairs remain outside the supervised sample.',
      'intervention_limits':'Four hand-built policy families and controlled interventions are an approximation, not the organizer generator. No invented example of the undisclosed fourth mechanism. All ordinary policies use only own cards and public state.',
      'validity_gate':{'illegal_state_or_accounting_errors':0,'duplicate_hand_payloads':0,'minimum_shared_hands_per_labelled_pair':20,'minimum_intervened_hands_per_positive_pair':1},
      'diversity_gate':{'all_16_policy_behavior_cells_required':True,'maximum_pool_weight_fraction':.04,'maximum_policy_family_fraction':.30,
                        'all_three_mechanisms_must_act':True},
      'domain_check':'Report original-training versus synthetic action/board/history distributions and grouped domain-classifier discrimination. No held labels used for generator calibration. A large discrepancy requires repair before volume scaling; do not declare validity alone sufficient.',
      'next_stage':'After validity and diversity checks, freeze a matched original-only vs pilot-augmented comparison for mHC, a matched residual model, the incumbent conditional tree and the unsubmitted v15 tree. If useful, progress to ~280 synthetic pools (about 2x total labelled pairs) then ~2520 (about10x). External action pretraining is a separate experiment, not pooled as pair labels.',
      'no_held_labels_or_original_evaluation_data':True}
(HERE/'pilot-plan.json').write_text(json.dumps(plan,indent=2))
files=['generator.py','run_pilot.py','test_generator.py','pilot-plan.json','training-calibration.json','pokerkit-0.7.5-py3-none-any.whl']
manifest={p:sha(HERE/p) for p in files}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(HERE/'pilot-package.tar.gz','w:gz') as tar:
    for p in files+['manifest.json']: tar.add(HERE/p,arcname=p)
print(json.dumps({'plan_sha256':sha(HERE/'pilot-plan.json'),'archive_sha256':sha(HERE/'pilot-package.tar.gz'),'bytes':(HERE/'pilot-package.tar.gz').stat().st_size}))
