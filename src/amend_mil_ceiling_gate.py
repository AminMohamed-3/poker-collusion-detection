"""Correct an impossible screening threshold before importing any MIL outcomes."""
from datetime import datetime,timezone
import json
from robust_pair_experiment import ROOT,sha

out=ROOT/'reports/mil-pair-experiment'
path=out/'ceiling-amendment.json'; assert not path.exists()
assert not list(out.glob('*/returned')) and not list(out.glob('*/screen.json'))
payload={'created_at':datetime.now(timezone.utc).isoformat(),'original_plan_sha256':sha(out/'plan.json'),
    'reason':'Historical incumbent full/short pair-behavior mean is .7991116686912437. Its absolute maximum is .8; the original minimum gain .001 exceeds the .0008883313087563 remaining ceiling. This was a plan-design error, not a model failure.',
    'timing':'No neural held predictions have been imported, read or scored. GPU training is in progress. A distinct event-tree experiment has negative deltas and remains rejected under its own unchanged plan.',
    'corrected_minimum':'For known screen/confirmation means only, min(original .001, .20 * max(0, .8 - mean incumbent pair_behavior)). All other gates unchanged. This demands at least one fifth of remaining upper-bound improvement when the metric is saturated.',
    'ceiling':.8,'fraction_remaining':.2,'maximum_minimum_gain':.001,
    'applies_to':['screen_mean_pair_behavior_delta_min','confirmation_known_mean_pair_behavior_delta_min'],
    'no_subgroup_rescue':True,'held_outcome_availability_checked':True}
path.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({'amendment_sha256':sha(path)}))
