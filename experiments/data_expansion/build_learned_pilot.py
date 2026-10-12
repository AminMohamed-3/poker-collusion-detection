"""Preserve pilot v1; derive and freeze the documented behavior-clone repair."""
from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone

H=Path(__file__).resolve().parent;D=H/'learned';R=H.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (D/'pilot-plan.json').exists()
s=(H/'generator.py').read_text()
s=s.replace("sys.path.insert(0, str(HERE/'vendor'))","sys.path.insert(0, str(HERE.parent/'vendor'))")
s=s.replace('    actor: int\n','    actor: int\n    prior_actions: int = 0\n    own_history: tuple = ()\n')
s=s.replace('    sizing: float\n','    sizing: float\n    style: tuple = (.6,.07,.13,.2)\n')
start=s.index('def base_decision(');end=s.index('\ndef intervene(',start)
s=s[:start]+'''def base_decision(obs, profile, rng):
    from learned_policy import decision
    return decision(obs,profile,rng)

'''+s[end:]
s=s.replace("f'S{pool_id:05d}H{hand_number:05d}'", "f'SL{pool_id:05d}H{hand_number:05d}'").replace("f'SPOOL{pool_id:05d}'","f'SLPOOL{pool_id:05d}'")
s=s.replace('previous,actor)\n','previous,actor,len(actions),tuple((0 if a[\'action\']==\'fold\' else 1 if a[\'action\']==\'check\' else 2 if a[\'action\']==\'call\' or (a[\'action\']==\'all_in\' and a[\'amount\']<=a[\'to_call\']) else 3,a[\'amount\'],STREETS.index(a[\'street\'])) for a in actions if a[\'player_id\']==players[actor]))\n')
needle='    board=tuple(repr(c) for c in state.get_board_cards(0))\n    if len(board)==5'
replacement='''    # Match the organizer's gross-contribution convention: uncalled winning
    # wagers count in the displayed pot and are returned in the winner's payout.
    # Engine net winnings remain authoritative and unchanged.
    contribution=[sum(a['amount'] for a in actions if a['player_id']==players[i])+(blind//2 if i==0 else blind if i==1 else 0) for i in range(6)]
    payouts=[state.stacks[i]-stacks[i]+contribution[i] for i in range(6)]
    assert min(payouts)>=0
    final_pot=sum(contribution)
    assert sum(payouts)==final_pot
    board=tuple(repr(c) for c in state.get_board_cards(0))
    if len(board)==5'''
assert needle in s;s=s.replace(needle,replacement)
(D/'generator.py').write_text(s)
s=(H/'run_pilot.py').read_text().replace('from generator import Profile,simulate_hand','from generator import Profile,simulate_hand\nfrom learned_policy import sample_style')
s=s.replace("f'SP{pool_id:04d}","f'SLP{pool_id:04d}")
s=s.replace('rng.uniform(.5,1.3)) for p in players','rng.uniform(.5,1.3),sample_style(rng)) for p in players')
s=s.replace('intensity=rng.uniform(.2,.85); onset=rng.uniform(.05,.45); duration=rng.uniform(.25,.55)','intensity=rng.uniform(.4,.9); onset=rng.uniform(.05,.35); duration=rng.uniform(.35,.6)')
s=s.replace("            evidence.append({'pair_id':pair_id,'hand_id':hand['hand_id']})", "            essential={'directed_transfer':{'donor_calls_partner_while_behind','donor_folds_stronger_hand'},'soft_play':{'partner_check','partner_raise_suppressed'},'coordinated_isolation':{'pressure_outsiders'}}[spec['behavior']]\n            if any(t['reason'] in essential for t in tt): evidence.append({'pair_id':pair_id,'hand_id':hand['hand_id']})")
(D/'run_pilot.py').write_text(s)
shutil.copy2(H/'test_generator.py',D/'test_generator.py')
for name in ['conditional_action_features.py','advanced_features.py']:shutil.copy2(R/'src'/name,D/name)
shutil.copy2(H/'training-calibration.json',D/'training-calibration.json')
plan=json.loads((H/'pilot-plan.json').read_text());plan['created_at']=datetime.now(timezone.utc).isoformat()
plan['parent_plan_sha256']=sha(H/'pilot-plan.json');plan['parent_audit_sha256']=sha(H/'pilot-audit.json')
plan['source_hashes']={n:sha(D/n) for n in ['generator.py','run_pilot.py','test_generator.py','learned_policy.py','conditional_action_features.py','advanced_features.py','models.joblib','fit.json']}
plan['repair']='Ordinary policies are four related behavior clones fit only to confirmed non-target outer-training actions; own/public state only, disjoint inner-pool diagnostic scoring. Use organizer gross-pot convention, verified on raw records. Increased intervention duration/intensity; positive evidence requires an essential mechanism action. Original held labels never used.'
plan['intervention_limits']='Behavior clones share one source dataset and are not independent sources. Controls approximate disclosed coordination; no unknown fourth-family simulation. A passing domain audit alone is insufficient to claim label quality or utility.'
(D/'pilot-plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps({'plan_sha256':sha(D/'pilot-plan.json'),'files':plan['source_hashes']},indent=2))
