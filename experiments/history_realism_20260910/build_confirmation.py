"""Trigger the prespecified fresh-seed confirmation without changing the recipe."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).resolve().parent;S=H/'persistent_sessions';D=H/'session_confirmation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((S/name).read_text())
c=json.loads((H/'confirmation-plan.json').read_text());plan=read('pilot-plan.json')
assert read('pair-domain-audit.json')['auc']<=c['pair_domain_auc_ceiling']
assert read('temporal-audit.json')['passed']
assert not read('pilot-audit.json')['domain_discrepancy_flag']
assert read('balance-audit.json')['continuation_gate_passed'] and read('balance-audit.json')['coverage_passed']
assert read('seat-schedule-audit.json')['schedule_gate_passed'] and read('replay-audit.json')['all_serialized_actions_replayed_legally']
assert all(sha(S/name)==value for name,value in plan['source_hashes'].items())
assert not D.exists();D.mkdir()
for name in [*plan['source_hashes'],'training-calibration.json','physical-seat-tests.json','generator-tests.json']:
    shutil.copy2(S/name,D/name)
plan['created_at']=datetime.now(timezone.utc).isoformat();plan['parent_plan_sha256']=sha(S/'pilot-plan.json');plan['confirmation_plan_sha256']=sha(H/'confirmation-plan.json')
for row in plan['pools']:
    row['seed']+=c['seed_offset_from_first_session_pilot'];row['pool_id']+=32
plan['confirmation']='Identical generator/model/source bytes. Fresh seeds with predeclared+400000 offset;pool identifiers offset32 to prevent collisions. No parameter or threshold change.'
(D/'pilot-plan.json').write_text(json.dumps(plan,indent=2))
for old,new in [('prepare_session_features.py','prepare_confirmation_features.py'),('session_pair_domain_audit.py','confirmation_pair_domain_audit.py'),('audit_session_pilot.py','audit_confirmation_pilot.py'),('replay_session_pilot.py','replay_confirmation_pilot.py')]:
    t=(H/old).read_text().replace("D=H/'persistent_sessions'","D=H/'session_confirmation'");(H/new).write_text(t)
    assert t.replace("D=H/'session_confirmation'","D=H/'persistent_sessions'")==(H/old).read_text()
auditnames=['prepare_confirmation_features.py','confirmation_pair_domain_audit.py','audit_confirmation_pilot.py','replay_confirmation_pilot.py','temporal_audit.py','audit_seat_schedule.py']
(D/'auditor-freeze.json').write_text(json.dumps({n:sha(H/n) for n in auditnames},indent=2))
print(json.dumps({'plan_sha256':sha(D/'pilot-plan.json'),'same_source_hashes':plan['source_hashes']==read('pilot-plan.json')['source_hashes'],'new_pools':32,'new_hands':96000}))
