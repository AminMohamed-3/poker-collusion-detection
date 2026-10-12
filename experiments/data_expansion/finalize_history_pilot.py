"""Before first generation, keep evidence only for pairs with adequate history."""
from pathlib import Path
import hashlib,json,tarfile
H=Path(__file__).resolve().parent;D=H/'history_repair'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (D/'pilot').exists()
old=(D/'pilot-plan.json').read_bytes();plan=json.loads(old)
assert all(sha(D/n)==v for n,v in plan['source_hashes'].items())
(D/'pilot-plan-before-label-filter.json').write_bytes(old)
p=D/'run_pilot.py';s=p.read_text();needle="    for name,rows in [('hands',hands)";assert needle in s
s=s.replace(needle,"    eligible={r['pair_id'] for r in labels}\n    evidence=[r for r in evidence if r['pair_id'] in eligible]\n"+needle);p.write_text(s)
plan['prelaunch_parent_sha256']=hashlib.sha256(old).hexdigest();plan['source_hashes']['run_pilot.py']=sha(p)
plan['prelaunch_repair']='Drop evidence for any assigned pair excluded by the minimum20 shared-hand rule. Recorded before first generation or any utility outcome.'
(D/'pilot-plan.json').write_text(json.dumps(plan,indent=2))
with tarfile.open(H/'history-repair-package.tar.gz','w:gz') as t:
    for p in D.iterdir():
        if p.is_file():t.add(p,arcname=p.name)
print(json.dumps({'plan_sha256':sha(D/'pilot-plan.json'),'package_sha256':sha(H/'history-repair-package.tar.gz')}))
