"""Numpy scalar JSON repair; preserve every generated payload for exact comparison."""
from pathlib import Path
import hashlib,json,sys
H=Path(sys.argv[1]).resolve()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=H/'run_pilot.py';before=sha(p)
assert before=='2a17c177c705d529c6b2f293605b1b64bb1e5608cfcebe29da95880f3fc436aa'
s=p.read_text().replace('json.dumps(provenance,indent=2)','json.dumps(provenance,indent=2,default=float)')
s=s.replace("    assert all(sha(HERE/p)==h for p,h in plan['source_hashes'].items())","    repair=json.loads((HERE/'serialization-repair.json').read_text()); assert repair['plan_sha256']==sha(plan_path)\n    expected={**plan['source_hashes'],**repair['source_overrides']}\n    assert all(sha(HERE/p)==h for p,h in expected.items())")
s=s.replace("receipt={'plan_sha256':sha(plan_path),", "receipt={'plan_sha256':sha(plan_path),'serialization_repair_sha256':sha(HERE/'serialization-repair.json'),")
p.write_text(s)
payloads={}
if (H/'pilot').exists():
    for d in (H/'pilot').iterdir():
        if (d/'interventions.json').exists():
            for f in d.iterdir():
                if f.is_file():payloads[f.relative_to(H/'pilot').as_posix()]=sha(f)
    (H/'pilot').rename(H/'pilot-before-serialization-repair')
receipt={'reason':'numpy.float32 profile values cannot be serialized by stdlib JSON. default=float changes provenance serialization only; all simulation/policy/model sources and seeds remain frozen. Existing completed raw payloads will be compared byte for byte after rerun.','old_source_sha256':before,'source_overrides':{'run_pilot.py':sha(p)},'plan_sha256':sha(H/'pilot-plan.json'),'preserved_payloads':payloads}
(H/'serialization-repair.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({'repair_sha256':sha(H/'serialization-repair.json'),'preserved_payloads':len(payloads),'run_source_sha256':sha(p)}))
