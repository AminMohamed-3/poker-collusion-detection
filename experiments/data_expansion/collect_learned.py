from pathlib import Path
import hashlib,json,tarfile
H=Path('/workspace/data-expansion-20260910/learned')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
g=json.loads((H/'pilot-generation.json').read_text());assert len(g['pools'])==32
repair=json.loads((H/'serialization-repair.json').read_text());assert g['serialization_repair_sha256']==sha(H/'serialization-repair.json')
assert all(sha(H/'pilot'/n)==v for n,v in repair['preserved_payloads'].items())
result={'all_preserved_payloads_identical':True,'verified_files':len(repair['preserved_payloads']),'plan_sha256':g['plan_sha256'],'repair_sha256':sha(H/'serialization-repair.json')}
(H/'repair-verification.json').write_text(json.dumps(result,indent=2))
files=[p for p in (H/'pilot').rglob('*') if p.is_file()]+[H/n for n in ['pilot-generation.json','generator-tests.json','generation.log','serialization-repair.json','repair-verification.json','run_pilot.py']]
manifest={p.relative_to(H).as_posix():sha(p) for p in files};(H/'results-manifest.json').write_text(json.dumps(manifest,indent=2));files.append(H/'results-manifest.json')
with tarfile.open(H/'learned-results.tar.gz','w:gz') as t:
    for p in files:t.add(p,arcname=p.relative_to(H))
print(json.dumps({'sha256':sha(H/'learned-results.tar.gz'),'bytes':(H/'learned-results.tar.gz').stat().st_size,**result}))
