"""Preserve this local mini-goal, including failed stages and both final batches."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile
H=Path(__file__).resolve().parent;W=H.parents[3];A=H.parent/'history-realism-20260910.tar.gz'
assert not A.exists()
files=[p for p in H.rglob('*') if p.is_file() and not {'vendor','__pycache__'}.intersection(p.relative_to(H).parts) and p.name not in ['archive-manifest.json','archive-receipt.json']]
files += [W/'STATUS.md',W/'research/POKER_RESEARCH_MEMORY.md',H.parent/'data_expansion/pokerkit-0.7.5-py3-none-any.whl']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

m={p.relative_to(W).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)}
(H/'archive-manifest.json').write_text(json.dumps(m,indent=2))
print(json.dumps({'files':len(m),'uncompressed_bytes':sum(v['bytes'] for v in m.values())}),flush=True)
with tarfile.open(A,'w:gz',compresslevel=3) as t:
    for p in sorted(files):t.add(p,arcname=p.relative_to(W))
    t.add(H/'archive-manifest.json',arcname='archive-manifest.json')
with tarfile.open(A,'r:gz') as t:
    assert len(t.getmembers())==len(m)+1
    for member in t.getmembers():
        assert member.isfile()
        if member.name=='archive-manifest.json':assert json.loads(t.extractfile(member).read())==m;continue
        assert member.size==m[member.name]['bytes']
        h=hashlib.sha256()
        with t.extractfile(member) as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==m[member.name]['sha256'],member.name
r={'at':datetime.now(timezone.utc).isoformat(),'archive':str(A),'sha256':sha(A),'bytes':A.stat().st_size,'verified_members':len(m),'excluded':'Extracted dependency and interpreter caches; pinned wheel is included. Existing original feature references and models also retain prior separate backups.','all_work_local':True,'no_vm_restart':True}
(H/'archive-receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
