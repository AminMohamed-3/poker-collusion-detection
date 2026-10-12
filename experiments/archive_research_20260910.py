"""Archive the completed local research and verify every archived file's bytes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile

H = Path(__file__).resolve().parent
W = H.parents[2]
O = H/'research-backup-20260910'
O.mkdir(exist_ok=True)
archive = O/'research.tar.gz'
assert not archive.exists()
excluded_parts = {'__pycache__', 'downloads', 'vendor'}
files = []
for folder in [H/'mhc_local', H/'data_expansion']:
    for p in folder.rglob('*'):
        if p.is_file() and not excluded_parts.intersection(p.relative_to(folder).parts) and not p.name.endswith('.part'):
            files.append(p)
files += [H/n for n in ['finalize_research_20260910.py', 'archive_research_20260910.py', 'instance_state_20260910.py', 'research-completion-20260910.json']]
files += [W/'research'/n for n in ['POKER_RESEARCH_MEMORY.md','poker-rl-generator-feasibility-20260910.md','poker-data-expansion-20260910.md']]
files.append(W/'STATUS.md')
files = sorted(set(files))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

manifest = {p.relative_to(W).as_posix(): {'sha256':sha(p), 'bytes':p.stat().st_size} for p in files}
(O/'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'files':len(files), 'uncompressed_bytes':sum(v['bytes'] for v in manifest.values())}), flush=True)
with tarfile.open(archive, 'w:gz', compresslevel=3) as t:
    for p in files:
        t.add(p, arcname=p.relative_to(W))
    t.add(O/'manifest.json', arcname='archive-manifest.json')
with tarfile.open(archive, 'r:gz') as t:
    members = t.getmembers()
    assert len(members) == len(manifest)+1
    for m in members:
        assert m.isfile()
        if m.name == 'archive-manifest.json':
            assert json.loads(t.extractfile(m).read()) == manifest
            continue
        assert m.size == manifest[m.name]['bytes']
        h = hashlib.sha256()
        with t.extractfile(m) as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''):
                h.update(chunk)
        assert h.hexdigest() == manifest[m.name]['sha256'], m.name
receipt = {'at':datetime.now(timezone.utc).isoformat(), 'archive':str(archive), 'sha256':sha(archive), 'bytes':archive.stat().st_size, 'verified_members':len(manifest), 'excluded':'Dependency/download caches; pinned PokerKit wheel is included. Original competition source data and existing production artifacts retain their earlier separate backups.', 'no_secrets_requested_or_included_by_this_archiver':True}
(O/'receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2), flush=True)
