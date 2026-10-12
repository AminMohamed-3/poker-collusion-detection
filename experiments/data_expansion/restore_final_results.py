"""Verify the two completed remote result archives before restoring artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

H = Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

receipts = []
for rel, expected, destination in [
    ('augmentation/remote-results.tar.gz', '46e5a68a3ad18e255b4de5239942dd9342f70004488faabcbd8831016387e68c', 'augmentation/remote-returned'),
    ('profile-repair-results.tar.gz', '569436805e284518e962d66187902aed3f1e1087a9417f5b11875b927e9c8f59', 'profile-returned'),
]:
    archive, out = H / rel, H / destination
    assert sha(archive) == expected, archive
    out.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:gz') as t:
        for member in t.getmembers():
            assert member.isfile(), member.name
            target = (out / member.name).resolve()
            assert target.is_relative_to(out.resolve()), member.name
            if target.exists():
                assert target.read_bytes() == t.extractfile(member).read(), target
        t.extractall(out, filter='data')
    manifest = json.loads((out / 'results-manifest.json').read_text())
    for name, expected_member in manifest.items():
        assert sha(out / name) == expected_member, name
    receipts.append({'archive': rel, 'sha256': expected, 'bytes': archive.stat().st_size, 'verified_members': len(manifest)})

profile_source = H / 'profile-returned'
assert not (H / 'profile_repair/pilot').exists()
shutil.copytree(profile_source / 'pilot', H / 'profile_repair/pilot')
for name in ['pilot-generation.json', 'generation.log', 'results-manifest.json']:
    assert not (H / 'profile_repair' / name).exists()
    shutil.copy2(profile_source / name, H / 'profile_repair' / name)
shutil.copy2(profile_source / 'generator-tests.json', H / 'profile_repair/remote-generator-tests.json')

source = H / 'augmentation/remote-returned'
queue = json.loads((source / 'queue.json').read_text())
assert len(queue['completed']) == 18
for row in queue['completed']:
    rel = Path(row['arm']) / f"{row['variant']}-{row['seed']}"
    assert not (H / 'augmentation' / rel).exists()
    shutil.copytree(source / rel, H / 'augmentation' / rel)

(H / 'final-remote-backup-receipt.json').write_text(json.dumps(receipts, indent=2))
print(json.dumps(receipts, indent=2))
