"""Verify and archive all new research artifacts before stopping the VM."""
from pathlib import Path
import hashlib,json,tarfile
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (H/'summary.json').exists() and (H/'trial/returned-manifest.json').exists()
remote=json.loads((H/'trial/returned-manifest.json').read_text())
assert all(sha(H/'trial'/n)==v for n,v in remote.items())
files=[p for p in H.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','analysis_package'])
       and not p.name.endswith(('.tar.gz','.zip')) and p.name not in ['archive-manifest.json','archive-receipt.json']]
manifest={p.relative_to(H).as_posix():sha(p) for p in files}
(H/'archive-manifest.json').write_text(json.dumps(manifest,indent=2))
dest=H.parent/'history-augmentation-20260910.tar.gz';assert not dest.exists()
with tarfile.open(dest,'w:gz') as t:
    for p in files+[H/'archive-manifest.json']:t.add(p,arcname=p.relative_to(H))
verified=0
with tarfile.open(dest) as t:
    for name,value in manifest.items():
        assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==value;verified+=1
result={'archive':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size,'verified_members':verified,
        'remote_result_members_verified':len(remote),'contains_new_raw_histories_models_predictions_code_and_audits':True,
        'excluded_redundant_archives_and_extracted_reference_package':True,
        'reference_package_sha256':sha(H/'analysis-package.tar.gz')}
(H/'archive-receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
