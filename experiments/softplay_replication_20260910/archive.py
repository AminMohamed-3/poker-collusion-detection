"""Hash and verify the completed local replication and accompanying notes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil
H=Path(__file__).resolve().parent;W=H.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert (H/'summary.json').exists() and (H/'supplementary-audit.json').exists()
    for source,name in [(W/'STATUS.md','workspace-status.md'),(W/'research/POKER_RESEARCH_MEMORY.md','research-memory.md')]:shutil.copy2(source,H/name)
    shutil.copy2(H.parent/'setup_softplay_replication.py',H/'setup-study-snapshot.py')
    files=[p for p in sorted(H.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive-manifest.json','archive-receipt.json']]
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'files':{p.relative_to(H).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
    (H/'archive-manifest.json').write_text(json.dumps(manifest,indent=2));files.append(H/'archive-manifest.json')
    out=H.parent/'softplay-replication-20260910.zip';assert not out.exists()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,p.relative_to(H).as_posix())
    with zipfile.ZipFile(out) as z:
        assert len(z.namelist())==len(files) and z.testzip() is None
        for n,r in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==r['sha256']
        assert hashlib.sha256(z.read('archive-manifest.json')).hexdigest()==sha(H/'archive-manifest.json')
    receipt={'archive':str(out.resolve()),'bytes':out.stat().st_size,'sha256':sha(out),'verified_members':len(files),'manifest_sha256':sha(H/'archive-manifest.json'),'all_archived_hashes_match':True}
    (H/'archive-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
