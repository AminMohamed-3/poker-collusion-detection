"""Verify a completed capacity/size research snapshot, including failed utility."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
E=Path(__file__).resolve().parent;W=E.parents[2]
FOLDERS=['softplay_capacity_20260910','softplay_capacity_confirmation_20260910','softplay_capacity_utility_20260910','softplay_capacity_diagnosis_20260910','bet_size_probe_20260910','bet_size_realism_20260910']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert (E/'bet_size_realism_20260910/summary.json').exists()
    out=E/'capacity-and-sizing-20260910.zip';d=E/'capacity-and-sizing-20260910';assert not out.exists();d.mkdir(exist_ok=False);files={}
    for name in FOLDERS:
        for p in sorted((E/name).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:files[p.relative_to(E).as_posix()]=p
    for name in ['STATUS.md','research/POKER_RESEARCH_MEMORY.md','research/POKER_AUGMENTATION_CONTINUATION.md']:files['notes/'+name]=W/name
    files['archive_capacity_and_sizing.py']=Path(__file__)
    manifest={'at':datetime.now(timezone.utc).isoformat(),'files':{n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in files.items()}};(d/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for n,p in files.items():z.write(p,n)
        z.write(d/'manifest.json','manifest.json')
    with zipfile.ZipFile(out) as z:
        assert len(z.namelist())==len(files)+1 and z.testzip() is None
        for n,v in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
        assert hashlib.sha256(z.read('manifest.json')).hexdigest()==sha(d/'manifest.json')
    receipt={'at':datetime.now(timezone.utc).isoformat(),'archive':str(out.resolve()),'bytes':out.stat().st_size,'sha256':sha(out),'verified_members':len(files)+1,'all_members_verified':True,'manifest_sha256':sha(d/'manifest.json')};(d/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
