"""Verified snapshot of a completed local research sequence, excluding bytecode."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile
E=Path(__file__).resolve().parent;W=E.parents[2]
FOLDERS=['softplay_mismatch_20260910','softplay_coupling_20260910','softplay_coupling_diagnosis_20260910','softplay_reference_audit_20260910','softplay_context_20260910','softplay_causal_style_20260910','softplay_style_confirmation_20260910','softplay_probability_20260910']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert (E/'softplay_causal_style_20260910/summary.json').exists()
    d=E/'augmentation-continuation-20260910';d.mkdir(exist_ok=False);out=E/'augmentation-continuation-20260910.zip';assert not out.exists()
    files={}
    for folder in FOLDERS:
        for p in sorted((E/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:files[p.relative_to(E).as_posix()]=p
    for n in ['STATUS.md','research/POKER_RESEARCH_MEMORY.md','research/POKER_AUGMENTATION_CONTINUATION.md']:files['notes/'+n]=W/n
    files['archive_augmentation_continuation.py']=Path(__file__)
    manifest={'at':datetime.now(timezone.utc).isoformat(),'files':{n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in files.items()}}
    (d/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for n,p in files.items():z.write(p,n)
        z.write(d/'manifest.json','manifest.json')
    with zipfile.ZipFile(out) as z:
        assert len(z.namelist())==len(files)+1 and z.testzip() is None
        for n,v in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
        assert hashlib.sha256(z.read('manifest.json')).hexdigest()==sha(d/'manifest.json')
    receipt={'archive':str(out.resolve()),'bytes':out.stat().st_size,'sha256':sha(out),'verified_members':len(files)+1,'all_members_verified':True,'manifest_sha256':sha(d/'manifest.json')};(d/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
