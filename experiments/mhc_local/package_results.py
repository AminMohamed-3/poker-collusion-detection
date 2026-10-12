from pathlib import Path
import hashlib,json,tarfile
r=Path('/workspace/mhc-20260910')
queue=json.loads((r/'fold0/queue.json').read_text());assert len(queue['completed'])==12
files=[]
for s in queue['completed']:
    d=r/'fold0'/f"{s['variant']}-{s['seed']}";rc=json.loads((d/'receipt.json').read_text())
    assert hashlib.sha256((d/'model.pt').read_bytes()).hexdigest()==rc['model_sha256']
    assert hashlib.sha256((d/'predictions.npz').read_bytes()).hexdigest()==rc['predictions_sha256']
    files.extend(p for p in d.iterdir() if p.is_file())
files.extend((r/'fold0').glob('*.log'))
files.extend(r/n for n in ['fold0/queue.json','remote-runtime.json','runtime-check.json','tests.json','worker.log'])
manifest={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(r/'results-manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(r/'remote-results.tar.gz','w:gz') as t:
    for p in files+[r/'results-manifest.json']:t.add(p,arcname=p.relative_to(r))
print(json.dumps({'sha256':hashlib.sha256((r/'remote-results.tar.gz').read_bytes()).hexdigest(),'bytes':(r/'remote-results.tar.gz').stat().st_size,'files':len(files)}))
