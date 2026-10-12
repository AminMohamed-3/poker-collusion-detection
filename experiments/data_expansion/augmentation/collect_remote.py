from pathlib import Path
import hashlib,json,tarfile
H=Path('/workspace/poker-augmentation-20260910');queue=json.loads((H/'queue.json').read_text());assert len(queue['completed'])==18
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[]
for r in queue['completed']:
    d=H/r['arm']/f"{r['variant']}-{r['seed']}";receipt=json.loads((d/'receipt.json').read_text())
    assert sha(d/'model.pt')==receipt['model_sha256'] and sha(d/'predictions.npz')==receipt['predictions_sha256']
    files.extend(p for p in d.iterdir() if p.is_file());files.append(d.with_suffix('.log'))
files.extend(H/n for n in ['queue.json','worker.log'])
m={p.relative_to(H).as_posix():sha(p) for p in files};(H/'results-manifest.json').write_text(json.dumps(m,indent=2))
with tarfile.open(H/'results.tar.gz','w:gz') as t:
    for p in files+[H/'results-manifest.json']:t.add(p,arcname=p.relative_to(H))
print(json.dumps({'sha256':sha(H/'results.tar.gz'),'bytes':(H/'results.tar.gz').stat().st_size}))
