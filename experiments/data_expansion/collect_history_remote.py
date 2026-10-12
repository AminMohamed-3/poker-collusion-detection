from pathlib import Path
import hashlib,json,tarfile
H=Path('/workspace/data-expansion-20260910/history_repair')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
g=json.loads((H/'pilot-generation.json').read_text());assert len(g['pools'])==32
for r in g['pools']:
    d=H/'pilot'/f"pool-{r['pool_id']:04d}";assert all(sha(d/n)==v for n,v in r['files'].items())
files=[p for p in (H/'pilot').rglob('*') if p.is_file()]+[H/n for n in ['pilot-generation.json','generator-tests.json','generation.log']]
m={p.relative_to(H).as_posix():sha(p) for p in files};(H/'results-manifest.json').write_text(json.dumps(m,indent=2))
with tarfile.open(H/'results.tar.gz','w:gz') as t:
    for p in files+[H/'results-manifest.json']:t.add(p,arcname=p.relative_to(H))
print(json.dumps({'sha256':sha(H/'results.tar.gz'),'bytes':(H/'results.tar.gz').stat().st_size}))
