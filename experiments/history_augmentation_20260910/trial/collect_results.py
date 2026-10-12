from pathlib import Path
import json,hashlib,tarfile
H=Path(__file__).resolve().parent
files=[p for p in H.rglob('*') if p.is_file() and p.name not in ['package.tar.gz','returned-manifest.json'] and '__pycache__' not in p.parts]
# Original training inputs are already safely local; preserve all new models,
# predictions, logs, manifests and code in the return archive.
files=[p for p in files if p.name!='train-arrays.npz']
m={p.relative_to(H).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(H/'returned-manifest.json').write_text(json.dumps(m,indent=2))
with tarfile.open(H.parent/'trial-results.tar.gz','w:gz') as t:
    for p in files+[H/'returned-manifest.json']:t.add(p,arcname=p.relative_to(H))
print('RESULTS_ARCHIVED',flush=True)
