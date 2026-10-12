"""Bundle completed private research results with a transfer digest."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
p=argparse.ArgumentParser(); p.add_argument('--family',choices=['mil','tabpfn'],required=True); p.add_argument('--job',required=True); args=p.parse_args()
assert args.job.replace('_','').isalnum()
base=Path('/workspace/poker-research'); root=base/(args.family+'-'+args.job+'-v1')
summary=json.loads((root/'results/summary.json').read_text())
assert summary['fits']==(6 if args.family=='mil' else 2)
archive=base/(args.family+'-'+args.job+'-results.tar.gz')
with tarfile.open(archive,'w:gz') as t:
    for f in sorted((root/'results').iterdir()): t.add(f,arcname=f.name)
    for name in ['worker-plan.json','run.log']: t.add(root/name,arcname=name)
    if (root/'serialization-equivalence.json').exists(): t.add(root/'serialization-equivalence.json',arcname='serialization-equivalence.json')
receipt={'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'archive':str(archive)}
print(json.dumps(receipt))
