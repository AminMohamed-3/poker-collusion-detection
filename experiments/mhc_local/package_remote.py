"""Export only the frozen training package, without held labels."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
files = ['model.py','prepare.py','train.py','evaluate.py','test_model.py','plan.json',
         'run_local.py','runtime_check.py','fold0/train-arrays.npz','fold0/data-audit.json']
manifest = {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in files}
(HERE/'remote-package-manifest.json').write_text(json.dumps(manifest,indent=2))
archive = HERE/'remote-package.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in files + ['remote-package-manifest.json']:
        tar.add(HERE/p,arcname=p)
receipt = {'files':manifest,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
           'bytes':archive.stat().st_size,'held_labels_in_package':False,
           'reason':'User requested a new remote GPU; full comparison restarted before held outcome evaluation.'}
(HERE/'remote-transfer.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
