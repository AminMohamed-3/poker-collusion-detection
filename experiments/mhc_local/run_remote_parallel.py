"""Three independent seeds in flight; predictive recipe and files are unchanged."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'plan.json').read_text())
plan_sha=hashlib.sha256((HERE/'plan.json').read_bytes()).hexdigest()
def run(variant,seed):
    dest=HERE/'fold0'/f'{variant}-{seed}'
    if not dest.exists():
        command=[sys.executable,'-u',str(HERE/'train.py'),'--job','fold0','--variant',variant,'--seed',str(seed),'--device','cuda']
        with (dest.with_suffix('.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    r=json.loads((dest/'receipt.json').read_text())
    assert r['plan_sha256']==plan_sha
    for name,key in [('model.pt','model_sha256'),('predictions.npz','predictions_sha256')]:
        assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==r[key]
    return {'variant':variant,'seed':seed,'verified_at':datetime.now(timezone.utc).isoformat()}

completed=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    for variant in plan['variants']:
        print('Starting fixed seeds for '+variant,flush=True)
        for future in as_completed([pool.submit(run,variant,seed) for seed in plan['seeds']]):
            completed.append(future.result())
            (HERE/'fold0/queue.json').write_text(json.dumps({'completed':completed,'total':12,'concurrency':3},indent=2))
            print(json.dumps({'completed':len(completed),'total':12,'latest':completed[-1]}),flush=True)
assert len(completed)==12
