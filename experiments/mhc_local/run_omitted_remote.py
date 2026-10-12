"""Run exactly the two admitted mHC variants on three pre-existing omitted splits."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
H=Path(__file__).resolve().parent
manifest=json.loads((H/'omitted-manifest.json').read_text())
assert all(hashlib.sha256((H/n).read_bytes()).hexdigest()==v for n,v in manifest.items())
plan=json.loads((H/'plan.json').read_text());jobs=['omit_directed_transfer','omit_soft_play','omit_coordinated_isolation']
def run(job,variant,seed):
    d=H/job/f'{variant}-{seed}'
    assert not (H/job/'held-metadata.json').exists()
    if not d.exists():
        with d.with_suffix('.log').open('w') as f:subprocess.run([sys.executable,'-u',str(H/'train.py'),'--job',job,'--variant',variant,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True)
    r=json.loads((d/'receipt.json').read_text());assert r['plan_sha256']==hashlib.sha256((H/'plan.json').read_bytes()).hexdigest()
    for n,k in [('model.pt','model_sha256'),('predictions.npz','predictions_sha256')]:assert hashlib.sha256((d/n).read_bytes()).hexdigest()==r[k]
    return {'job':job,'variant':variant,'seed':seed,'verified_at':datetime.now(timezone.utc).isoformat()}
done=[]
with ThreadPoolExecutor(max_workers=3) as ex:
    for job in jobs:
        for variant in ['mhc','single_pass']:
            for future in as_completed([ex.submit(run,job,variant,s) for s in plan['seeds']]):
                done.append(future.result());(H/'omitted-queue.json').write_text(json.dumps({'completed':done,'total':18},indent=2));print(json.dumps({'completed':len(done),'latest':done[-1]}),flush=True)
assert len(done)==18
