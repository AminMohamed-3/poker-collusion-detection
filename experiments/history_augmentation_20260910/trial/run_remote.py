from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((H/'remote-manifest.json').read_text());assert all(sha(H/n)==v for n,v in m.items())
p=json.loads((H/'plan.json').read_text());assert not (H/'held-metadata.json').exists()
def run(arm,variant,seed):
    d=H/arm/f'{variant}-{seed}'
    if not d.exists():
        with d.with_suffix('.log').open('w') as f:subprocess.run([sys.executable,'-u',str(H/'train.py'),'--job',arm,'--variant',variant,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True)
    r=json.loads((d/'receipt.json').read_text());assert r['plan_sha256']==sha(H/'plan.json')
    assert sha(d/'model.pt')==r['model_sha256'] and sha(d/'predictions.npz')==r['predictions_sha256']
    return {'arm':arm,'variant':variant,'seed':seed,'verified_at':datetime.now(timezone.utc).isoformat()}
done=[]
with ThreadPoolExecutor(max_workers=3) as ex:
    for variant in p['variants']:
        for arm in p['arms']:
            for future in as_completed([ex.submit(run,arm,variant,s) for s in p['seeds']]):
                done.append(future.result());(H/'queue.json').write_text(json.dumps({'completed':done,'total':len(p['variants'])*len(p['arms'])*len(p['seeds'])},indent=2));print(json.dumps({'completed':len(done),'latest':done[-1]}),flush=True)
assert len(done)==len(p['variants'])*len(p['arms'])*len(p['seeds'])
