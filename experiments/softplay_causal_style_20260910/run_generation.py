"""Sequential paired batches, at most two local generation workers."""
from pathlib import Path
import subprocess,sys,json,hashlib,time
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());start=time.perf_counter()
    for arm in plan['generator_arms']:
        assert sha(H/arm/'pilot-plan.json')==plan['generator_plan_sha256'][arm]
        print('Generating '+arm,flush=True)
        with (H/(arm+'-generation.log')).open('w') as log:subprocess.run([sys.executable,str(H/arm/'run_pilot.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed '+arm,flush=True)
    (H/'generation-time.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'local_workers':2},indent=2))
if __name__=='__main__':main()
