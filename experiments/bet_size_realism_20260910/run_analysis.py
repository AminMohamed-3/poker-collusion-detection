"""Development realism and legal audits only; never loads outer-held labels."""
from pathlib import Path
import json,sys,subprocess,time,hashlib
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(script,args,logname):
    with (H/logname).open('w') as log:subprocess.run([sys.executable,str(H/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
    plan=json.loads((H/'study-plan.json').read_text());assert (H/'causal-style-tests.json').exists()
    for arm in plan['generator_arms']:
        began=time.monotonic()
        while not (H/arm/'pilot-generation.json').exists():
            if time.monotonic()-began>1000:raise TimeoutError(arm)
            time.sleep(5)
        for script,name in [('audit_batch.py','audit'),('replay_batch.py','replay'),('prepare_features.py','features')]:run(script,[arm],arm+'-'+name+'.log')
        print(arm+' audits/features complete',flush=True)
    run('diagnostics.py',[],'diagnostics.log');print('Training-only realism diagnosed',flush=True)
    assert all(sha(H.parents[1]/p)==v for p,v in plan['protected'].items())
if __name__=='__main__':main()
