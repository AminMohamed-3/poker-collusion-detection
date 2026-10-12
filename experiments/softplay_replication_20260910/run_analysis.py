"""Audit each generated batch then freeze, fit and evaluate once."""
from pathlib import Path
import time,subprocess,sys,json,hashlib
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(script,args,logname):
    with (H/logname).open('w') as log:subprocess.run([sys.executable,str(H/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
    plan=json.loads((H/'study-plan.json').read_text());freeze=json.loads((H/'analysis-freeze.json').read_text())
    assert sha(H/'study-plan.json')==freeze['study_plan_sha256']
    assert all(sha(H/k)==v for k,v in freeze['sources'].items())
    run('test_statistics.py',[],'statistics-tests.log');run('test_randomness.py',[],'randomness-tests.log')
    for mode in plan['generator_arms']:
        start=time.monotonic()
        while not (H/mode/'pilot-generation.json').exists():
            if time.monotonic()-start>1200:raise TimeoutError(mode+' generation not ready; inspect logs')
            time.sleep(5)
        for script,logname in [('audit_batch.py',mode+'-audit.log'),('replay_batch.py',mode+'-replay.log'),('prepare_features.py',mode+'-features.log')]:run(script,[mode],logname)
        print(mode+' audit/replay/features complete',flush=True)
    run('diagnostics.py',[],'diagnostics.log');print('Paired diagnostics complete',flush=True)
    run('prepare_trial.py',[],'preparation.log');run('train_trial.py',[],'training.log');print('All 24 detector fits frozen',flush=True)
    run('evaluate_trial.py',[],'evaluation.log');print('All comparisons evaluated',flush=True)
if __name__=='__main__':main()
