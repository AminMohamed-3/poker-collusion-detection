"""Complete fixed audits and fits when each paired generation batch is ready."""
from pathlib import Path
import time,subprocess,sys,json,hashlib
H=Path(__file__).resolve().parent
def run(script,args,logname):
    with (H/logname).open('w') as log:subprocess.run([sys.executable,str(H/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
    for mode in ['baseline','strength','learned']:
        start=time.monotonic()
        while not (H/mode/'pilot-generation.json').exists():
            if time.monotonic()-start>900:raise TimeoutError(mode+' generation not ready; inspect logs')
            time.sleep(5)
        for script,output,logname in [('audit_batch.py','pilot-audit.json',mode+'-audit.log'),('replay_batch.py','intervention-replay-audit.json',mode+'-replay.log'),('prepare_features.py','features/metadata.json',mode+'-features.log')]:
            if not (H/mode/output).exists():run(script,[mode],logname)
        print(mode+' audit/replay/features complete',flush=True)
    run('diagnostics.py',[],'diagnostics.log');print('Paired diagnostics complete',flush=True)
    run('prepare_trial.py',[],'preparation.log');run('train_trial.py',[],'training.log');print('All six detector fits frozen',flush=True)
    run('evaluate_trial.py',[],'evaluation.log');print('All six arms evaluated',flush=True)
if __name__=='__main__':main()
